from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.optim as optim
from enum import Enum
from tensordict import TensorDict

from rsl_rl.algorithms import PPO
from rsl_rl.env import VecEnv
from rsl_rl.models import MLPModel
from rsl_rl.modules import EmpiricalNormalization
from rsl_rl.storage import RolloutStorage

from ..utils.AMP_data_loader import AMPDataLoader
from ..utils.AMP_discriminator import Discriminator
from ..utils.disc_replay_buffer import DiscReplayBuffer


class AMPLossType(Enum):
    """Supported adversarial objectives for the AMP discriminator."""

    GAN = 0
    LSGAN = 1
    WGAN = 2


_LOSS_TYPE_MAP = {"GAN": AMPLossType.GAN, "LSGAN": AMPLossType.LSGAN, "WGAN": AMPLossType.WGAN}

# Cap on the samples fed to the discriminator input normalizer per iteration. The policy and
# the expert side are capped equally: MimicKit records both in equal amounts, and using the
# full rollout would both let the policy dominate the statistics and make the matching expert
# draw prohibitively large.
_NORMALIZER_SAMPLE_CAP = 16384


def _resolve_step_dt(env: VecEnv, amp_cfg: dict) -> float:
    """Best-effort resolution of the environment control step (used to scale the style reward)."""
    if amp_cfg.get("step_dt") is not None:
        return amp_cfg["step_dt"]
    unwrapped = getattr(env, "unwrapped", None)
    if unwrapped is not None and hasattr(unwrapped, "step_dt"):
        return unwrapped.step_dt
    inner = getattr(env, "env", None)
    if inner is not None:
        inner_unwrapped = getattr(inner, "unwrapped", inner)
        if hasattr(inner_unwrapped, "step_dt"):
            return inner_unwrapped.step_dt
    return 1.0


class AMP(PPO):
    """Adversarial Motion Priors on top of ``rsl_rl`` PPO, fusing ``legged_lab`` behaviours."""

    discriminator: Discriminator

    def __init__(
        self,
        actor: MLPModel,
        critic: MLPModel,
        storage: RolloutStorage,
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.01,
        learning_rate: float = 0.001,
        max_grad_norm: float = 1.0,
        optimizer: str = "adam",
        use_clipped_value_loss: bool = True,
        schedule: str = "adaptive",
        desired_kl: float = 0.01,
        normalize_advantage_per_mini_batch: bool = False,
        device: str = "cpu",
        discriminator: Discriminator | None = None,
        data_loader: AMPDataLoader | None = None,
        amp_obs_group: str = "amp",
        amp_condition_group: str = "amp_command",
        condition_dim: int = 0,
        style_reward_scale: float = 2.0,
        task_style_lerp: float = 0.5,
        loss_type: str = "GAN",
        grad_penalty_scale: float = 10.0,
        disc_logit_reg: float = 0.01,
        disc_learning_rate: float = 2.5e-4,
        disc_weight_decay: float = 1.0e-4,
        disc_max_grad_norm: float = 1.0,
        disc_epochs: int = 2,
        disc_batch_size: float = 2.0,
        disc_update_interval: int = 2,
        disc_replay_buffer_size: int = 200000,
        disc_replay_samples: int = 1000,
        step_dt: float = 1.0,
        rnd_cfg: dict | None = None,
        symmetry_cfg: dict | None = None,
        multi_gpu_cfg: dict | None = None,
    ) -> None:
        super().__init__(
            actor=actor,
            critic=critic,
            storage=storage,
            num_learning_epochs=num_learning_epochs,
            num_mini_batches=num_mini_batches,
            clip_param=clip_param,
            gamma=gamma,
            lam=lam,
            value_loss_coef=value_loss_coef,
            entropy_coef=entropy_coef,
            learning_rate=learning_rate,
            max_grad_norm=max_grad_norm,
            optimizer=optimizer,
            use_clipped_value_loss=use_clipped_value_loss,
            schedule=schedule,
            desired_kl=desired_kl,
            normalize_advantage_per_mini_batch=normalize_advantage_per_mini_batch,
            device=device,
            rnd_cfg=rnd_cfg,
            symmetry_cfg=symmetry_cfg,
            multi_gpu_cfg=multi_gpu_cfg,
        )

        if discriminator is None:
            raise ValueError("A discriminator must be provided for AMP.")
        if data_loader is None:
            raise ValueError("A reference data loader must be provided for AMP.")
        if loss_type not in _LOSS_TYPE_MAP:
            raise ValueError(f"Unknown AMP loss type '{loss_type}'. Should be 'GAN', 'LSGAN', or 'WGAN'.")

        self.discriminator = discriminator.to(self.device)
        self.data_loader = data_loader
        self.amp_obs_group = amp_obs_group
        self.amp_condition_group = amp_condition_group
        self.condition_dim = condition_dim
        self.loss_type = _LOSS_TYPE_MAP[loss_type]

        self.style_reward_scale = style_reward_scale
        self.task_style_lerp = task_style_lerp
        self.step_dt = step_dt

        self.grad_penalty_scale = grad_penalty_scale
        self.disc_logit_reg = disc_logit_reg
        self.disc_max_grad_norm = disc_max_grad_norm
        self.disc_epochs = max(1, int(disc_epochs))
        self.disc_batch_size = float(disc_batch_size)
        self.disc_update_interval = max(1, int(disc_update_interval))
        self.disc_replay_samples = max(1, int(disc_replay_samples))

        # WGAN reward normalizer (kept in eval mode; updated explicitly during disc updates).
        if self.loss_type == AMPLossType.WGAN:
            self.disc_output_normalizer = EmpiricalNormalization(shape=(1,)).to(self.device)
            self.disc_output_normalizer.eval()
        else:
            self.disc_output_normalizer = torch.nn.Identity()

        # Negative samples are drawn half from the current rollout, half from earlier policies.
        self.disc_replay_buffer = DiscReplayBuffer(disc_replay_buffer_size, device=self.device)

        # Dedicated discriminator optimizer (MimicKit uses SGD with momentum for the disc).
        self.disc_optimizer = optim.SGD(
            self.discriminator.parameters(),
            lr=disc_learning_rate,
            momentum=0.9,
            weight_decay=disc_weight_decay,
        )

        # Iteration counter throttling the discriminator update phase, and the last reported
        # statistics so throttled iterations do not log a sawtooth of zeros.
        self._policy_iteration = 0
        # Numbered so TensorBoard's Loss/amp group sorts: scores, accuracy, then losses.
        self._last_disc_info: dict[str, float] = {
            "amp/1_policy_score": 0.0,
            "amp/2_expert_score": 0.0,
            "amp/3_agent_acc": 0.0,
            "amp/4_demo_acc": 0.0,
            "amp/5_disc_loss": 0.0,
            "amp/6_grad_penalty": 0.0,
        }

        print(
            f"[AMP] loss_type={loss_type} style_reward_scale={style_reward_scale} "
            f"task_style_lerp={task_style_lerp} disc_update_interval={self.disc_update_interval} "
            f"disc_epochs={self.disc_epochs} disc_batch_size={self.disc_batch_size}xnum_envs "
            f"grad_penalty_scale={grad_penalty_scale} disc_logit_reg={disc_logit_reg} "
            f"disc_lr={disc_learning_rate} replay_capacity={disc_replay_buffer_size} "
            f"replay_samples={self.disc_replay_samples} step_dt={step_dt}"
        )

    # Reward computation
    def _style_reward_from_score(self, disc_score: torch.Tensor) -> torch.Tensor:
        """Map a discriminator score to a style reward, per the configured adversarial objective."""
        if self.loss_type == AMPLossType.GAN:
            prob = torch.sigmoid(disc_score)
            # MimicKit's floor: caps the reward at ~9.2 instead of letting it blow up.
            return -torch.log(torch.clamp(1.0 - prob, min=1e-4))
        if self.loss_type == AMPLossType.LSGAN:
            # The clipped quadratic becomes exactly zero when a strong
            # discriminator drives policy scores to -1, eliminating the style
            # learning signal. An exponential kernel preserves the same
            # optimum at +1 while retaining a useful gradient everywhere.
            return torch.exp(-0.25 * torch.square(disc_score - 1.0))
        normed = self.disc_output_normalizer(disc_score.unsqueeze(-1))
        return normed.squeeze(-1)

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        """Blend task and style rewards (lerp) before delegating to the base PPO bookkeeping."""
        amp_obs = obs[self.amp_obs_group]
        condition = None
        if self.condition_dim > 0:
            condition = obs[self.amp_condition_group]

        # Optional terminal-observation correction: at episode boundaries the environment has
        # already reset, so the discriminator observation would come from the fresh episode.
        # If the environment exposes the pre-reset observation, use it for done envs.
        if "terminal_obs" in extras and self.amp_obs_group in extras["terminal_obs"]:
            done_mask = dones.to(dtype=torch.bool)
            if torch.any(done_mask):
                amp_obs = amp_obs.clone()
                amp_obs[done_mask] = extras["terminal_obs"][self.amp_obs_group][done_mask]
                if condition is not None and self.amp_condition_group in extras["terminal_obs"]:
                    condition = condition.clone()
                    condition[done_mask] = extras["terminal_obs"][self.amp_condition_group][done_mask]

        if condition is not None:
            amp_obs = torch.cat([amp_obs, condition], dim=-1)

        # Compute the style reward without polluting the discriminator's running statistics
        # (mirrors Discriminator.get_amp_reward toggling eval/train).
        was_training = self.discriminator.training
        self.discriminator.eval()
        with torch.no_grad():
            disc_score = self.discriminator(amp_obs)
            style_reward = self.step_dt * self.style_reward_scale * self._style_reward_from_score(disc_score)
        if was_training:
            self.discriminator.train()

        task_reward = rewards.clone()
        # Linear interpolation between task and style reward (legged_lab semantics).
        blended_reward = self.task_style_lerp * task_reward + (1.0 - self.task_style_lerp) * style_reward
        if condition is not None:
            supported = self.data_loader.condition_support_mask(condition)
            style_reward = style_reward * supported
            # There is no standing expert clip in the dataset. Do not dilute the
            # task reward for an unsupported condition.
            total_reward = torch.where(supported, blended_reward, task_reward)
        else:
            total_reward = blended_reward

        super().process_env_step(obs, total_reward, dones, extras)

        if "log" in extras:
            extras["log"]["Step_Reward/style_reward"] = style_reward.mean()
            extras["log"]["Step_Reward/task_reward"] = task_reward.mean()

    # Discriminator objective
    def _discriminator_loss(self, policy_score: torch.Tensor, expert_score: torch.Tensor) -> torch.Tensor:
        """Adversarial loss for the discriminator given policy and expert scores."""
        if self.loss_type == AMPLossType.GAN:
            bce = torch.nn.BCEWithLogitsLoss()
            policy_loss = bce(policy_score, torch.zeros_like(policy_score))
            expert_loss = bce(expert_score, torch.ones_like(expert_score))
            return 0.5 * (policy_loss + expert_loss)
        if self.loss_type == AMPLossType.LSGAN:
            mse = torch.nn.MSELoss()
            policy_loss = mse(policy_score, -1.0 * torch.ones_like(policy_score))
            expert_loss = mse(expert_score, torch.ones_like(expert_score))
            return 0.5 * (policy_loss + expert_loss)
        return -torch.mean(expert_score) + torch.mean(policy_score)

    @staticmethod
    def _mean_squared_grad(score: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        """Mean squared gradient norm of ``score`` with respect to ``inputs``."""
        grad = torch.autograd.grad(
            outputs=score,
            inputs=inputs,
            grad_outputs=torch.ones_like(score),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        return grad.square().flatten(start_dim=1).sum(dim=-1).mean()

    def _compute_disc_loss(
        self, agent_batch: torch.Tensor, expert_batch: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Adversarial loss with a two-sided gradient penalty and output-layer regularization.

        Normalization is applied explicitly so the penalty gradient is taken with respect to
        the normalized input, and so a single forward pass per side serves both terms.
        """
        norm_agent = self.discriminator.normalize_input(agent_batch).detach().requires_grad_(True)
        norm_expert = self.discriminator.normalize_input(expert_batch).detach().requires_grad_(True)

        agent_score = self.discriminator.forward_normalized(norm_agent)
        expert_score = self.discriminator.forward_normalized(norm_expert)

        loss = self._discriminator_loss(agent_score, expert_score)

        # Drive the gradient norm towards zero on both sides of the objective.
        grad_penalty = 0.5 * (
            self._mean_squared_grad(expert_score, norm_expert)
            + self._mean_squared_grad(agent_score, norm_agent)
        )
        total_loss = loss + self.grad_penalty_scale * grad_penalty

        if self.disc_logit_reg != 0.0:
            total_loss = total_loss + self.disc_logit_reg * self.discriminator.linear_layer.weight.square().sum()

        info = {
            "amp/1_policy_score": agent_score.detach().mean().item(),
            "amp/2_expert_score": expert_score.detach().mean().item(),
            "amp/3_agent_acc": (agent_score.detach() < 0).float().mean().item(),
            "amp/4_demo_acc": (expert_score.detach() > 0).float().mean().item(),
            "amp/5_disc_loss": loss.detach().item(),
            "amp/6_grad_penalty": grad_penalty.detach().item(),
        }
        return total_loss, info

    # Discriminator update phase (decoupled from the PPO minibatch loop)
    def _collect_rollout_disc_obs(self) -> torch.Tensor:
        """Flatten the whole rollout's discriminator observations into ``[T * num_envs, ...]``."""
        disc_obs = self.storage.observations[self.amp_obs_group].flatten(0, 1)
        if self.condition_dim > 0:
            condition = self.storage.observations[self.amp_condition_group].flatten(0, 1)
            supported = self.data_loader.condition_support_mask(condition)
            disc_obs = torch.cat([disc_obs[supported], condition[supported]], dim=-1)
        return disc_obs

    def _store_disc_replay_data(self, disc_obs: torch.Tensor) -> None:
        """Insert the rollout into the replay buffer, throttled once the buffer is full."""
        num_samples = disc_obs.shape[0]
        if self.disc_replay_buffer.is_full():
            num_samples = min(num_samples, self.disc_replay_samples)
            indices = torch.randperm(disc_obs.shape[0], device=disc_obs.device)[:num_samples]
            self.disc_replay_buffer.push(disc_obs[indices])
        else:
            self.disc_replay_buffer.push(disc_obs)

    def _update_disc_normalization(self, disc_obs: torch.Tensor) -> None:
        """Refresh the discriminator input statistics with equal policy and expert counts."""
        num_samples = min(disc_obs.shape[0], _NORMALIZER_SAMPLE_CAP)
        indices = torch.randperm(disc_obs.shape[0], device=disc_obs.device)[:num_samples]
        self.discriminator.update_normalization(disc_obs[indices].detach())
        self.discriminator.update_normalization(self._sample_expert_batch(disc_obs[indices], num_samples))

    def _sample_expert_batch(self, agent_batch: torch.Tensor, num_samples: int) -> torch.Tensor:
        """Draw ``num_samples`` expert frames, matching the agent commands when conditioned."""
        if self.condition_dim == 0:
            return self.data_loader.sample(num_samples)
        feature_end = self.discriminator.history_length * self.discriminator.frame_size
        condition = agent_batch[:num_samples, feature_end:]
        return self.data_loader.sample_conditioned(condition)

    def _reduce_disc_parameters(self) -> None:
        """Average the discriminator gradients across GPUs (mirrors PPO.reduce_parameters)."""
        params = [p for p in self.discriminator.parameters() if p.grad is not None]
        if not params:
            return
        all_grads = torch.cat([p.grad.view(-1) for p in params])
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size
        offset = 0
        for param in params:
            numel = param.numel()
            param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
            offset += numel

    def _update_discriminator(self) -> dict[str, float]:
        """Run one discriminator update phase over the stored rollout."""
        disc_obs = self._collect_rollout_disc_obs()
        if disc_obs.shape[0] == 0:
            self._policy_iteration += 1
            return dict(self._last_disc_info)

        # Keep the replay buffer and the input statistics fresh on every iteration, even when
        # the gradient steps below are throttled.
        self._store_disc_replay_data(disc_obs)
        self._update_disc_normalization(disc_obs)

        run_update = self._policy_iteration % self.disc_update_interval == 0
        self._policy_iteration += 1
        if not run_update:
            return dict(self._last_disc_info)

        num_samples = disc_obs.shape[0]
        batch_size = max(1, min(int(math.ceil(self.disc_batch_size * self.storage.num_envs)), num_samples))
        num_disc_steps = int(math.ceil(num_samples / batch_size)) * self.disc_epochs

        mean_info = {key: 0.0 for key in self._last_disc_info}
        for _ in range(num_disc_steps):
            indices = torch.randint(0, num_samples, (batch_size,), device=self.device)
            policy_batch = disc_obs[indices]
            replay_batch = self.disc_replay_buffer.sample(policy_batch.shape[0])
            agent_batch = torch.cat([policy_batch, replay_batch], dim=0)
            expert_batch = self._sample_expert_batch(policy_batch, batch_size)

            loss, info = self._compute_disc_loss(agent_batch, expert_batch)

            self.disc_optimizer.zero_grad()
            loss.backward()
            if self.is_multi_gpu:
                self._reduce_disc_parameters()
            nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.disc_max_grad_norm)
            self.disc_optimizer.step()

            for key, value in info.items():
                mean_info[key] += value

            if self.loss_type == AMPLossType.WGAN:
                with torch.no_grad():
                    self.disc_output_normalizer.update(
                        self.discriminator(policy_batch).detach().unsqueeze(-1)
                    )

        mean_info = {key: value / num_disc_steps for key, value in mean_info.items()}
        self._last_disc_info = mean_info
        return mean_info

    # Update (PPO minibatch loop, then the discriminator phase)
    def update(self) -> dict[str, float]:  # noqa: C901
        """Run PPO optimization, then the decoupled AMP discriminator update phase."""
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_rnd_loss = 0.0 if self.rnd else None
        mean_symmetry_loss = 0.0 if self.symmetry else None

        if self.actor.is_recurrent or self.critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for batch in generator:
            original_batch_size = batch.observations.batch_size[0]

            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)

            if self.symmetry and self.symmetry["use_data_augmentation"]:
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                batch.observations, batch.actions = data_augmentation_func(
                    env=self.symmetry["_env"], obs=batch.observations, actions=batch.actions
                )
                num_aug = int(batch.observations.batch_size[0] / original_batch_size)
                batch.old_actions_log_prob = batch.old_actions_log_prob.repeat(num_aug, 1)
                batch.values = batch.values.repeat(num_aug, 1)
                batch.advantages = batch.advantages.repeat(num_aug, 1)
                batch.returns = batch.returns.repeat(num_aug, 1)

            self.actor(
                batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[0], stochastic_output=True
            )
            actions_log_prob = self.actor.get_output_log_prob(batch.actions)
            values = self.critic(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1])
            distribution_params = tuple(p[:original_batch_size] for p in self.actor.output_distribution_params)
            entropy = self.actor.output_entropy[:original_batch_size]

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = self.actor.get_kl_divergence(batch.old_distribution_params, distribution_params)
                    kl_mean = torch.mean(kl)

                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))
            surrogate = -torch.squeeze(batch.advantages) * ratio
            surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
                value_losses = (values - batch.returns).pow(2)
                value_losses_clipped = (value_clipped - batch.returns).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (batch.returns - values).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()

            if self.symmetry:
                if not self.symmetry["use_data_augmentation"]:
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    batch.observations, _ = data_augmentation_func(
                        obs=batch.observations, actions=None, env=self.symmetry["_env"]
                    )
                mean_actions = self.actor(batch.observations.detach().clone())
                action_mean_orig = mean_actions[:original_batch_size]
                _, actions_mean_symm = data_augmentation_func(
                    obs=None, actions=action_mean_orig, env=self.symmetry["_env"]
                )
                mse_loss = torch.nn.MSELoss()
                symmetry_loss = mse_loss(
                    mean_actions[original_batch_size:], actions_mean_symm.detach()[original_batch_size:]
                )
                if self.symmetry["use_mirror_loss"]:
                    loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                else:
                    symmetry_loss = symmetry_loss.detach()

            if self.rnd:
                with torch.no_grad():
                    rnd_state = self.rnd.get_rnd_state(batch.observations[:original_batch_size])
                    rnd_state = self.rnd.state_normalizer(rnd_state)
                predicted_embedding = self.rnd.predictor(rnd_state)
                target_embedding = self.rnd.target(rnd_state).detach()
                mseloss = torch.nn.MSELoss()
                rnd_loss = mseloss(predicted_embedding, target_embedding)

            self.optimizer.zero_grad()
            loss.backward()
            if self.rnd:
                self.rnd_optimizer.zero_grad()
                rnd_loss.backward()

            if self.is_multi_gpu:
                self.reduce_parameters()

            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            if self.rnd_optimizer:
                self.rnd_optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.mean().item()
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates

        # The discriminator reads the rollout straight from the storage, so it must run before
        # the write cursor is reset.
        disc_info = self._update_discriminator()

        self.storage.clear()

        loss_dict = {
            "value": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss
        loss_dict.update(disc_info)
        return loss_dict

    # Construction
    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> "AMP":
        """Build the discriminator + data loader and inject AMP parameters, then defer to PPO."""
        if "amp" not in cfg:
            raise KeyError(
                "The runner configuration has no 'amp' entry. AMP agents must derive from "
                "'engineai_rl_lab.tasks.locomotion.amp.agents.AmpBasePpoRunnerCfg' so that the AMP "
                "hyperparameters of 'RslRlAmpCfg' are available."
            )
        alg_cfg = cfg["algorithm"]
        amp_cfg = cfg["amp"]

        # Discriminator accepts either flattened or [history, feature] observations.
        alg_cfg["discriminator"] = Discriminator(
            input_dim_per_frame=amp_cfg["frame_dim"],
            input_history_length=amp_cfg["frame_length"],
            condition_dim=amp_cfg["condition_dim"],
            hidden_dims=amp_cfg["discriminator_hidden_dims"],
            feature_normalization=amp_cfg["frame_normalization"],
            feature_normalization_clip=amp_cfg["frame_normalization_clip"],
            device=device,
        ).to(device)

        alg_cfg["data_loader"] = AMPDataLoader(
            amp_cfg["dataset_path"],
            history_length=amp_cfg["frame_length"],
            include_joint_vel=amp_cfg["include_joint_vel"],
            include_base_lin_vel=amp_cfg["include_base_lin_vel"],
            include_projected_gravity=amp_cfg["include_projected_gravity"],
            include_foot_features=amp_cfg["include_foot_features"],
            flatten_history_dim=amp_cfg["flatten_history_dim"],
            condition_dim=amp_cfg["condition_dim"],
            device=device,
        )

        # AMP hyperparameters, tunable in the robot's agent config via ``RslRlAmpCfg``.
        alg_cfg["amp_obs_group"] = amp_cfg["amp_obs_group"]
        alg_cfg["amp_condition_group"] = amp_cfg["amp_condition_group"]
        alg_cfg["condition_dim"] = amp_cfg["condition_dim"]
        alg_cfg["style_reward_scale"] = amp_cfg["style_reward_scale"]
        alg_cfg["task_style_lerp"] = amp_cfg["task_style_lerp"]
        alg_cfg["loss_type"] = amp_cfg["loss_type"]
        alg_cfg["grad_penalty_scale"] = amp_cfg["grad_penalty_scale"]
        alg_cfg["disc_logit_reg"] = amp_cfg["disc_logit_reg"]
        alg_cfg["disc_learning_rate"] = amp_cfg["disc_learning_rate"]
        alg_cfg["disc_weight_decay"] = amp_cfg["disc_weight_decay"]
        alg_cfg["disc_max_grad_norm"] = amp_cfg["disc_max_grad_norm"]
        alg_cfg["disc_epochs"] = amp_cfg["disc_epochs"]
        alg_cfg["disc_batch_size"] = amp_cfg["disc_batch_size"]
        alg_cfg["disc_update_interval"] = amp_cfg["disc_update_interval"]
        alg_cfg["disc_replay_buffer_size"] = amp_cfg["disc_replay_buffer_size"]
        alg_cfg["disc_replay_samples"] = amp_cfg["disc_replay_samples"]
        alg_cfg["step_dt"] = _resolve_step_dt(env, amp_cfg)

        return PPO.construct_algorithm(obs, env, cfg, device)
