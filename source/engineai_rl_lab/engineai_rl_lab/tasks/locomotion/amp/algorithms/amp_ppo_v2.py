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


class AMPLossType(Enum):
    """Supported adversarial objectives for the AMP discriminator."""

    GAN = 0
    LSGAN = 1
    WGAN = 2


_LOSS_TYPE_MAP = {"GAN": AMPLossType.GAN, "LSGAN": AMPLossType.LSGAN, "WGAN": AMPLossType.WGAN}


def _resolve_step_dt(env: VecEnv, cfg: dict) -> float:
    """Best-effort resolution of the environment control step (used to scale the style reward)."""
    if cfg.get("step_dt") is not None:
        return cfg["step_dt"]
    unwrapped = getattr(env, "unwrapped", None)
    if unwrapped is not None and hasattr(unwrapped, "step_dt"):
        return unwrapped.step_dt
    inner = getattr(env, "env", None)
    if inner is not None:
        inner_unwrapped = getattr(inner, "unwrapped", inner)
        if hasattr(inner_unwrapped, "step_dt"):
            return inner_unwrapped.step_dt
    return 1.0


class AMPPPOV2(PPO):
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
        # AMP parameters
        discriminator: Discriminator | None = None,
        data_loader: AMPDataLoader | None = None,
        amp_obs_group: str = "amp",
        amp_condition_group: str = "amp_command",
        condition_dim: int = 0,
        style_reward_scale: float = 2.0,
        task_style_lerp: float = 0.5,
        loss_type: str = "LSGAN",
        grad_penalty_scale: float = 10.0,
        disc_learning_rate: float = 1.0e-4,
        disc_trunk_weight_decay: float = 1.0e-4,
        disc_linear_weight_decay: float = 1.0e-2,
        disc_max_grad_norm: float = 1.0,
        disc_update_interval: int = 1,
        step_dt: float = 1.0,
        # RND parameters
        rnd_cfg: dict | None = None,
        # Symmetry parameters
        symmetry_cfg: dict | None = None,
        # Distributed training parameters
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
            raise ValueError("A discriminator must be provided for AMPPPOV2.")
        if data_loader is None:
            raise ValueError("A reference data loader must be provided for AMPPPOV2.")
        if loss_type not in _LOSS_TYPE_MAP:
            raise ValueError(f"Unknown AMP loss type '{loss_type}'. Should be 'GAN', 'LSGAN', or 'WGAN'.")

        self.discriminator = discriminator.to(self.device)
        self.data_loader = data_loader
        self.amp_obs_group = amp_obs_group
        self.amp_condition_group = amp_condition_group
        self.condition_dim = condition_dim
        self.loss_type = _LOSS_TYPE_MAP[loss_type]

        # Reward shaping parameters
        self.style_reward_scale = style_reward_scale
        self.task_style_lerp = task_style_lerp
        self.step_dt = step_dt

        # Discriminator training parameters
        self.grad_penalty_scale = grad_penalty_scale
        self.disc_max_grad_norm = disc_max_grad_norm
        self.disc_update_interval = max(1, int(disc_update_interval))

        # WGAN reward normalizer (kept in eval mode; updated explicitly during disc updates).
        if self.loss_type == AMPLossType.WGAN:
            self.disc_output_normalizer = EmpiricalNormalization(shape=(1,)).to(self.device)
            self.disc_output_normalizer.eval()
        else:
            self.disc_output_normalizer = torch.nn.Identity()

        # Dedicated discriminator optimizer with per-group weight decay (trunk vs. output layer).
        disc_param_groups = [
            {
                "name": "disc_trunk",
                "params": self.discriminator.model.parameters(),
                "weight_decay": disc_trunk_weight_decay,
            },
            {
                "name": "disc_linear",
                "params": self.discriminator.linear_layer.parameters(),
                "weight_decay": disc_linear_weight_decay,
            },
        ]
        self.disc_optimizer = optim.Adam(disc_param_groups, lr=disc_learning_rate)

        print(
            f"[AMPPPOV2] loss_type={loss_type} style_reward_scale={style_reward_scale} "
            f"task_style_lerp={task_style_lerp} disc_update_interval={self.disc_update_interval} "
            f"grad_penalty_scale={grad_penalty_scale} disc_lr={disc_learning_rate} step_dt={step_dt}"
        )

    # ------------------------------------------------------------------ #
    # Reward computation
    # ------------------------------------------------------------------ #
    def _style_reward_from_score(self, disc_score: torch.Tensor) -> torch.Tensor:
        """Map a discriminator score to a style reward, per the configured adversarial objective."""
        if self.loss_type == AMPLossType.GAN:
            prob = torch.sigmoid(disc_score)
            return -torch.log(torch.clamp(1.0 - prob, min=1e-6))
        if self.loss_type == AMPLossType.LSGAN:
            # The clipped quadratic becomes exactly zero when a strong
            # discriminator drives policy scores to -1, eliminating the style
            # learning signal. An exponential kernel preserves the same
            # optimum at +1 while retaining a useful gradient everywhere.
            return torch.exp(-0.25 * torch.square(disc_score - 1.0))
        # WGAN: reward is the (normalized) raw score.
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
            disc_score = self.discriminator(amp_obs)  # [num_envs]
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

        # Per-step logging (guarded; Isaac Lab environments provide the 'log' dict).
        if "log" in extras:
            extras["log"]["Step_Reward/style_reward"] = style_reward.mean()
            extras["log"]["Step_Reward/task_reward"] = task_reward.mean()

    # ------------------------------------------------------------------ #
    # Discriminator objective
    # ------------------------------------------------------------------ #
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
        # WGAN
        return -torch.mean(expert_score) + torch.mean(policy_score)

    # ------------------------------------------------------------------ #
    # Update (PPO + discriminator, integrated in a single minibatch loop)
    # ------------------------------------------------------------------ #
    def update(self) -> dict[str, float]:  # noqa: C901
        """Run PPO optimization and the AMP discriminator update over stored batches."""
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_rnd_loss = 0.0 if self.rnd else None
        mean_symmetry_loss = 0.0 if self.symmetry else None
        # AMP discriminator statistics
        mean_disc_loss = 0.0
        mean_disc_grad_penalty = 0.0
        mean_policy_score = 0.0
        mean_expert_score = 0.0

        # PPO minibatches and, for legacy unconditional AMP, an aligned expert iterator.
        if self.actor.is_recurrent or self.critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        reference_iterator = None
        if self.condition_dim == 0:
            reference_iterator = iter(
                self.data_loader.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
            )
        conditional_reference_batch_size = math.ceil(
            self.data_loader.time_step_total / self.num_mini_batches
        )

        mini_batch_idx = 0
        disc_updates_done = 0
        for batch in generator:
            original_batch_size = batch.observations.batch_size[0]

            # Capture the agent's discriminator observation before any symmetry augmentation.
            amp_policy_features = batch.observations[self.amp_obs_group]
            if self.condition_dim > 0:
                amp_condition = batch.observations[self.amp_condition_group]
                supported = self.data_loader.condition_support_mask(amp_condition)
                amp_condition = amp_condition[supported]
                amp_policy_batch = torch.cat(
                    [amp_policy_features[supported], amp_condition],
                    dim=-1,
                )
                expert_batch = self.data_loader.sample_conditioned(
                    amp_condition,
                    max_samples=conditional_reference_batch_size,
                )
            else:
                amp_policy_batch = amp_policy_features
                expert_batch = next(reference_iterator)

            # Normalize advantages per minibatch if requested.
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)

            # Symmetry data augmentation.
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

            # Recompute log-probs, values and entropy under the current policy.
            self.actor(
                batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[0], stochastic_output=True
            )
            actions_log_prob = self.actor.get_output_log_prob(batch.actions)
            values = self.critic(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1])
            distribution_params = tuple(p[:original_batch_size] for p in self.actor.output_distribution_params)
            entropy = self.actor.output_entropy[:original_batch_size]

            # Adaptive learning rate via KL divergence.
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

            # Surrogate (policy) loss.
            ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))
            surrogate = -torch.squeeze(batch.advantages) * ratio
            surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value loss.
            if self.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
                value_losses = (values - batch.returns).pow(2)
                value_losses_clipped = (value_clipped - batch.returns).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (batch.returns - values).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()

            # Symmetry mirror loss.
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

            # RND loss.
            if self.rnd:
                with torch.no_grad():
                    rnd_state = self.rnd.get_rnd_state(batch.observations[:original_batch_size])
                    rnd_state = self.rnd.state_normalizer(rnd_state)
                predicted_embedding = self.rnd.predictor(rnd_state)
                target_embedding = self.rnd.target(rnd_state).detach()
                mseloss = torch.nn.MSELoss()
                rnd_loss = mseloss(predicted_embedding, target_embedding)

            # AMP discriminator scores (computed every minibatch for saturation logging).
            policy_score = self.discriminator(amp_policy_batch)
            expert_score = self.discriminator(expert_batch)
            disc_loss = self._discriminator_loss(policy_score, expert_score)

            do_disc_update = (mini_batch_idx % self.disc_update_interval == 0)
            if do_disc_update:
                disc_grad_penalty = self.discriminator.compute_grad_pen(expert_batch, lambda_=self.grad_penalty_scale)
                disc_total_loss = disc_loss + disc_grad_penalty
            else:
                disc_grad_penalty = None
                disc_total_loss = None

            # --- Gradients ---
            self.optimizer.zero_grad()
            loss.backward()
            if self.rnd:
                self.rnd_optimizer.zero_grad()
                rnd_loss.backward()
            if do_disc_update:
                self.disc_optimizer.zero_grad()
                disc_total_loss.backward()

            if self.is_multi_gpu:
                self.reduce_parameters()

            # --- Optimizer steps ---
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            if self.rnd_optimizer:
                self.rnd_optimizer.step()
            if do_disc_update:
                nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.disc_max_grad_norm)
                self.disc_optimizer.step()
                # Update discriminator input normalization and (for WGAN) reward normalization.
                self.discriminator.update_normalization(amp_policy_batch.detach())
                self.discriminator.update_normalization(expert_batch.detach())
                if self.loss_type == AMPLossType.WGAN:
                    with torch.no_grad():
                        self.disc_output_normalizer.update(policy_score.detach().unsqueeze(-1))

            # --- Bookkeeping ---
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.mean().item()
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()
            mean_policy_score += policy_score.mean().item()
            mean_expert_score += expert_score.mean().item()
            if do_disc_update:
                mean_disc_loss += disc_loss.item()
                mean_disc_grad_penalty += disc_grad_penalty.item()
                disc_updates_done += 1

            mini_batch_idx += 1

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates
        mean_policy_score /= num_updates
        mean_expert_score /= num_updates
        if disc_updates_done > 0:
            mean_disc_loss /= disc_updates_done
            mean_disc_grad_penalty /= disc_updates_done

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
        loss_dict["amp/disc_loss"] = mean_disc_loss
        loss_dict["amp/disc_grad_penalty"] = mean_disc_grad_penalty
        loss_dict["amp/disc_policy_score"] = mean_policy_score
        loss_dict["amp/disc_expert_score"] = mean_expert_score
        return loss_dict

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> "AMPPPOV2":
        """Build the discriminator + data loader and inject AMP parameters, then defer to PPO."""
        alg_cfg = cfg["algorithm"]

        # Discriminator accepts either flattened or [history, feature] observations.
        alg_cfg["discriminator"] = Discriminator(
            input_dim_per_frame=cfg["frame_dim"],
            input_history_length=cfg["frame_length"],
            condition_dim=cfg.get("condition_dim", 0),
            hidden_dims=cfg["discriminator_hidden_dims"],
            feature_normalization=cfg.get("frame_normalization", True),
            device=device,
        ).to(device)

        # Offline reference motion data.
        alg_cfg["data_loader"] = AMPDataLoader(
            cfg["dataset_path"],
            history_length=cfg["frame_length"],
            include_joint_vel=cfg.get("include_joint_vel", False),
            include_base_lin_vel=cfg.get("include_base_lin_vel", True),
            include_projected_gravity=cfg.get("include_projected_gravity", True),
            include_foot_features=cfg.get("include_foot_features", False),
            flatten_history_dim=cfg.get("flatten_history_dim", True),
            condition_dim=cfg.get("condition_dim", 0),
            device=device,
        )

        # AMP hyperparameters (defaults keep the runner cfg minimal; override in the agent cfg).
        alg_cfg["amp_obs_group"] = cfg.get("amp_obs_group", "amp")
        alg_cfg["amp_condition_group"] = cfg.get("amp_condition_group", "amp_command")
        alg_cfg["condition_dim"] = cfg.get("condition_dim", 0)
        # ``style_reward_scale`` falls back to the legacy ``style_reward_weight`` if not set.
        alg_cfg["style_reward_scale"] = cfg.get("style_reward_scale", cfg.get("style_reward_weight", 2.0))
        alg_cfg["task_style_lerp"] = cfg.get("task_style_lerp", 0.5)
        alg_cfg["loss_type"] = cfg.get("loss_type", "LSGAN")
        alg_cfg["grad_penalty_scale"] = cfg.get("grad_penalty_scale", 10.0)
        alg_cfg["disc_learning_rate"] = cfg.get("disc_learning_rate", 1.0e-4)
        alg_cfg["disc_trunk_weight_decay"] = cfg.get("disc_trunk_weight_decay", 1.0e-4)
        alg_cfg["disc_linear_weight_decay"] = cfg.get("disc_linear_weight_decay", 1.0e-2)
        alg_cfg["disc_max_grad_norm"] = cfg.get("disc_max_grad_norm", 1.0)
        alg_cfg["disc_update_interval"] = cfg.get("disc_update_interval", 1)
        alg_cfg["step_dt"] = _resolve_step_dt(env, cfg)

        return PPO.construct_algorithm(obs, env, cfg, device)
