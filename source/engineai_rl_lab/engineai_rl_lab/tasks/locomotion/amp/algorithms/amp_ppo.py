from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from tensordict import TensorDict

from rsl_rl.env import VecEnv
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from rsl_rl.algorithms import PPO

from ..utils.AMP_data_loader import AMPDataLoader
from ..utils.AMP_discriminator import Discriminator


class AMPPPO(PPO):
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
        #  AMP parameters
        discriminator: MLPModel = None,
        data_loader: AMPDataLoader = None,
        style_reward_weight: float = 2.0,
        # RND parameters
        rnd_cfg: dict | None = None,
        # Symmetry parameters
        symmetry_cfg: dict | None = None,
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
    ) -> None:

        self.style_reward_weight = style_reward_weight
        print(f"Initialized AMPPPO with style reward weight: {self.style_reward_weight}")

        self.discriminator = discriminator
        if self.discriminator is None:
            raise ValueError("Discriminator must be provided for AMPPPO.")
    
        self.discriminator_data_loader = data_loader
        if self.discriminator_data_loader is None:
            raise ValueError("Data loader must be provided for AMPPPO.")
    
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
            # RND parameters
            rnd_cfg=rnd_cfg,
            # Symmetry parameters
            symmetry_cfg=symmetry_cfg,
            # Distributed training parameters
            multi_gpu_cfg=multi_gpu_cfg,
        )

        self.disc_optimizer = optim.Adam(self.discriminator.parameters(), lr=1e-4)

        self.ppo_update_counter = 0

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> AMPPPO:

        cfg["algorithm"]["style_reward_weight"]= cfg["style_reward_weight"]
        cfg["algorithm"]["discriminator"] = Discriminator(
            input_dim_per_frame=cfg["frame_dim"],
            input_history_length=cfg["frame_length"],
            hidden_dims=cfg["discriminator_hidden_dims"],
            feature_normalization=cfg["frame_normalization"],
            device=device
        ).to(device)

        cfg["algorithm"]["data_loader"] = AMPDataLoader(
            cfg["dataset_path"],
            history_length=cfg["frame_length"],
            device=device
        )

        alg:AMPPPO = PPO.construct_algorithm(obs, env, cfg, device)

        return alg

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
    
        with torch.no_grad():
            amp_reward = 0.01*self.style_reward_weight * self.discriminator.get_amp_reward(obs["amp"])
        
        task_reward = rewards.clone()
        total_reward = task_reward + amp_reward

        super().process_env_step(obs, total_reward, dones, extras)
        
        # log the single step reward 
        extras['log']['Step_Reward/style_reward'] = amp_reward
        extras['log']['Step_Reward/task_reward'] = task_reward
        

    def update(self):  # noqa: C901
        loss_dict = {}
        if self.ppo_update_counter % 4 ==0:
            mean_amp_policy_score = 0
            mean_amp_expert_score = 0
            mean_amp_grad_penalty = 0
            mean_amp_loss = 0
            reference_data_generator = self.discriminator_data_loader.mini_batch_generator(self.num_mini_batches//2, self.num_learning_epochs)
            generator = self.storage.mini_batch_generator(self.num_mini_batches//2, self.num_learning_epochs)
            # Iterate over batches
            for (batch,amp_ref_batch) in zip(generator, reference_data_generator):

                amp_policy_batch = batch.observations["amp"]
                
                expert_score = self.discriminator(amp_ref_batch)
                policy_score = self.discriminator(amp_policy_batch)

                expert_loss = torch.nn.MSELoss()(expert_score, torch.ones_like(expert_score))
                policy_loss = torch.nn.MSELoss()(policy_score, -1 * torch.ones_like(policy_score))

                discrim_loss = 0.5 * (expert_loss + policy_loss)

                grad_pen_loss = self.discriminator.compute_grad_pen(amp_ref_batch)

                discrim_total_loss = discrim_loss + grad_pen_loss

                self.disc_optimizer.zero_grad()
                discrim_total_loss.backward()

                nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.max_grad_norm)
                self.disc_optimizer.step()
        
                with torch.no_grad():
                    self.discriminator.update_normalization(amp_ref_batch.detach())
                    self.discriminator.update_normalization(amp_policy_batch.detach())

                mean_amp_loss += discrim_total_loss.item()
                mean_amp_grad_penalty += grad_pen_loss.item()
                mean_amp_expert_score += expert_score.mean().item()
                mean_amp_policy_score += policy_score.mean().item()

            mean_amp_expert_score /= (self.num_mini_batches * self.num_learning_epochs)
            mean_amp_policy_score /= (self.num_mini_batches * self.num_learning_epochs)
            mean_amp_grad_penalty /= (self.num_mini_batches * self.num_learning_epochs)
            mean_amp_loss /= (self.num_mini_batches * self.num_learning_epochs)

            loss_dict.update({
                "discriminator_loss": mean_amp_loss,
                "amp_grad_penalty": mean_amp_grad_penalty,
                "amp_expert_score": mean_amp_expert_score,  
                "amp_policy_score": mean_amp_policy_score,
            })
            self.policy_update_counter = 0
            

        ppo_loss_dict = super().update()
        loss_dict.update(ppo_loss_dict)
        self.ppo_update_counter += 1

        return loss_dict
