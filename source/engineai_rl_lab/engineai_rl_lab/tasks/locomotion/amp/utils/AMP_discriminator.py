from __future__ import annotations

import torch
import torch.nn as nn

from dataclasses import dataclass, field
from rsl_rl.modules import MLP, EmpiricalNormalization
from rsl_rl.utils import resolve_nn_activation

class Discriminator(nn.Module):
    """
    Simple feedforward neural network as a discriminator for Adversarial Motion Prior.
    """

    def __init__(
        self,
        input_dim_per_frame: int = 58,
        input_history_length: int = 1,
        condition_dim: int = 0,
        hidden_dims: list[int] = [256, 128],
        activation: str = "relu",
        feature_normalization: bool = False,  # if True, normalize input features with EmpiricalNormalization
        device: str = "cpu"
    ):
        super().__init__()

        self.device = device
        curr_in_dim = input_dim_per_frame * input_history_length + condition_dim
        print("Discriminator input dim:", curr_in_dim)
        self.frame_size = input_dim_per_frame
        self.history_length = input_history_length
        self.condition_dim = condition_dim
        
        self.activation = resolve_nn_activation(activation) #type: ignore
        
        layers = []
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(curr_in_dim, hidden_dim))
            layers.append(self.activation) # use resolved activation
            curr_in_dim = hidden_dim
        self.model = nn.Sequential(*layers).to(self.device) # type: ignore
        self.linear_layer = nn.Linear(hidden_dims[-1], 1).to(self.device)

        self.feature_normalization = feature_normalization
        if self.feature_normalization:
            self.feature_norm = EmpiricalNormalization(shape=(self.frame_size,)).to(self.device)

    def normalize_input(self, x):
        if self.feature_normalization:
            feature_end = self.history_length * self.frame_size
            feature_input = x[:, :feature_end]
            condition = x[:, feature_end:]
            # Normalize physical frame slices with shared statistics. Keep the
            # command condition in its original velocity units.
            frames = torch.split(feature_input, self.frame_size, dim=1)
            norm_frames = [self.feature_norm(frame) for frame in frames]
            x = torch.cat([*norm_frames, condition], dim=1)
        return x

    def forward(self, x):
        expected_size = self.history_length * self.frame_size + self.condition_dim
        assert expected_size == x.shape[1], \
            f"Input feature dimension {x.shape[1]} does not match expected size {expected_size}"
        if self.feature_normalization:
            x = self.normalize_input(x)
        return self.linear_layer(self.model(x)).squeeze(-1)

    def update_normalization(self, x):
        if self.feature_normalization:
            for i in range(self.history_length):
                start_idx = i * self.frame_size
                end_idx = (i + 1) * self.frame_size
                self.feature_norm.update(x[:, start_idx:end_idx])

    @torch.no_grad()
    def get_amp_reward(self, x):
        self.eval()
        d = self.forward(x)
        r = torch.clamp(1 - 0.25 * (d - 1) ** 2, min=0)
        self.train()
        return r

    def compute_grad_pen(self, expert_data, lambda_=10):
        feature_end = self.history_length * self.frame_size
        physical_features = expert_data[:, :feature_end].detach().requires_grad_(True)
        condition = expert_data[:, feature_end:].detach()
        discriminator_input = torch.cat([physical_features, condition], dim=1)
        disc = self.forward(discriminator_input)
        ones = torch.ones(disc.size(), device=disc.device)
        grad = torch.autograd.grad(
            outputs=disc, inputs=physical_features,
            grad_outputs=ones, create_graph=True,
            retain_graph=True, only_inputs=True)[0]

        # Enforce that the grad norm approaches 0.
        grad_pen = lambda_ * (grad.norm(2, dim=1) - 0).pow(2).mean()
        return grad_pen
    
