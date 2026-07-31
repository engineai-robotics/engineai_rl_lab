from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.commands import (
    UniformVelocityCommand,
    UniformVelocityCommandCfg,
)
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class XYZVelocityCommand(UniformVelocityCommand):
    """Sample mixed and single-axis SE(2) velocity commands.

    Command definition:
        command[:, 0]: linear velocity x
        command[:, 1]: linear velocity y
        command[:, 2]: angular velocity around z

    Command categories:
        standing_ratio: all components are zero
        only_x_ratio:  only vx is non-zero
        only_y_ratio:  only vy is non-zero
        only_z_ratio:  only wz is non-zero
        remaining:     vx, vy and wz are sampled normally

    All ratios are proportions of the complete command distribution.
    """

    cfg: XYZVelocityCommandCfg

    def __init__(self, cfg: XYZVelocityCommandCfg, env: ManagerBasedEnv):
        if cfg.heading_command:
            raise ValueError(
                "XYZVelocityCommand does not support heading_command=True. "
                "Configure ang_vel_z directly."
            )

        ratios = (
            cfg.standing_ratio,
            cfg.only_x_ratio,
            cfg.only_y_ratio,
            cfg.only_z_ratio,
        )

        if any(ratio < 0.0 or ratio > 1.0 for ratio in ratios):
            raise ValueError(
                "standing_ratio, only_x_ratio, only_y_ratio and "
                "only_z_ratio must be in [0, 1]."
            )

        ratio_sum = sum(ratios)
        if ratio_sum > 1.0 + 1.0e-6:
            raise ValueError(
                f"Command category ratio sum must be <= 1.0, got {ratio_sum}."
            )

        self.mixed_ratio = max(0.0, 1.0 - ratio_sum)

        super().__init__(cfg, env)

    def __str__(self) -> str:
        message = super().__str__()
        message += "\n\tXYZ command category ratios:"
        message += f"\n\t\tStanding: {self.cfg.standing_ratio:.3f}"
        message += f"\n\t\tOnly x:   {self.cfg.only_x_ratio:.3f}"
        message += f"\n\t\tOnly y:   {self.cfg.only_y_ratio:.3f}"
        message += f"\n\t\tOnly z:   {self.cfg.only_z_ratio:.3f}"
        message += f"\n\t\tMixed:    {self.mixed_ratio:.3f}"
        return message

    def _sample_range(
        self,
        count: int,
        value_range: tuple[float, float],
        min_abs: float = 0.0,
    ) -> torch.Tensor:
        """Sample uniformly while optionally excluding values near zero."""

        low, high = value_range

        if low > high:
            raise ValueError(
                f"Invalid velocity range ({low}, {high}): low must be <= high."
            )

        if count == 0:
            return torch.empty(0, device=self.device)

        if min_abs <= 0.0:
            return torch.empty(count, device=self.device).uniform_(low, high)

        # Allowed negative interval: [low, min(high, -min_abs)]
        negative_low = low
        negative_high = min(high, -min_abs)
        negative_length = max(negative_high - negative_low, 0.0)

        # Allowed positive interval: [max(low, min_abs), high]
        positive_low = max(low, min_abs)
        positive_high = high
        positive_length = max(positive_high - positive_low, 0.0)

        total_length = negative_length + positive_length
        if total_length <= 0.0:
            raise ValueError(
                f"Range {value_range} contains no value satisfying "
                f"abs(value) >= {min_abs}."
            )

        selector = torch.rand(count, device=self.device) * total_length
        values = torch.empty(count, device=self.device)

        negative_mask = selector < negative_length
        positive_mask = ~negative_mask

        if torch.any(negative_mask):
            values[negative_mask] = (
                negative_low + selector[negative_mask]
            )

        if torch.any(positive_mask):
            values[positive_mask] = (
                positive_low
                + selector[positive_mask]
                - negative_length
            )

        return values

    def _resample_command(self, env_ids: Sequence[int]):
        num_commands = len(env_ids)
        if num_commands == 0:
            return

        ranges = self.cfg.ranges

        # Start with mixed xyz commands.
        sampled_commands = torch.empty(
            (num_commands, 3),
            device=self.device,
        )
        sampled_commands[:, 0].uniform_(*ranges.lin_vel_x)
        sampled_commands[:, 1].uniform_(*ranges.lin_vel_y)
        sampled_commands[:, 2].uniform_(*ranges.ang_vel_z)

        # Select one mutually exclusive command category per environment.
        category = torch.rand(num_commands, device=self.device)

        standing_end = self.cfg.standing_ratio
        only_x_end = standing_end + self.cfg.only_x_ratio
        only_y_end = only_x_end + self.cfg.only_y_ratio
        only_z_end = only_y_end + self.cfg.only_z_ratio

        standing_mask = category < standing_end
        only_x_mask = (
            (category >= standing_end)
            & (category < only_x_end)
        )
        only_y_mask = (
            (category >= only_x_end)
            & (category < only_y_end)
        )
        only_z_mask = (
            (category >= only_y_end)
            & (category < only_z_end)
        )

        # Standing command.
        sampled_commands[standing_mask] = 0.0

        # Only x command.
        num_only_x = int(only_x_mask.sum().item())
        if num_only_x > 0:
            x_range = (
                self.cfg.only_x_range
                if self.cfg.only_x_range is not None
                else ranges.lin_vel_x
            )
            sampled_commands[only_x_mask] = 0.0
            sampled_commands[only_x_mask, 0] = self._sample_range(
                num_only_x,
                x_range,
                self.cfg.only_x_min_abs,
            )

        # Only y command.
        num_only_y = int(only_y_mask.sum().item())
        if num_only_y > 0:
            y_range = (
                self.cfg.only_y_range
                if self.cfg.only_y_range is not None
                else ranges.lin_vel_y
            )
            sampled_commands[only_y_mask] = 0.0
            sampled_commands[only_y_mask, 1] = self._sample_range(
                num_only_y,
                y_range,
                self.cfg.only_y_min_abs,
            )

        # Only z angular-velocity command.
        num_only_z = int(only_z_mask.sum().item())
        if num_only_z > 0:
            z_range = (
                self.cfg.only_z_range
                if self.cfg.only_z_range is not None
                else ranges.ang_vel_z
            )
            sampled_commands[only_z_mask] = 0.0
            sampled_commands[only_z_mask, 2] = self._sample_range(
                num_only_z,
                z_range,
                self.cfg.only_z_min_abs,
            )

        self.vel_command_b[env_ids] = sampled_commands
        self.is_standing_env[env_ids] = standing_mask
        self.is_heading_env[env_ids] = False


@configclass
class XYZVelocityCommandCfg(UniformVelocityCommandCfg):
    """Configuration for XYZVelocityCommand."""

    class_type: type = XYZVelocityCommand

    # Angular velocity is sampled directly.
    heading_command: bool = False

    # Proportions of all generated commands.
    standing_ratio: float = 0.0
    only_x_ratio: float = 0.0
    only_y_ratio: float = 0.0
    only_z_ratio: float = 0.0

    # Optional ranges used specifically for single-axis commands.
    # None means use the corresponding range in `ranges`.
    only_x_range: tuple[float, float] | None = None
    only_y_range: tuple[float, float] | None = None
    only_z_range: tuple[float, float] | None = None

    # Exclude near-zero values in single-axis commands.
    only_x_min_abs: float = 0.0
    only_y_min_abs: float = 0.0
    only_z_min_abs: float = 0.0

    @configclass
    class Ranges(UniformVelocityCommandCfg.Ranges):
        lin_vel_x: tuple[float, float] = MISSING
        lin_vel_y: tuple[float, float] = MISSING
        ang_vel_z: tuple[float, float] = MISSING
        heading: tuple[float, float] | None = None

    ranges: Ranges = MISSING