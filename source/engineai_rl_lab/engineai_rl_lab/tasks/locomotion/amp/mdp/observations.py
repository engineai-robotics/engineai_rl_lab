from __future__ import annotations

import re
import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

# The expert dataset zeroes the waist/torso yaw joint and drops the head joints
# (see AMPDataLoader._load_joint_arrays), so the discriminator observation must match.
DISC_ZEROED_JOINT_PATTERN = r".*(WAIST|TORSO)_YAW.*"
DISC_DROPPED_JOINT_SUBSTRING = "HEAD"


def robot_base_lin_vel_b(env: ManagerBasedEnv) -> torch.Tensor:
    """Base linear velocity expressed in the base frame."""
    asset = env.scene["robot"]
    if getattr(asset.data, "root_lin_vel_b", None) is not None:
        return asset.data.root_lin_vel_b
    return math_utils.quat_apply_inverse(asset.data.root_quat_w, asset.data.root_lin_vel_w)


def robot_base_ang_vel_b(env: ManagerBasedEnv) -> torch.Tensor:
    """Base angular velocity expressed in the base frame."""
    asset = env.scene["robot"]
    if getattr(asset.data, "root_ang_vel_b", None) is not None:
        return asset.data.root_ang_vel_b
    return math_utils.quat_apply_inverse(asset.data.root_quat_w, asset.data.root_ang_vel_w)


def _discriminator_joint_state(env: ManagerBasedEnv, joint_state: torch.Tensor) -> torch.Tensor:
    """Align a joint state with the joint layout of the headless expert data."""
    asset: Articulation = env.scene["robot"]
    joint_names = asset.joint_names
    joint_state = joint_state.clone()

    zeroed_ids = [i for i, name in enumerate(joint_names) if re.fullmatch(DISC_ZEROED_JOINT_PATTERN, name)]
    if zeroed_ids:
        joint_state[:, torch.as_tensor(zeroed_ids, device=joint_state.device)] = 0.0

    kept_ids = [i for i, name in enumerate(joint_names) if DISC_DROPPED_JOINT_SUBSTRING not in name]
    if len(kept_ids) != len(joint_names):
        joint_state = joint_state[:, torch.as_tensor(kept_ids, device=joint_state.device)]
    return joint_state


def discriminator_joint_pos(env: ManagerBasedEnv) -> torch.Tensor:
    """Absolute joint positions in the expert data joint layout."""
    return _discriminator_joint_state(env, env.scene["robot"].data.joint_pos)


def discriminator_joint_vel(env: ManagerBasedEnv) -> torch.Tensor:
    """Joint velocities in the expert data joint layout."""
    return _discriminator_joint_state(env, env.scene["robot"].data.joint_vel)
