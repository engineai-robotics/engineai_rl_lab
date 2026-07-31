from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def robot_base_lin_vel_b(env: ManagerBasedEnv) -> torch.Tensor:
    """Base linear velocity expressed in the base frame."""
    asset = env.scene["robot"]
    # prefer direct base-frame velocity if available
    if getattr(asset.data, "root_lin_vel_b", None) is not None:
        return asset.data.root_lin_vel_b
    # fallback: rotate world velocity into base frame
    return math_utils.quat_apply_inverse(asset.data.root_quat_w, asset.data.root_lin_vel_w)


def robot_base_ang_vel_b(env: ManagerBasedEnv) -> torch.Tensor:
    """Base angular velocity expressed in the base frame."""
    asset = env.scene["robot"]
    # prefer direct base-frame velocity if available
    if getattr(asset.data, "root_ang_vel_b", None) is not None:
        return asset.data.root_ang_vel_b
    # fallback: rotate world velocity into base frame
    return math_utils.quat_apply_inverse(asset.data.root_quat_w, asset.data.root_ang_vel_w)