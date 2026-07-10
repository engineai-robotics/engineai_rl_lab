from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.utils.math import  quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def robot_base_lin_vel_b(env: ManagerBasedEnv) -> torch.Tensor:
    """Base linear velocity expressed in the base frame."""
    asset = env.scene["robot"]
    # prefer direct base-frame velocity if available
    if getattr(asset.data, "root_lin_vel_b", None) is not None:
        return asset.data.root_lin_vel_b
    # fallback: rotate world velocity into base frame
    return quat_apply_inverse(asset.data.root_quat_w, asset.data.root_lin_vel_w)

