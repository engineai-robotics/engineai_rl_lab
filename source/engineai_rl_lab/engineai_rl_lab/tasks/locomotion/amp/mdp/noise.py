from __future__ import annotations

import inspect
import re

import torch
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg


def joint_uniform_noise(data: torch.Tensor, cfg: Unoise) -> torch.Tensor:
    """Apply uniform noise with optional per-joint ranges."""

    if cfg.joint_noise_scales:
        _resolve_joint_noise_ranges(data, cfg)

    if isinstance(cfg.n_max, torch.Tensor):
        cfg.n_max = cfg.n_max.to(data.device)
    if isinstance(cfg.n_min, torch.Tensor):
        cfg.n_min = cfg.n_min.to(data.device)

    noise = torch.rand_like(data) * (cfg.n_max - cfg.n_min) + cfg.n_min
    if cfg.operation == "add":
        return data + noise
    elif cfg.operation == "scale":
        return data * noise
    elif cfg.operation == "abs":
        return noise
    else:
        raise ValueError(f"Unknown operation in noise: {cfg.operation}")


def _resolve_joint_noise_ranges(data: torch.Tensor, cfg: Unoise) -> None:
    if isinstance(cfg.n_min, torch.Tensor) and isinstance(cfg.n_max, torch.Tensor):
        return

    if cfg.joint_names is not None:
        joint_names = list(cfg.joint_names)
    else:
        env = _find_env_from_call_stack()
        if env is None:
            raise RuntimeError(
                "Joint-wise Unoise requires either joint_names or access to the active env. "
                "Use it from an IsaacLab observation term after the ObservationManager is initialized."
            )
        joint_names = list(env.scene[cfg.asset_name].joint_names)

    if data.shape[-1] != len(joint_names):
        raise ValueError(
            f"Joint-wise Unoise expected the observation last dimension to match "
            f"the configured joint_names ({len(joint_names)}), but got {data.shape[-1]}."
        )

    n_min = torch.full((len(joint_names),), float(cfg.default_n_min), dtype=data.dtype, device=data.device)
    n_max = torch.full((len(joint_names),), float(cfg.default_n_max), dtype=data.dtype, device=data.device)

    for name_id, joint_name in enumerate(joint_names):
        for joint_pattern, scale in cfg.joint_noise_scales.items():
            if re.fullmatch(joint_pattern, joint_name):
                n_min[name_id] = -float(scale)
                n_max[name_id] = float(scale)
                break

    cfg.n_min = n_min
    cfg.n_max = n_max


def _find_env_from_call_stack():
    frame = inspect.currentframe()
    while frame is not None:
        maybe_self = frame.f_locals.get("self")
        maybe_env = getattr(maybe_self, "_env", None)
        if maybe_env is not None and hasattr(maybe_env, "scene"):
            return maybe_env
        frame = frame.f_back
    return None


@configclass
class Unoise(UniformNoiseCfg):
    """Uniform noise with optional per-joint noise ranges.

    When ``joint_noise_scales`` is provided, values are matched against
    ``joint_names`` in order, or against ``env.scene[asset_name].joint_names``
    when no explicit order is supplied. Matched joints receive ``[-scale,
    scale]`` noise and all other joints use ``default_n_min/max``.
    """

    func = joint_uniform_noise

    asset_name: str = "robot"
    default_n_min: float = -0.5
    default_n_max: float = 0.5
    joint_noise_scales: dict[str, float] | None = None
    joint_names: list[str] | tuple[str, ...] | None = None
