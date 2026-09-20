"""Sagittal reflection for the T800 AMP actor and critic observations."""

from __future__ import annotations

import math

import torch
from tensordict import TensorDict


def _joint_reflection(joint_names: list[str]) -> tuple[list[int], list[float]]:
    """Resolve counterparts by anatomical name, independently of articulation order.

    In serial_t800.usd, mirrored pitch axes retain their joint coordinate sign;
    roll/yaw axes reverse it (including the oblique shoulder and elbow axes).
    """
    names = [name.split("_", 1)[1] for name in joint_names]
    indices, signs = [], []
    for name in names:
        if name.endswith("_L"):
            counterpart = name[:-2] + "_R"
        elif name.endswith("_R"):
            counterpart = name[:-2] + "_L"
        elif name == "TORSO_YAW":
            counterpart = name
        else:
            raise ValueError(f"Unsupported T800 symmetry joint: {name}")
        if counterpart not in names:
            raise ValueError(f"Missing T800 symmetry counterpart: {counterpart}")
        indices.append(names.index(counterpart))
        if "PITCH" in name:
            signs.append(1.0)
        elif "ROLL" in name or "YAW" in name:
            signs.append(-1.0)
        else:
            raise ValueError(f"Unsupported T800 joint axis: {name}")
    return indices, signs


def _observation_reflection(env, group: str, action_names: list[str]) -> tuple[list[int], list[float]]:
    """Build a flat mapping from manager term dimensions, including per-term history."""
    manager = env.observation_manager
    indices, signs = [], []
    offset = 0
    for name, shape in zip(manager.active_terms[group], manager.group_obs_term_dim[group], strict=True):
        term_cfg = getattr(getattr(env.cfg.observations, group), name)
        if name in ("joint_pos", "joint_vel"):
            asset_cfg = term_cfg.params["asset_cfg"]
            if not asset_cfg.preserve_order:
                raise ValueError("T800 symmetry requires explicitly ordered joint observations.")
            term_indices, term_signs = _joint_reflection(asset_cfg.joint_names)
        elif name == "actions":
            term_indices, term_signs = _joint_reflection(action_names)
        elif name == "base_ang_vel":
            term_indices, term_signs = [0, 1, 2], [-1.0, 1.0, -1.0]
        elif name in ("base_lin_vel", "projected_gravity"):
            term_indices, term_signs = [0, 1, 2], [1.0, -1.0, 1.0]
        elif name == "velocity_commands":
            term_indices, term_signs = [0, 1, 2], [1.0, -1.0, -1.0]
        else:
            raise ValueError(f"No T800 symmetry rule for observation {group}.{name}")

        width = len(term_indices)
        size = math.prod(shape)
        if size % width:
            raise ValueError(f"Unexpected shape for {group}.{name}: {shape}")
        # Isaac Lab flattens each term's [history, features] before concatenation.
        for frame in range(size // width):
            indices.extend(offset + frame * width + index for index in term_indices)
            signs.extend(term_signs)
        offset += size
    return indices, signs


def _reflect(env, value: torch.Tensor, group: str) -> torch.Tensor:
    """Cache the signed permutation on the batch device, then reflect its last axis."""
    cache = getattr(env, "_t800_symmetry_cache", None)
    if cache is None:
        cache = env._t800_symmetry_cache = {}
    key = (group, value.device, value.dtype)
    if key not in cache:
        action_cfg = env.cfg.actions.joint_pos
        if not action_cfg.preserve_order or len(action_cfg.joint_names) != 23:
            raise ValueError("T800 symmetry requires the ordered 23-joint position action.")
        if group == "actions":
            indices, signs = _joint_reflection(action_cfg.joint_names)
        else:
            indices, signs = _observation_reflection(env, group, action_cfg.joint_names)
        cache[key] = (
            torch.tensor(indices, device=value.device, dtype=torch.long),
            value.new_tensor(signs),
        )
    indices, signs = cache[key]
    if value.shape[-1] != len(indices):
        raise ValueError(f"T800 symmetry {group}: expected {len(indices)} features, got {value.shape[-1]}")
    return value.index_select(-1, indices) * signs


def compute_t800_symmetry(
    env, obs: TensorDict | None = None, actions: torch.Tensor | None = None
) -> tuple[TensorDict | None, torch.Tensor | None]:
    """Return original samples followed by their left/right reflections for RSL-RL.

    Transform raw observations before the actor's normalization. Joint positions
    and actions are relative to the default posture, with equal left/right action
    scales. Random calibration offsets are interpreted as mirrored latent robot
    parameters, not as offsets from an arbitrary environment in a shuffled batch.

    Auxiliary groups such as ``disc`` are carried along unchanged. They are not
    consumed by the actor mirror loss; this callback does not augment AMP expert
    data or the separate discriminator update.
    """
    env = env.unwrapped
    augmented_obs = None
    if obs is not None:
        mirrored = obs.clone()
        for group in ("policy", "critic"):
            if group in obs:
                mirrored[group] = _reflect(env, obs[group], group)
        augmented_obs = torch.cat((obs, mirrored), dim=0)
    augmented_actions = None
    if actions is not None:
        augmented_actions = torch.cat((actions, _reflect(env, actions, "actions")), dim=0)
    return augmented_obs, augmented_actions
