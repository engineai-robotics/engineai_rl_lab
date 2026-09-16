from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def action_smoothness(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize action second-order differences to encourage smooth control."""
    action_manager = env.action_manager
    prev_prev_action = getattr(action_manager, "_prev_prev_action", None)
    if prev_prev_action is None:
        prev_prev_action = torch.zeros_like(action_manager.action)
        action_manager._prev_prev_action = prev_prev_action

    second_diff = action_manager.action + prev_prev_action - 2.0 * action_manager.prev_action
    reward = torch.sum(torch.square(second_diff), dim=1)

    prev_prev_action.copy_(action_manager.prev_action)
    reset_buf = getattr(env, "reset_buf", None)
    if reset_buf is not None:
        reset_env_ids = reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if reset_env_ids.numel() > 0:
            prev_prev_action[reset_env_ids] = 0.0

    return reward


def _epoch_curriculum_scale(env: ManagerBasedRLEnv, start_scale: float, power: float, interval_epochs: int) -> float:
    """Compute the epoch-based curriculum scale."""
    interval = max(int(interval_epochs), 1)
    updates = env.common_step_counter // interval
    return float(start_scale) ** (float(power) ** updates)


def action_smoothness_with_curriculum(
    env: ManagerBasedRLEnv, start_scale: float, power: float, interval_epochs: int
) -> torch.Tensor:
    """Apply epoch-based curriculum scaling to the action-smoothness penalty."""
    return action_smoothness(env) * _epoch_curriculum_scale(env, start_scale, power, interval_epochs)


def command_stall_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.1,
    response_fraction: float = 0.2,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize insufficient motion in response to an active velocity command.

    Linear progress is projected onto the commanded XY direction. Yaw progress
    is evaluated independently to cover in-place turning commands.
    """
    if response_fraction <= 0.0:
        raise ValueError("response_fraction must be greater than zero.")

    command = env.command_manager.get_command(command_name)
    asset = env.scene[asset_cfg.name]

    command_lin_speed = torch.linalg.vector_norm(command[:, :2], dim=1)
    command_yaw_speed = torch.abs(command[:, 2])
    linear_active = command_lin_speed > command_threshold
    yaw_active = command_yaw_speed > command_threshold

    command_lin_direction = command[:, :2] / command_lin_speed.clamp_min(1.0e-6).unsqueeze(1)
    linear_progress = torch.sum(asset.data.root_lin_vel_b[:, :2] * command_lin_direction, dim=1)
    linear_ratio = torch.clamp(linear_progress / command_lin_speed.clamp_min(1.0e-6), min=0.0)

    yaw_progress = asset.data.root_ang_vel_b[:, 2] * torch.sign(command[:, 2])
    yaw_ratio = torch.clamp(yaw_progress / command_yaw_speed.clamp_min(1.0e-6), min=0.0)

    linear_stall = torch.clamp((response_fraction - linear_ratio) / response_fraction, min=0.0, max=1.0)
    yaw_stall = torch.clamp((response_fraction - yaw_ratio) / response_fraction, min=0.0, max=1.0)

    active_count = linear_active.float() + yaw_active.float()
    penalty = linear_stall * linear_active + yaw_stall * yaw_active
    return penalty / active_count.clamp_min(1.0)


def feet_air_time_similarity(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    scale: float = 4.0,
    min_air_time: float = 0.0,
) -> torch.Tensor:
    """Reward similar completed air times when either foot makes contact."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    body_ids = sensor_cfg.body_ids
    if body_ids is None or len(body_ids) != 2:
        raise ValueError("feet_air_time_similarity expects exactly two foot body ids in sensor_cfg.body_ids.")

    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, body_ids]

    recent_contact = torch.any(first_contact > 0.0, dim=1)
    valid = torch.all(last_air_time > min_air_time, dim=1)
    difference = torch.abs(last_air_time[:, 0] - last_air_time[:, 1])
    return torch.exp(-difference * scale) * (recent_contact & valid)


def feet_air_time_positive_biped_pure_yaw(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    yaw_threshold: float = 0.2,
    linear_velocity_threshold: float = 0.05,
    target_clearance: float = 0.14,
    clearance_std: float = 0.04,
) -> torch.Tensor:
    """Reward high, sustained steps that make progress on a pure-yaw command."""
    if threshold <= 0.0:
        raise ValueError("threshold must be greater than zero.")
    if target_clearance <= 0.0:
        raise ValueError("target_clearance must be greater than zero.")
    if clearance_std <= 0.0:
        raise ValueError("clearance_std must be greater than zero.")

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)

    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    air_time_reward = torch.min(
        torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0),
        dim=1,
    ).values
    air_time_reward = torch.clamp(air_time_reward / threshold, min=0.0, max=1.0)

    asset = env.scene[asset_cfg.name]
    foot_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    clearance = torch.max(foot_height, dim=1).values - torch.min(foot_height, dim=1).values
    clearance_progress = torch.clamp(clearance / target_clearance, min=0.0, max=1.0)
    clearance_target_reward = torch.exp(-torch.square((clearance - target_clearance) / clearance_std))
    # The progress term provides a dense signal; the kernel preserves an optimum at the target height.
    clearance_reward = 0.7 * clearance_progress + 0.3 * clearance_target_reward

    command = env.command_manager.get_command(command_name)
    pure_yaw_command = (
        torch.linalg.vector_norm(command[:, :2], dim=1) < linear_velocity_threshold
    ) & (torch.abs(command[:, 2]) > yaw_threshold)

    yaw_progress = asset.data.root_ang_vel_w[:, 2] * torch.sign(command[:, 2])
    yaw_progress_ratio = torch.clamp(
        yaw_progress / torch.abs(command[:, 2]).clamp_min(1.0e-6),
        min=0.0,
        max=1.0,
    )

    reward = 0.35 * air_time_reward * single_stance + 0.65 * clearance_reward
    progress_scale = 0.4 + 0.6 * yaw_progress_ratio
    return reward * progress_scale * pure_yaw_command


def feet_clearance_turning(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    stand_threshold: float = 0.1,
    yaw_threshold: float = 0.2,
    force_threshold: float = 5.0,
    ankle_height: float = 0.045,
    target_clearance: float = 0.14,
) -> torch.Tensor:
    """Reward lifting either foot for a pure-yaw velocity command."""
    commands = env.command_manager.get_command(command_name)
    pure_yaw_command = (
        (torch.abs(commands[:, 0]) < stand_threshold)
        & (torch.abs(commands[:, 1]) < stand_threshold)
        & (torch.abs(commands[:, 2]) > yaw_threshold)
    )

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2] > force_threshold
    has_support = torch.any(contacts, dim=1)

    asset = env.scene[asset_cfg.name]
    foot_clearance = torch.clamp(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - ankle_height, min=0.0)
    normalized_clearance = torch.clamp(foot_clearance / target_clearance, min=0.0, max=1.0)
    reward = torch.max(normalized_clearance, dim=1).values
    return reward * (pure_yaw_command & has_support)


def feet_air_time_all_direction(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
) -> torch.Tensor:
    """Reward foot air time for active linear or yaw commands."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    feet_air_time = torch.clamp(last_air_time - threshold, min=0.0)
    reward = torch.sum(feet_air_time * first_contact, dim=1)

    command = env.command_manager.get_command(command_name)
    command_active = (torch.norm(command[:, :2], dim=1) > 0.1) | (torch.abs(command[:, 2]) > 0.1)
    return reward * command_active
