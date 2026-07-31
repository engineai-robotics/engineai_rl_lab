from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.envs import mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import (
    euler_xyz_from_quat,
    quat_apply_inverse,
    quat_from_euler_xyz,
    quat_rotate_inverse,
    wrap_to_pi,
    yaw_quat,
)

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

    # Update action history for the next step and clear reset environments.
    prev_prev_action.copy_(action_manager.prev_action)
    reset_buf = getattr(env, "reset_buf", None)
    if reset_buf is not None:
        reset_env_ids = reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if reset_env_ids.numel() > 0:
            prev_prev_action[reset_env_ids] = 0.0

    return reward


def _epoch_curriculum_scale(env: ManagerBasedRLEnv, start_scale: float, power: float, interval_epochs: int) -> float:
    """Compute epoch-based scale by exponentiating by power every interval_epochs."""
    num_step = env.common_step_counter
    interval = max(int(interval_epochs), 1)
    updates = num_step // interval
    return float(start_scale) ** (float(power) ** updates)


def action_smoothness_with_curriculum(
    env: ManagerBasedRLEnv, start_scale: float, power: float, interval_epochs: int
) -> torch.Tensor:
    """Action smoothness penalty with epoch-based curriculum scaling."""
    reward = action_smoothness(env)
    return reward * _epoch_curriculum_scale(env, start_scale, power, interval_epochs)


def command_stall_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.1,
    response_fraction: float = 0.2,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize near-zero motion when a non-trivial velocity command is active.

    Linear progress is projected onto the commanded XY direction, so movement in
    the wrong direction does not avoid the penalty. Yaw progress is evaluated
    independently, which also covers in-place turning commands.
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


# def action_rate_with_curriculum(env: ManagerBasedRLEnv, start_scale: float, power: float, interval_epochs: int) -> torch.Tensor:
#     """Action rate penalty with epoch-based curriculum scaling."""
#     reward = mdp.action_rate_l2(env)
#     return reward * _epoch_curriculum_scale(env, start_scale, power, interval_epochs)


# def energy_cost_with_curriculum(
#     env: ManagerBasedRLEnv,
#     start_scale: float,
#     power: float,
#     interval_epochs: int,
#     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
# ) -> torch.Tensor:
#     """Energy cost penalty with epoch-based curriculum scaling."""
#     reward = energy_cost(env, asset_cfg=asset_cfg)
#     return reward * _epoch_curriculum_scale(env, start_scale, power, interval_epochs)


def feet_air_time_similarity(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    scale: float = 4.0,
    min_air_time: float = 0.0,
) -> torch.Tensor:
    """Reward similar air time between two feet.

    The reward is computed when either foot makes a new contact. It compares the last completed air times
    for the two feet and rewards small differences.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    body_ids = sensor_cfg.body_ids
    if body_ids is None or len(body_ids) != 2:
        raise ValueError("feet_air_time_similarity expects exactly two foot body ids in sensor_cfg.body_ids.")

    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, body_ids]

    recent_contact = torch.any(first_contact > 0.0, dim=1)
    valid = torch.all(last_air_time > min_air_time, dim=1)
    diff = torch.abs(last_air_time[:, 0] - last_air_time[:, 1])
    reward = torch.exp(-diff * scale)
    return reward * (recent_contact & valid)

# T800
# def feet_clearance_turning(
#     env: ManagerBasedRLEnv,
#     asset_cfg: SceneEntityCfg,
#     sensor_cfg: SceneEntityCfg,
#     command_name: str,
#     stand_threshold: float = 0.1,
#     yaw_threshold: float = 0.2,
#     force_threshold: float = 5.0,
#     ankle_height: float = 0.045,
#     target_clearance: float = 0.14,
#     scale: float = 80.0,
# ) -> torch.Tensor:
#     """Reward high stepping when tracking an in-place yaw command."""
#     commands = env.command_manager.get_command(command_name)
#     turning_in_place = (torch.norm(commands[:, :2], dim=1) < stand_threshold) & (
#         torch.abs(commands[:, 2]) > yaw_threshold
#     )

#     contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
#     contacts = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2] > force_threshold
#     single_support = torch.sum(contacts, dim=1) == 1
#     airborne = ~contacts

#     asset = env.scene[asset_cfg.name]
#     foot_clearance = torch.clamp(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - ankle_height, min=0.0)
#     clearance_error = torch.clamp(target_clearance - foot_clearance, min=0.0)
#     foot_reward = torch.exp(-torch.square(clearance_error) * scale) * airborne.float()
#     reward = torch.sum(foot_reward, dim=1) / torch.clamp(torch.sum(airborne.float(), dim=1), min=1.0)

#     return reward * (turning_in_place & single_support)

# PM01
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
    """Reward lifting either foot for a pure yaw velocity command."""
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


# def feet_clearance(
#     env: ManagerBasedRLEnv,
#     asset_cfg: SceneEntityCfg,
#     sensor_cfg: SceneEntityCfg,
#     command_name: str,
#     target_clearance: float = 0.10,
#     ankle_height: float = 0.045,
#     std: float = 0.04,
#     command_threshold: float = 0.1,
#     force_threshold: float = 5.0,
# ) -> torch.Tensor:
#     """Reward the airborne foot for reaching a target clearance on flat terrain.

#     The reward is active only for non-zero motion commands and single-support
#     states. This encourages a lifted swing foot without rewarding two-foot jumps.
#     """
#     commands = env.command_manager.get_command(command_name)
#     moving = (torch.norm(commands[:, :2], dim=1) > command_threshold) | (
#         torch.abs(commands[:, 2]) > command_threshold
#     )

#     contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
#     contacts = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2] > force_threshold
#     single_support = torch.sum(contacts, dim=1) == 1
#     airborne = ~contacts

#     asset = env.scene[asset_cfg.name]
#     foot_clearance = torch.clamp(
#         asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - ankle_height,
#         min=0.0,
#     )
#     clearance_error = foot_clearance - target_clearance
#     foot_reward = torch.exp(-0.5 * torch.square(clearance_error / std)) * airborne.float()
#     reward = torch.sum(foot_reward, dim=1)
#     return reward * moving * single_support


# def track_lin_vel_xy_yaw_frame_exp(
#     env, sigma: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), stand_threshold: float = 0.06
# ) -> torch.Tensor:
#     """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
#     commands = env.command_manager.get_command(command_name)
#     stand_command = (torch.norm(commands[:, :2], dim=1) < stand_threshold) & (
#         torch.abs(commands[:, 2]) < stand_threshold
#     )
#     asset = env.scene[asset_cfg.name]
#     vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
#     lin_vel_error_square = torch.sum(torch.square(commands[:, :2] - vel_yaw[:, :2]), dim=1)
#     lin_vel_error_abs = torch.sum(torch.abs(commands[:, :2] - vel_yaw[:, :2]), dim=1)
#     rew_square = torch.exp(-lin_vel_error_square * sigma)
#     rew_abs = torch.exp(-lin_vel_error_abs * sigma)
#     return torch.where(stand_command, rew_abs, rew_square)


# def track_ang_vel_z_world_exp(
#     env, command_name: str, sigma: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), stand_threshold: float = 0.06
# ) -> torch.Tensor:
#     """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
#     commands = env.command_manager.get_command(command_name)
#     stand_command = (torch.norm(commands[:, :2], dim=1) < stand_threshold) & (
#         torch.abs(commands[:, 2]) < stand_threshold
#     )
#     asset = env.scene[asset_cfg.name]
#     ang_vel_error_square = torch.square(commands[:, 2] - asset.data.root_ang_vel_w[:, 2])
#     ang_vel_error_abs = torch.abs(commands[:, 2] - asset.data.root_ang_vel_w[:, 2])
#     rew_square = torch.exp(-ang_vel_error_square * sigma)
#     rew_abs = torch.exp(-ang_vel_error_abs * sigma)
#     return torch.where(stand_command, rew_abs, rew_square)

# def feet_stumble(
#     env, sensor_cfg: SceneEntityCfg, tangential_threshold: float = 2.0, normal_threshold: float = 1.0
# ) -> torch.Tensor:
#     """Penalize feet hitting vertical surfaces using contact forces.

#     Flags a stumble when tangential force exceeds ``tangential_threshold`` while the normal force stays
#     below ``normal_threshold``. Returns the count of stumbling feet per environment.
#     """
#     contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
#     forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
#     tangential = torch.norm(forces[..., :2], dim=-1) > tangential_threshold
#     small_normal = torch.abs(forces[..., 2]) < normal_threshold
#     stumble = tangential & small_normal
#     return stumble.sum(dim=1)

# def feet_contact(
#     env, sensor_cfg: SceneEntityCfg, command_name: str, stand_threshold: float = 0.06, force_threshold: float = 5.0
# ) -> torch.Tensor:
#     """Reward valid foot contacts during walking and standing.

#     - When the command is effectively zero (stand), reward 1 only if both feet are in contact.
#     - Otherwise, reward 1 if any recent timestep had exactly one foot in contact.
#     Uses contact force history if available, else falls back to the latest forces.
#     """
#     commands = env.command_manager.get_command(command_name)
#     stand_command = (torch.norm(commands[:, :2], dim=1) < stand_threshold) & (
#         torch.abs(commands[:, 2]) < stand_threshold
#     )

#     contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
#     contact_history = contact_sensor.data.net_forces_w_history
#     if contact_history is None:
#         contact_history = contact_sensor.data.net_forces_w.unsqueeze(1)

#     contacts = contact_history[:, :, sensor_cfg.body_ids, 2] > force_threshold
#     contact_num_buf = torch.sum(contacts, dim=-1)

#     reward = stand_command.float()
#     contact_mask = (~stand_command) & torch.any(contact_num_buf == 1, dim=1)  
#     reward[contact_mask] = 1.0
    

#     return reward

# def feet_contact_fixed(
#     env, sensor_cfg: SceneEntityCfg, command_name: str, stand_threshold: float = 0.06, force_threshold: float = 5.0
# ) -> torch.Tensor:
#     """Reward valid foot contacts during walking and standing.

#     - When the command is effectively zero (stand), reward 1.
#     - Otherwise, reward 1 if any recent timestep had exactly one foot in contact.
#     Uses contact force history if available, else falls back to the latest forces.
#     """
#     commands = env.command_manager.get_command(command_name)
#     stand_command = (torch.norm(commands[:, :2], dim=1) < stand_threshold) & (
#         torch.abs(commands[:, 2]) < stand_threshold
#     )

#     contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
#     contact_history = contact_sensor.data.net_forces_w_history
#     if contact_history is None:
#         contact_history = contact_sensor.data.net_forces_w.unsqueeze(1)

#     contacts = contact_history[:, :, sensor_cfg.body_ids, 2] > force_threshold
#     contact_num_buf = torch.sum(contacts, dim=-1)

#     # For stand, require both feet in contact at the latest timestep.
#     stand_contact = contact_num_buf[:, -1] == 2
#     reward = (stand_command & stand_contact).float()
#     contact_mask = (~stand_command) & torch.any(contact_num_buf == 1, dim=1)
#     reward[contact_mask] = 1.0
    
#     return reward



# def feet_position(env,
#     asset_cfg: SceneEntityCfg,
#     command_name: str,
#     stand_threshold: float = 0.06,
#     ankle_distance: float = 0.22,
#     base_height_target: float = 0.82,
# ) -> torch.Tensor:
#     """Reward keeping feet near a desired stance when standing; otherwise return 1."""
#     commands = env.command_manager.get_command(command_name)
#     stand_command = (torch.norm(commands[:, :2], dim=1) < stand_threshold) & (
#         torch.abs(commands[:, 2]) < stand_threshold
#     )
#     asset = env.scene[asset_cfg.name]
    

#     feet_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
#     base_pos_w = asset.data.root_pos_w
#     base_quat_w = asset.data.root_quat_w

#     # isolate yaw heading; zero roll/pitch
#     r, p, y = euler_xyz_from_quat(base_quat_w)
#     heading_quat = quat_from_euler_xyz(torch.zeros_like(r), torch.zeros_like(p), y)
#     feet_pos_rel = feet_pos_w - base_pos_w.unsqueeze(1)
#     # Expand heading quaternions per foot to satisfy broadcasting expected by quat_apply_inverse.
#     num_envs, num_feet, _ = feet_pos_rel.shape
#     heading_quat_per_foot = heading_quat.unsqueeze(1).expand(-1, num_feet, -1).reshape(-1, 4)
#     feet_pos_rel_flat = feet_pos_rel.reshape(-1, 3)
#     feet_pos_heading = quat_apply_inverse(heading_quat_per_foot, feet_pos_rel_flat).reshape(num_envs, num_feet, 3)

#     desired_x = torch.zeros((num_envs, num_feet), device=feet_pos_heading.device)
#     desired_y = torch.cat(
#         (
#             (ankle_distance * 0.5) * torch.ones((num_envs, num_feet // 2), device=feet_pos_heading.device),
#             (-ankle_distance * 0.5) * torch.ones((num_envs, num_feet - num_feet // 2), device=feet_pos_heading.device),
#         ),
#         dim=1,
#     )
#     desired_z = -(base_height_target - 0.045) * torch.ones((num_envs, num_feet), device=feet_pos_heading.device)
#     desired = torch.stack((desired_x, desired_y, desired_z), dim=-1)

#     position_error = torch.sum(torch.abs(feet_pos_heading - desired), dim=(1, 2))
#     reward_stand = torch.exp(-position_error * 3.0)
#     return torch.where(stand_command, reward_stand, torch.ones_like(reward_stand))


# def feet_step_symmetry(
#     env,
#     asset_cfg: SceneEntityCfg,
#     command_name: str,
#     stand_threshold: float = 0.06,
#     scale: float = 8.0,
# ) -> torch.Tensor:
#     """Reward left/right feet having symmetric fore-aft stride around the base while walking."""
#     commands = env.command_manager.get_command(command_name)
#     stand_command = (torch.norm(commands[:, :2], dim=1) < stand_threshold) & (
#         torch.abs(commands[:, 2]) < stand_threshold
#     )

#     asset = env.scene[asset_cfg.name]
#     if asset_cfg.body_ids is None or len(asset_cfg.body_ids) != 2:
#         raise ValueError("feet_step_symmetry expects exactly two foot body ids in asset_cfg.body_ids.")

#     feet_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
#     base_pos_w = asset.data.root_pos_w
#     base_quat_w = asset.data.root_quat_w

#     roll, pitch, yaw = euler_xyz_from_quat(base_quat_w)
#     heading_quat = quat_from_euler_xyz(torch.zeros_like(roll), torch.zeros_like(pitch), yaw)
#     feet_pos_rel = feet_pos_w - base_pos_w.unsqueeze(1)
#     heading_quat_per_foot = heading_quat.unsqueeze(1).expand(-1, 2, -1).reshape(-1, 4)
#     feet_pos_heading = quat_apply_inverse(heading_quat_per_foot, feet_pos_rel.reshape(-1, 3)).reshape(-1, 2, 3)

#     stride_asymmetry = torch.square(feet_pos_heading[:, 0, 0] + feet_pos_heading[:, 1, 0])
#     reward_walk = torch.exp(-stride_asymmetry * scale)
#     return torch.where(stand_command, torch.ones_like(reward_walk), reward_walk)


# def feet_regulation(
#     env,
#     asset_cfg: SceneEntityCfg,
#     sensor_cfg: SceneEntityCfg,
#     ankle_height: float = 0.045,
#     base_height_target: float = 0.82,
#     clearance_clip: float = 1.0,
#     scale: float = 40.0,
# ) -> torch.Tensor:
#     """Penalize feet moving fast while too close to terrain."""
#     asset = env.scene[asset_cfg.name]
#     raycaster = env.scene.sensors[sensor_cfg.name]

#     # terrain height estimate from rays
#     ray_hits = raycaster.data.ray_hits_w
#     if ray_hits is None:
#         terrain_height = torch.zeros(asset.data.body_pos_w.shape[0], device=asset.data.body_pos_w.device)
#     else:
#         # use the highest hit point in world frame as terrain height reference
#         terrain_height = torch.max(ray_hits[..., 2], dim=-1).values

#     foot_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
#     foot_vel_xy = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]

#     foot_clearance = torch.clamp(foot_pos_w - ankle_height - terrain_height.unsqueeze(1), min=0.0, max=clearance_clip)
#     speed_term = torch.square(torch.norm(foot_vel_xy, dim=-1))
#     height_term = torch.exp(-foot_clearance / max(base_height_target, 1e-6) * scale)
#     reward = torch.sum(height_term * speed_term, dim=1)
#     return reward


# def feet_landing_velocity(
#     env,
#     asset_cfg: SceneEntityCfg,
#     sensor_cfg: SceneEntityCfg,
#     velocity_threshold: float = 0.25,
#     power: float = 2.0,
# ) -> torch.Tensor:
#     """Penalize high downward landing speed at first contact to reduce impact noise."""
#     contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
#     first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]

#     asset = env.scene[asset_cfg.name]
#     foot_vel_z = asset.data.body_lin_vel_w[:, sensor_cfg.body_ids, 2]
#     landing_speed = torch.clamp(-foot_vel_z - velocity_threshold, min=0.0)
#     penalty = torch.sum(torch.pow(landing_speed, power) * first_contact, dim=1)
#     return penalty

# def feet_z_velocity(
#     env,
#     asset_cfg: SceneEntityCfg,
#     sensor_cfg: SceneEntityCfg,
#     power: float = 2.0,    
# ) -> torch.Tensor:
#     asset = env.scene[asset_cfg.name]
#     foot_vel_z = asset.data.body_lin_vel_w[:, sensor_cfg.body_ids, 2]
#     z_speed = torch.clamp(-foot_vel_z, min=0.0)
#     penalty = torch.sum(torch.pow(z_speed, power), dim=1)
#     return penalty


# def foot_sound_suppression(
#     env,
#     asset_cfg: SceneEntityCfg,
#     sensor_cfg: SceneEntityCfg,
#     max_delta: float = 0.5,
# ) -> torch.Tensor:
#     """Penalize frame-to-frame changes in foot vertical velocity to suppress landing sounds."""
#     asset = env.scene[asset_cfg.name]
#     foot_vel_z = asset.data.body_lin_vel_w[:, sensor_cfg.body_ids, 2]

#     try:
#         prev_vel = env._prev_foot_z_vel
#     except AttributeError:
#         env._prev_foot_z_vel = foot_vel_z.clone()
#         return foot_vel_z.new_zeros(foot_vel_z.shape[0])

#     delta = foot_vel_z - prev_vel
#     delta = torch.clamp(delta, min=-max_delta, max=max_delta)
#     penalty = torch.sum(delta * delta, dim=1)

#     prev_vel.copy_(foot_vel_z)
#     reset_env_ids = env.reset_buf.nonzero(as_tuple=False).squeeze(-1)
#     if reset_env_ids.numel() > 0:
#         prev_vel[reset_env_ids] = foot_vel_z[reset_env_ids]
#         penalty[reset_env_ids] = 0.0

#     return penalty



# def base_height_tracking(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), target_height: float = 0.82) -> torch.Tensor:
#     """Reward keeping the base height near a target height."""
#     asset = env.scene[asset_cfg.name]
#     height_error = torch.abs(asset.data.root_pos_w[:, 2] - target_height)
#     return torch.exp(-height_error * 30.0)

# def energy_cost(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
#     """Penalize energy consumption approximated by the sum of squared joint torques."""
#     asset = env.scene[asset_cfg.name]
#     joint_torques = asset.data.applied_torque[:, :]
#     joint_vel = asset.data.joint_vel[:, :]
#     power = joint_torques * joint_vel
#     energy = torch.sum(torch.abs(power), dim=1)
#     return energy

# def feet_orientation(env, asset_cfg: SceneEntityCfg, command_name: str, stand_threshold: float = 0.06) -> torch.Tensor:
#     """Reward aligning feet orientation; ignore yaw error while turning."""
#     commands = env.command_manager.get_command(command_name)
#     yaw_command = torch.abs(commands[:, 2]) > stand_threshold

#     asset = env.scene[asset_cfg.name]
#     feet_quat = asset.data.body_quat_w[:, asset_cfg.body_ids, :]
#     base_quat = asset.data.root_quat_w

#     num_envs, num_feet, _ = feet_quat.shape
#     feet_flat = feet_quat.reshape(-1, 4)
#     roll, pitch, yaw = euler_xyz_from_quat(feet_flat)
#     roll = roll.reshape(num_envs, num_feet)
#     pitch = pitch.reshape(num_envs, num_feet)
#     yaw = yaw.reshape(num_envs, num_feet)

#     _, _, base_yaw = euler_xyz_from_quat(base_quat)

#     feet_roll_pitch_error = torch.sum(torch.abs(torch.stack((roll, pitch), dim=-1)), dim=-1)
#     feet_yaw_error = torch.abs(wrap_to_pi(yaw - base_yaw.unsqueeze(1)))

#     rew = torch.sum(feet_roll_pitch_error + feet_yaw_error, dim=1)
#     rew[yaw_command] = torch.sum(feet_roll_pitch_error[yaw_command], dim=1)
#     return torch.exp(-rew * 2.0)


# def base_orientation(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
#     """Reward keeping the base roll/pitch near zero."""
#     asset = env.scene[asset_cfg.name]
#     roll, pitch, yaw = euler_xyz_from_quat(asset.data.root_quat_w)
#     base_euler = torch.stack((roll, pitch, yaw), dim=-1)
#     return torch.exp(-torch.sum(torch.abs(base_euler[:, :2]), dim=-1) * 10.0)

# def reward_waist_pos(
#     env,
#     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
#     joint_names: list[str] | str | None = None,
#     scale: float = 5.0,
# ) -> torch.Tensor:
#     """Penalize motion on specific waist joints; joint_names provided by user."""
#     asset = env.scene[asset_cfg.name]
#     if joint_names is not None:
#         joint_ids = asset.find_joints(joint_names, preserve_order=True)[0]
#     elif asset_cfg.joint_ids is not None:
#         joint_ids = asset_cfg.joint_ids
#     else:
#         raise ValueError("penalize_waist_joint_motion requires joint_names or asset_cfg.joint_ids.")

#     pos_err = torch.abs(asset.data.joint_pos[:, joint_ids])
#     return torch.exp(-torch.sum(pos_err, dim=1) * scale)

# def penalize_foot_stumble(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
#     contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
#     contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
#     asset = env.scene[asset_cfg.name]
#     body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
#     return torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)

# def joint_deviation_exp(
#     env: ManagerBasedRLEnv,
#     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
#     tolerance: float = 0.1,
#     scale: float = 3.0,
#     max_err: float = 50.0,
# ) -> torch.Tensor:
#     """Penalize joint positions deviating from defaults, in an exponential, configurable way."""
#     asset: Articulation = env.scene[asset_cfg.name]
#     joint_ids = asset_cfg.joint_ids if asset_cfg.joint_ids is not None else slice(None)
#     joint_pos = asset.data.joint_pos[:, joint_ids]
#     default_pos = getattr(asset.data, "default_joint_pos", None)
#     if default_pos is not None:
#         default_pos = default_pos[:, joint_ids]
#     else:
#         default_pos = torch.zeros_like(joint_pos)
#         print("Warning: joint_deviation_exp reward called but default_joint_pos not set in asset data; assuming zeros.")

#     joint_error = torch.norm(joint_pos - default_pos, dim=1)
#     joint_error = torch.clamp(joint_error - tolerance, min=0.0, max=max_err)
#     return torch.exp(-joint_error * scale)

# def feet_air_time(
#     env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
# ) -> torch.Tensor:
#     """Reward long steps taken by the feet using L2-kernel.

#     This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
#     that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
#     the time for which the feet are in the air.

#     If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
#     Active for non-trivial linear OR yaw commands (so in-place turning also gets step incentive).
#     """
#     # extract the used quantities (to enable type-hinting)
#     contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
#     # compute the reward
#     first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
#     last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
#     reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
#     # no reward for zero command (lin_xy or yaw)
#     cmd = env.command_manager.get_command(command_name)
#     reward *= (torch.norm(cmd[:, :2], dim=1) > 0.1) | (torch.abs(cmd[:, 2]) > 0.1)
#     return reward


def feet_air_time_all_direction(env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    Active for non-trivial linear OR yaw commands (so in-place turning also gets step incentive).
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    feet_air_time = torch.clamp(last_air_time - threshold, min=0.0)
    reward = torch.sum(feet_air_time * first_contact, dim=1)
    # no reward for zero command (lin_xy or yaw)
    cmd = env.command_manager.get_command(command_name)
    reward *= (torch.norm(cmd[:, :2], dim=1) > 0.1) | (torch.abs(cmd[:, 2]) > 0.1)
    return reward
