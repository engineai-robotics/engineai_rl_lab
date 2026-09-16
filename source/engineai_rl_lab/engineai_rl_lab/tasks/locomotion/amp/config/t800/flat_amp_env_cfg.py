from __future__ import annotations

from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from engineai_rl_lab.tasks.locomotion.amp import mdp
from engineai_rl_lab.tasks.locomotion.amp.amp_env_cfg import AmpTerminationsCfg, LocomotionAmpEnvCfg
from engineai_rl_lab.tasks.locomotion.amp.robots.t800 import (
    T800_ACTION_SCALE,
    T800_CFG,
    T800_DFS_JOINT_NAMES,
    T800_DFS_JOINT_ORDER_ASSET_CFG,
)
from engineai_rl_lab.tasks.locomotion.amp.utils.actuators import build_delayed_actuators

ACTUATOR_DELAY_RANGE = (2, 8)


@configclass
class T800AmpTerminationsCfg(AmpTerminationsCfg):
    """T800 terminations; pelvis is ~1.06 m standing, so 0.65 m only fires on collapse."""

    root_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.65})


@configclass
class T800AMPFlatEnvCfg(LocomotionAmpEnvCfg):
    """AMP flat environment configuration for the T800 robot."""

    terminations: T800AmpTerminationsCfg = T800AmpTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = T800_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            actuators=build_delayed_actuators(T800_CFG.actuators, *ACTUATOR_DELAY_RANGE),
        )

        self.actions.joint_pos.joint_names = T800_DFS_JOINT_NAMES
        self.actions.joint_pos.scale = T800_ACTION_SCALE

        for group in (self.observations.policy, self.observations.critic):
            group.joint_pos.params["asset_cfg"] = T800_DFS_JOINT_ORDER_ASSET_CFG
            group.joint_vel.params["asset_cfg"] = T800_DFS_JOINT_ORDER_ASSET_CFG
        self.observations.policy.joint_vel.noise.joint_names = T800_DFS_JOINT_NAMES

        self.events.add_joint_default_pos.params["asset_cfg"] = T800_DFS_JOINT_ORDER_ASSET_CFG

        self.commands.base_velocity.rel_standing_envs = 0.1
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 0.8)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.6, 0.6)

        self.rewards.track_lin_vel_xy_exp.weight = 3.0
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.3
        self.rewards.track_ang_vel_z_exp.weight = 2.0
        self.rewards.track_ang_vel_z_exp.params["std"] = 0.5
        self.rewards.flat_orientation_l2.weight = -1.0
        self.rewards.lin_vel_z_l2.weight = -0.8
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.dof_torques_l2.weight = -1.0e-6
        self.rewards.dof_acc_l2.weight = -1.0e-8
        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.action_smoothness.weight = -0.0001
        self.rewards.dof_pos_limits.weight = -1.0
        self.rewards.joint_deviation_hip.weight = -0.05
        self.rewards.joint_deviation_arms.weight = -0.05
        self.rewards.joint_deviation_waist.weight = -1.0
        self.rewards.feet_air_time.weight = 1.0
        self.rewards.feet_air_time.params["threshold"] = 0.3
        self.rewards.feet_slide.weight = -0.1
        self.rewards.termination_penalty.weight = -50.0

        self.terminations.base_contact.params["sensor_cfg"].body_names = [
            "LINK_BASE",
            "LINK_KNEE_PITCH.*",
            ".*SHOULDER.*",
            ".*ELBOW.*",
            "LINK_WAIST_YAW",
        ]
