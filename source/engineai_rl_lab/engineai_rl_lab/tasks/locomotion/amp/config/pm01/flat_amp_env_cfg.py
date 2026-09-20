from __future__ import annotations

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass

from engineai_rl_lab.tasks.locomotion.amp import mdp
from engineai_rl_lab.tasks.locomotion.amp.amp_env_cfg import (
    DISC_HISTORY_LENGTH,
    AmpObservationsCfg,
    LocomotionAmpEnvCfg,
    PM01_DISC_TERM_ORDER,
    pin_observation_term_order,
)
from engineai_rl_lab.tasks.locomotion.amp.robots.pm01 import (
    PM01_CFG,
    PM01_DFS_JOINT_ORDER_ASSET_CFG,
    PM_WAIST_DFS_JOINT_NAMES,
)
from engineai_rl_lab.tasks.locomotion.amp.utils.actuators import build_delayed_actuators

ACTUATOR_DELAY_RANGE = (2, 8)

ACTION_SCALE = {
    ".*_HIP_PITCH_.*": 0.5,
    ".*_HIP_ROLL_.*": 0.2,
    ".*_HIP_YAW_.*": 0.2,
    ".*_KNEE_PITCH_.*": 0.5,
    ".*_ANKLE_PITCH_.*": 0.5,
    ".*_ANKLE_ROLL_.*": 0.2,
    ".*WAIST_YAW.*": 0.2,
    ".*_SHOULDER_PITCH_.*": 0.2,
    ".*_SHOULDER_ROLL_.*": 0.2,
    ".*_SHOULDER_YAW_.*": 0.2,
    ".*_ELBOW_PITCH_.*": 0.2,
    ".*_ELBOW_YAW_.*": 0.2,
}


@configclass
class PM01AmpObservationsCfg(AmpObservationsCfg):
    """PM01 observations. Disc terms are declared in expert-frame order, not inherited."""

    @configclass
    class DiscriminatorCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        joint_pos = ObsTerm(func=mdp.discriminator_joint_pos)
        joint_vel = ObsTerm(func=mdp.discriminator_joint_vel)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            self.concatenate_dim = -1
            self.history_length = DISC_HISTORY_LENGTH
            self.flatten_history_dim = False

    disc: DiscriminatorCfg = DiscriminatorCfg()


@configclass
class PM01AMPFlatEnvCfg(LocomotionAmpEnvCfg):
    """AMP flat environment configuration for the PM01 robot."""

    observations: PM01AmpObservationsCfg = PM01AmpObservationsCfg()

    def __post_init__(self):
        super().__post_init__()

        # PM01 trains on a plain ground plane rather than the generated rough terrain
        # configured by the shared AMP environment.
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.robot = PM01_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            actuators=build_delayed_actuators(PM01_CFG.actuators, *ACTUATOR_DELAY_RANGE),
        )

        self.actions.joint_pos.joint_names = PM_WAIST_DFS_JOINT_NAMES
        self.actions.joint_pos.scale = ACTION_SCALE

        pin_observation_term_order(self.observations.disc, PM01_DISC_TERM_ORDER)
        for group in (self.observations.policy, self.observations.critic):
            group.joint_pos.params["asset_cfg"] = PM01_DFS_JOINT_ORDER_ASSET_CFG
            group.joint_vel.params["asset_cfg"] = PM01_DFS_JOINT_ORDER_ASSET_CFG
        self.observations.policy.joint_vel.noise.joint_names = PM_WAIST_DFS_JOINT_NAMES

        self.events.add_joint_default_pos.params["asset_cfg"] = PM01_DFS_JOINT_ORDER_ASSET_CFG

        self.commands.base_velocity.rel_standing_envs = 0.1
        # Active reference frames: vx P05/P95 ~= 0.29/0.59 m/s and |wz| P95 ~= 0.57 rad/s.
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 0.8)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.6, 0.6)

        self.rewards.track_lin_vel_xy_exp.weight = 3.0
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.5
        self.rewards.track_ang_vel_z_exp.weight = 2.0
        self.rewards.track_ang_vel_z_exp.params["std"] = 0.5
        self.rewards.flat_orientation_l2.weight = -1.0
        self.rewards.lin_vel_z_l2.weight = -0.75
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.dof_torques_l2.weight = -1.0e-6
        self.rewards.dof_acc_l2.weight = -1.0e-8
        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.action_smoothness.weight = 0.0
        self.rewards.dof_pos_limits.weight = -0.1
        self.rewards.joint_deviation_hip.weight = -0.01
        self.rewards.joint_deviation_arms.weight = -0.01
        self.rewards.joint_deviation_waist.weight = -1.0
        self.rewards.feet_air_time.weight = 2.0
        self.rewards.feet_air_time.params["threshold"] = 0.3
        self.rewards.feet_slide.weight = -0.4
        self.rewards.termination_penalty.weight = -50.0

        self.terminations.base_contact.params["sensor_cfg"].body_names = [
            "LINK_BASE",
            "LINK_KNEE_PITCH.*",
            ".*SHOULDER.*",
            ".*ELBOW.*",
            "LINK_TORSO_YAW",
        ]
