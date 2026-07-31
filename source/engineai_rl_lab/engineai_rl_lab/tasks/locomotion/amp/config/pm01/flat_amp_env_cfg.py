from __future__ import annotations

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
##
# Pre-defined configs
##
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from engineai_rl_lab.tasks.locomotion.amp import mdp
from engineai_rl_lab.tasks.locomotion.amp.robots.pm01 import PM01_CFG, PM_WAIST_DFS_JOINT_NAMES, PM01_DFS_JOINT_ORDER_ASSET_CFG
from engineai_rl_lab.tasks.tracking.robots.actuator import DelayedImplicitActuatorCfg
from engineai_rl_lab.tasks.locomotion.amp.mdp.noise import Unoise as MyNiose


def dummy_history_term(env):
    # zero-out waist joint observations to avoid AMP mismatch between robot variants
    robot = env.scene["robot"]
    joint_pos = mdp.joint_pos(env).clone()
    joint_vel = mdp.joint_vel(env).clone()
    waist_joint_ids = robot.find_joints(
        ".*WAIST_YAW.*", preserve_order=True)[0]
    if len(waist_joint_ids) > 0:
        waist_joint_ids = torch.as_tensor(
            waist_joint_ids, device=joint_pos.device)
        joint_pos[:, waist_joint_ids] = 0.0
        joint_vel[:, waist_joint_ids] = 0.0
    head_joint_ids = [i for i, name in enumerate(
        robot.joint_names) if name == "J23_HEAD_YAW"]
    if len(head_joint_ids) > 0:
        keep_joint_mask = torch.ones(
            joint_pos.shape[1], dtype=torch.bool, device=joint_pos.device)
        keep_joint_mask[torch.as_tensor(
            head_joint_ids, device=joint_pos.device)] = False
        joint_pos = joint_pos[:, keep_joint_mask]
    lin_vel = mdp.robot_base_lin_vel_b(env)
    ang_vel = mdp.robot_base_ang_vel_b(env)
    projected_gravity = mdp.projected_gravity(env)

    # feature order must match gather_frame_features in AMP_data_loader.py
    return torch.cat([joint_pos * 9, lin_vel * 7, ang_vel, projected_gravity], dim=-1)


ACTUATOR_DELAY_RANGE = (2, 8)
def _build_delayed_actuators():
    delayed_actuators = {}
    for name, cfg in PM01_CFG.actuators.items():
        delayed_actuators[name] = DelayedImplicitActuatorCfg(
            joint_names_expr=cfg.joint_names_expr,
            effort_limit=cfg.effort_limit,
            effort_limit_sim=cfg.effort_limit_sim,
            velocity_limit=cfg.velocity_limit,
            velocity_limit_sim=cfg.velocity_limit_sim,
            stiffness=cfg.stiffness,
            damping=cfg.damping,
            armature=cfg.armature,
            friction=cfg.friction,
            dynamic_friction=cfg.dynamic_friction,
            viscous_friction=cfg.viscous_friction,
            min_delay=ACTUATOR_DELAY_RANGE[0],
            max_delay=ACTUATOR_DELAY_RANGE[1],
        )
    return delayed_actuators


@configclass
class PM01SceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
    )
    # robots
    robot: ArticulationCfg = PM01_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        actuators=_build_delayed_actuators(),
    )
    # sensors
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class PM01Rewards:
    """Reward terms migrated from G1AmpRewards and mapped to PM01 names."""

    # -- task
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.2)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-2.0e-6)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-1.0e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
    action_smoothness = RewTerm(
        func=mdp.action_smoothness_with_curriculum,
        weight=-0.04,
        params={"start_scale": 0.1,
                "power": 0.8,
                "interval_epochs": 200*24
                },
    )
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*_ANKLE_PITCH_.*", ".*_ANKLE_ROLL_.*"],
            )
        },
    )
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*_HIP_YAW_.*", ".*_HIP_ROLL_.*"],
            ),
        },
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.05,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*_SHOULDER_.*", ".*_ELBOW_.*"],
            ),
        },
    )
    joint_deviation_waist = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.3,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["J12_WAIST_YAW"]),
        },
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_all_direction,
        weight=0.5,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["LINK_ANKLE_ROLL_L", "LINK_ANKLE_ROLL_R"],
            ),
            "threshold": 0.25,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["LINK_ANKLE_ROLL_L", "LINK_ANKLE_ROLL_R"],
            ),
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["LINK_ANKLE_ROLL_L", "LINK_ANKLE_ROLL_R"],
            ),
        },
    )
    feet_clearance_turning = RewTerm(
        func=mdp.feet_clearance_turning,
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["LINK_ANKLE_ROLL_L", "LINK_ANKLE_ROLL_R"]),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["LINK_ANKLE_ROLL_L", "LINK_ANKLE_ROLL_R"]),
            "command_name": "base_velocity",
            "stand_threshold": 0.1,
            "yaw_threshold": 0.3,
            "target_clearance": 0.1,
        },
    )
    feet_air_time_similarity = RewTerm(
        func=mdp.feet_air_time_similarity,
        weight=0.5,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["LINK_ANKLE_ROLL_L", "LINK_ANKLE_ROLL_R"],
            ),
            "scale": 4.0,
            "min_air_time": 0.05,
        },
    )
    command_stall = RewTerm(
        func=mdp.command_stall_penalty,
        weight=-2.0,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.1,
            "response_fraction": 0.2,
        },
    )    
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-50.0)


@configclass
class PM01Termination:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[
                                             "LINK_BASE", "LINK_KNEE_PITCH.*", ".*SHOULDER.*", ".*ELBOW.*", "LINK_TORSO_YAW"]), "threshold": 1.0},
    )


@configclass
class PM01ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot",
                                           use_default_offset=True,
                                           preserve_order=True,
                                           joint_names=PM_WAIST_DFS_JOINT_NAMES,
                                           scale={".*_HIP_PITCH_.*": 0.5,
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
                                                  ".*_ELBOW_YAW_.*": 0.2
                                                  }
                                           )


@configclass
class PM01ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            params={
                "asset_cfg": PM01_DFS_JOINT_ORDER_ASSET_CFG,
            },
            history_length=15,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            noise=MyNiose(
                joint_names=PM_WAIST_DFS_JOINT_NAMES,
                joint_noise_scales={".*ANKLE.*": 3.0},
                default_n_min=-0.5,
                default_n_max=0.5,
            ),
            params={
                "asset_cfg": PM01_DFS_JOINT_ORDER_ASSET_CFG,
            },
            history_length=15,
        )
        actions = ObsTerm(func=mdp.last_action,
                          history_length=15)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel,
                               noise=Unoise(n_min=-0.2, n_max=0.2),
                               history_length=15)
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            history_length=15
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands,params={"command_name": "base_velocity"})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):

        # observation terms (order preserved)
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            params={
                "asset_cfg": PM01_DFS_JOINT_ORDER_ASSET_CFG,
            },
            history_length=15,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-1.5, n_max=1.5),
            params={
                "asset_cfg": PM01_DFS_JOINT_ORDER_ASSET_CFG,
            },
            history_length=15,
        )
        actions = ObsTerm(func=mdp.last_action,
                          history_length=15)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel,
                               noise=Unoise(n_min=-0.2, n_max=0.2),
                               history_length=15)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            history_length=15
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands,
                                    params={"command_name": "base_velocity"})
        
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class AMPCfg(ObsGroup):
        history = ObsTerm(func=dummy_history_term)

        def __post_init__(self):
            self.history_length = 5

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    amp: AMPCfg = AMPCfg()


@configclass
class PM01Commands:
    """Command specifications for the MDP."""

    # base_velocity = mdp.UniformVelocityCommandCfg(
    #     asset_name="robot",
    #     resampling_time_range=(7.5, 7.5),
    #     rel_standing_envs=0.1,
    #     rel_heading_envs=1.0,
    #     heading_command=False,
    #     heading_control_stiffness=0.5,
    #     debug_vis=True,
    #     ranges=mdp.UniformVelocityCommandCfg.Ranges(
    #         lin_vel_x=(0, 0.8),
    #         lin_vel_y=(0, 0),
    #         ang_vel_z=(-1.0, 1.0),
    #     ),
    # )
    base_velocity = mdp.XYZVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        debug_vis=True,

        # 所有指令中的占比。
        standing_ratio=0.1,
        only_x_ratio=0.1,
        only_y_ratio=0.0,
        only_z_ratio=0.1,

        # 剩余自动作为 xyz 混合指令。
        ranges=mdp.XYZVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.6),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(-0.6, 0.6),
        ),

        # 单轴指令可以使用独立范围。
        only_x_range=(0.4, 0.7),
        only_y_range=(-0.4, 0.4),
        only_z_range=(-0.6, 0.6),

        # 纯旋转时排除 |wz| < 0.3 的弱指令。
        only_x_min_abs=0.1,
        only_y_min_abs=0.1,
        only_z_min_abs=0.3,
    )

@configclass
class PM01EventCfg:
    """PM01-specific randomizations."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.6),
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range": (0.0, 0.5),
            "num_buckets": 64,
        },
    )

    add_joint_default_pos = EventTerm(
        func=mdp.randomize_joint_default_pos,
        mode="startup",
        params={
            "asset_cfg": PM01_DFS_JOINT_ORDER_ASSET_CFG,
            "pos_distribution_params": (-0.01, 0.01),
            "operation": "add",
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup", 
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="LINK_BASE"),
            "com_range": {"x": (-0.025, 0.025), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        # interval_range_s=(1.0, 3.0),
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.2, 0.2),
                "roll": (-0.52, 0.52),
                "pitch": (-0.52, 0.52),
                "yaw": (-0.78, 0.78),
                }
                },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (0.8, 1.2), "velocity_range": (-0.5, 0.5)},
    )


@configclass
class PM01AMPFlatEnvCfg(ManagerBasedRLEnvCfg):
    """AMP flat environment configuration directly extending the base RL env config."""
    scene: PM01SceneCfg = PM01SceneCfg(num_envs=4096, env_spacing=2.5)
    observations: PM01ObservationsCfg = PM01ObservationsCfg()
    actions: PM01ActionsCfg = PM01ActionsCfg()
    commands: PM01Commands = PM01Commands()
    rewards: PM01Rewards = PM01Rewards()
    terminations: PM01Termination = PM01Termination()
    events: PM01EventCfg = PM01EventCfg()
    curriculum = None

    def __post_init__(self):
        """Apply sim wiring and curriculum toggles."""
        # simulation settings
        self.decimation = 5
        self.episode_length_s = 20.0
        self.sim.dt = 0.002
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = 0.005

        # command
        self.commands.base_velocity.ranges.lin_vel_x = (0.3, 0.7)
        self.commands.base_velocity.ranges.lin_vel_y = (0, 0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.6, 0.6)
        self.commands.base_velocity.resampling_time_range = (5, 12)

        # reward weights
        self.rewards.track_ang_vel_z_exp.weight = 3.0   
        self.rewards.track_ang_vel_z_exp.params["std"] = 0.3         
        self.rewards.track_lin_vel_xy_exp.weight = 3.0    
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.3
        self.rewards.flat_orientation_l2.weight = -1.0
        self.rewards.lin_vel_z_l2.weight = -0.8
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.dof_torques_l2.weight = -2e-6
        self.rewards.dof_acc_l2.weight = -2.5e-7
        self.rewards.action_rate_l2.weight = -0.02
        self.rewards.action_smoothness.weight = -0.03
        self.rewards.dof_pos_limits.weight = -0.1   
        self.rewards.joint_deviation_hip.weight = -0.05
        self.rewards.joint_deviation_arms.weight = -0.05
        self.rewards.joint_deviation_waist.weight = -1.0  
        self.rewards.feet_air_time.params["threshold"] = 0.25
        self.rewards.feet_air_time.weight = 0.75
        self.rewards.feet_slide.weight = -0.15
        self.rewards.feet_clearance_turning.weight = 0.0
        self.rewards.feet_air_time_similarity.weight = 0.5
        self.rewards.command_stall.weight = -0.5 #3.0
        self.rewards.termination_penalty.weight = -50.0
