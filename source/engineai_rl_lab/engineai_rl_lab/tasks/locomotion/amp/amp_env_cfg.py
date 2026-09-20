"""Base configuration shared by the AMP locomotion environments.

Robot-specific joint/body names are left as :attr:`MISSING` and must be filled in by the
robot configuration in ``__post_init__``. Everything expressible with joint/body name
patterns that hold for every supported robot lives here, including the reward terms.
"""

from __future__ import annotations

from dataclasses import MISSING

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from engineai_rl_lab.tasks.locomotion.amp import mdp
from engineai_rl_lab.tasks.locomotion.amp.mdp.noise import Unoise as JointUnoise

# Number of observation frames stacked for the policy and critic groups.
OBS_HISTORY_LENGTH = 15
# Number of physical frames per discriminator observation (must match ``RslRlAmpCfg.frame_length``).
DISC_HISTORY_LENGTH = 10

# Must match AMPDataLoader._gather_frame_features when joint vel, base lin vel, and
# projected gravity are all enabled.
PM01_DISC_TERM_ORDER = (
    "base_ang_vel",
    "joint_pos",
    "joint_vel",
    "base_lin_vel",
    "projected_gravity",
)


def pin_observation_term_order(group: ObsGroup, term_names: tuple[str, ...]) -> None:
    """Rewrite ``group.__dict__`` so ObservationManager concatenates terms in ``term_names`` order.

    Isaac Lab concatenates observation terms by ``__dict__.items()``. Filling an optional
    slot after construction, or overriding it on a subclass, can move that term to the
    front of ``__dict__`` and silently break AMP expert alignment.
    """
    pinned = {name: getattr(group, name) for name in term_names}
    rest = {key: value for key, value in group.__dict__.items() if key not in pinned}
    group.__dict__.clear()
    group.__dict__.update(rest)
    group.__dict__.update(pinned)

# Joint and body name patterns valid for every supported robot. The waist pattern matches
# both the T800 naming (``J12_TORSO_YAW``) and the PM01 naming (``J12_WAIST_YAW``).
FOOT_BODY_NAMES = ["LINK_ANKLE_ROLL_L", "LINK_ANKLE_ROLL_R"]
ANKLE_JOINT_NAMES = [".*_ANKLE_PITCH_.*", ".*_ANKLE_ROLL_.*"]
HIP_JOINT_NAMES = [".*_HIP_YAW_.*", ".*_HIP_ROLL_.*"]
ARM_JOINT_NAMES = [".*_SHOULDER_.*", ".*_ELBOW_.*"]
WAIST_JOINT_NAMES = [".*(WAIST|TORSO)_YAW.*"]

# Flat ground with ±2 cm random undulations. Shared by every AMP robot.
AMP_FLAT_ROUGH_TERRAINS_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=False,
    sub_terrains={
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=1.0,
            noise_range=(-0.02, 0.02),
            noise_step=0.005,
            downsampled_scale=0.2,
            border_width=0.25,
        ),
    },
)


@configclass
class AmpSceneCfg(InteractiveSceneCfg):
    """AMP scene: flat ground with ±2 cm undulations, shared by every robot."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=AMP_FLAT_ROUGH_TERRAINS_CFG,
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
    robot: ArticulationCfg = MISSING
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class AmpActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        use_default_offset=True,
        preserve_order=True,
        joint_names=MISSING,
        scale=MISSING,
    )


@configclass
class AmpObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            params={"asset_cfg": MISSING},
            history_length=OBS_HISTORY_LENGTH,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            noise=JointUnoise(
                joint_names=MISSING,
                joint_noise_scales={".*ANKLE.*": 3.0},
                default_n_min=-0.5,
                default_n_max=0.5,
            ),
            params={"asset_cfg": MISSING},
            history_length=OBS_HISTORY_LENGTH,
        )
        actions = ObsTerm(func=mdp.last_action, history_length=OBS_HISTORY_LENGTH)
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.2, n_max=0.2),
            history_length=OBS_HISTORY_LENGTH,
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            history_length=OBS_HISTORY_LENGTH,
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group (adds the privileged base linear velocity)."""

        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            params={"asset_cfg": MISSING},
            history_length=OBS_HISTORY_LENGTH,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-1.5, n_max=1.5),
            params={"asset_cfg": MISSING},
            history_length=OBS_HISTORY_LENGTH,
        )
        actions = ObsTerm(func=mdp.last_action, history_length=OBS_HISTORY_LENGTH)
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.2, n_max=0.2),
            history_length=OBS_HISTORY_LENGTH,
        )
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            history_length=OBS_HISTORY_LENGTH,
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class DiscriminatorCfg(ObsGroup):
        """AMP discriminator observations, aligned with the expert dataset frame layout."""

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        joint_pos = ObsTerm(func=mdp.discriminator_joint_pos)
        joint_vel = ObsTerm(func=mdp.discriminator_joint_vel)
        # Optional PM01 slot. Do not fill this by subclass override or late assignment:
        # both reorder ``__dict__`` and break AMP. Reconstruct the full term list instead.
        base_lin_vel: ObsTerm | None = None
        projected_gravity = ObsTerm(func=mdp.projected_gravity)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            self.concatenate_dim = -1
            self.history_length = DISC_HISTORY_LENGTH
            self.flatten_history_dim = False

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    disc: DiscriminatorCfg = DiscriminatorCfg()


@configclass
class AmpCommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 12.0),
        heading_command=False,
        rel_standing_envs=0.1,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.6, 1.2),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(-1.0, 1.0),
        ),
    )


@configclass
class AmpEventCfg:
    """Domain randomization shared by the AMP locomotion environments."""

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

    rigid_body_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.9, 1.1),
            "operation": "scale",
            "recompute_inertia": True,
        },
    )

    actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )

    add_joint_default_pos = EventTerm(
        func=mdp.randomize_joint_default_pos,
        mode="startup",
        params={
            "asset_cfg": MISSING,
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

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(2.5, 6.0),
        params={
            "velocity_range": {
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
class AmpRewardsCfg:
    """Reward terms for the MDP."""

    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=3.0,
        params={"command_name": "base_velocity", "std": 0.3},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )

    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.8)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-6)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-1.0e-8)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.001)
    action_smoothness = RewTerm(
        func=mdp.action_smoothness_with_curriculum,
        weight=-0.04,
        params={
            "start_scale": 0.1,
            "power": 0.8,
            "interval_epochs": 200 * 24,
        },
    )
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ANKLE_JOINT_NAMES)},
    )

    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=HIP_JOINT_NAMES)},
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES)},
    )
    joint_deviation_waist = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=WAIST_JOINT_NAMES)},
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODY_NAMES),
            "threshold": 0.3,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODY_NAMES),
            "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_BODY_NAMES),
        },
    )

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-50.0)


@configclass
class AmpTerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=MISSING), "threshold": 1.0},
    )


@configclass
class LocomotionAmpEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the AMP locomotion environment on ±2 cm undulating flat terrain."""

    scene: AmpSceneCfg = AmpSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: AmpObservationsCfg = AmpObservationsCfg()
    actions: AmpActionsCfg = AmpActionsCfg()
    commands: AmpCommandsCfg = AmpCommandsCfg()
    rewards: AmpRewardsCfg = AmpRewardsCfg()
    terminations: AmpTerminationsCfg = AmpTerminationsCfg()
    events: AmpEventCfg = AmpEventCfg()
    curriculum: object | None = None

    def __post_init__(self):
        """Post initialization."""
        self.decimation = 5
        self.episode_length_s = 20.0
        self.sim.dt = 0.002
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = 0.005
