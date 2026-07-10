import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils import configclass

# T800 motor/inertia parameters aligned with the original T800 setup.
ARMATURE_Q300H_L = 0.2427264
ARMATURE_Q300H = 0.14110848
ARMATURE_Q200H = 0.0448737
ARMATURE_Q50H = 0.0354625
ARMATURE_Q25H = 0.00671625

EFFORT_LIMIT_Q300H_L = 415.0
EFFORT_LIMIT_Q300H = 370.0
EFFORT_LIMIT_Q200H = 222.0
EFFORT_LIMIT_Q50H = 160.0
EFFORT_LIMIT_Q25H = 52.0

VELOCITY_LIMIT_Q300H_L = 25.96
VELOCITY_LIMIT_Q300H = 25.31
VELOCITY_LIMIT_Q200H = 23.19
VELOCITY_LIMIT_Q50H = 33.51
VELOCITY_LIMIT_Q25H = 35.2

DEFAULT_Q_HIP_PITCH = -0.06
DEFAULT_Q_HIP_ROLL = 0.0
DEFAULT_Q_HIP_YAW = 0.0
DEFAULT_Q_KNEE_PITCH = 0.12
DEFAULT_Q_ANKLE_PITCH = -0.06
DEFAULT_Q_ANKLE_ROLL = 0.0
DEFAULT_Q_TORSO_YAW = 0.0
DEFAULT_Q_SHOULDER_PITCH = 0.0
DEFAULT_Q_SHOULDER_ROLL_L = 0.15
DEFAULT_Q_SHOULDER_ROLL_R = -0.15
DEFAULT_Q_SHOULDER_YAW = 0.0
DEFAULT_Q_ELBOW_PITCH = -0.25
DEFAULT_Q_ELBOW_YAW = 0.0
DEFAULT_Q_HEAD_PITCH = 0.0
DEFAULT_Q_HEAD_YAW = 0.0

STIFFNESS_Q300H_L = 180.0
STIFFNESS_Q300H = 100.0
STIFFNESS_Q200H = 100.0
STIFFNESS_Q50H = 40.0
STIFFNESS_Q25H = 50.0

DAMPING_Q300H_L = 5.0
DAMPING_Q300H = 3.0
DAMPING_Q200H = 3.0
DAMPING_Q50H = 0.3
DAMPING_Q25H = 0.3

@configclass
class UnitreeArticulationCfg(ArticulationCfg):
    """Configuration for Unitree articulations."""

    joint_sdk_names: list[str] = None

    soft_joint_pos_limit_factor = 0.9

@configclass
class UnitreeUsdFileCfg(sim_utils.UsdFileCfg):
    activate_contact_sensors: bool = True
    rigid_props = sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=False,
        retain_accelerations=False,
        linear_damping=0.0,
        angular_damping=0.0,
        max_linear_velocity=1000.0,
        max_angular_velocity=1000.0,
        max_depenetration_velocity=1.0,
    )
    articulation_props = sim_utils.ArticulationRootPropertiesCfg(
        enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
    )

T800_CFG = UnitreeArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="source/engineai_lab/assets/serial_t800/serial_t800.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),        
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.8),
        joint_pos={
            "J00_HIP_PITCH_L": DEFAULT_Q_HIP_PITCH,
            "J01_HIP_ROLL_L": DEFAULT_Q_HIP_ROLL,
            "J02_HIP_YAW_L": DEFAULT_Q_HIP_YAW,
            "J03_KNEE_PITCH_L": DEFAULT_Q_KNEE_PITCH,
            "J04_ANKLE_PITCH_L": DEFAULT_Q_ANKLE_PITCH,
            "J05_ANKLE_ROLL_L": DEFAULT_Q_ANKLE_ROLL,
            "J06_HIP_PITCH_R": DEFAULT_Q_HIP_PITCH,
            "J07_HIP_ROLL_R": DEFAULT_Q_HIP_ROLL,
            "J08_HIP_YAW_R": DEFAULT_Q_HIP_YAW,
            "J09_KNEE_PITCH_R": DEFAULT_Q_KNEE_PITCH,
            "J10_ANKLE_PITCH_R": DEFAULT_Q_ANKLE_PITCH,
            "J11_ANKLE_ROLL_R": DEFAULT_Q_ANKLE_ROLL,
            "J12_TORSO_YAW": DEFAULT_Q_TORSO_YAW,
            "J13_SHOULDER_PITCH_L": DEFAULT_Q_SHOULDER_PITCH,
            "J14_SHOULDER_ROLL_L": DEFAULT_Q_SHOULDER_ROLL_L,
            "J15_SHOULDER_YAW_L": DEFAULT_Q_SHOULDER_YAW,
            "J16_ELBOW_PITCH_L": DEFAULT_Q_ELBOW_PITCH,
            "J17_ELBOW_YAW_L": DEFAULT_Q_ELBOW_YAW,
            "J18_SHOULDER_PITCH_R": DEFAULT_Q_SHOULDER_PITCH,
            "J19_SHOULDER_ROLL_R": DEFAULT_Q_SHOULDER_ROLL_R,
            "J20_SHOULDER_YAW_R": DEFAULT_Q_SHOULDER_YAW,
            "J21_ELBOW_PITCH_R": DEFAULT_Q_ELBOW_PITCH,
            "J22_ELBOW_YAW_R": DEFAULT_Q_ELBOW_YAW,
            "J23_HEAD_PITCH": DEFAULT_Q_HEAD_PITCH,
            "J24_HEAD_YAW": DEFAULT_Q_HEAD_YAW,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_HIP_PITCH.*",
                ".*_HIP_ROLL.*",
                ".*_HIP_YAW.*",
                ".*_KNEE_PITCH.*",
            ],
            effort_limit_sim={
                ".*_HIP_PITCH.*": EFFORT_LIMIT_Q300H_L,
                ".*_HIP_ROLL.*": EFFORT_LIMIT_Q300H,
                ".*_HIP_YAW.*": EFFORT_LIMIT_Q200H,
                ".*_KNEE_PITCH.*": EFFORT_LIMIT_Q300H_L,
            },
            velocity_limit_sim={
                ".*_HIP_PITCH.*": VELOCITY_LIMIT_Q300H_L,
                ".*_HIP_ROLL.*": VELOCITY_LIMIT_Q300H,
                ".*_HIP_YAW.*": VELOCITY_LIMIT_Q200H,
                ".*_KNEE_PITCH.*": VELOCITY_LIMIT_Q300H_L,
            },
            stiffness={
                ".*_HIP_PITCH.*": STIFFNESS_Q300H_L,
                ".*_HIP_ROLL.*": STIFFNESS_Q300H,
                ".*_HIP_YAW.*": STIFFNESS_Q200H,
                ".*_KNEE_PITCH.*": STIFFNESS_Q300H_L,
            },
            damping={
                ".*_HIP_PITCH.*": DAMPING_Q300H_L,
                ".*_HIP_ROLL.*": DAMPING_Q300H,
                ".*_HIP_YAW.*": DAMPING_Q200H,
                ".*_KNEE_PITCH.*": DAMPING_Q300H_L,
            },
            armature={
                ".*_HIP_PITCH.*": ARMATURE_Q300H_L,
                ".*_HIP_ROLL.*": ARMATURE_Q300H,
                ".*_HIP_YAW.*": ARMATURE_Q200H,
                ".*_KNEE_PITCH.*": ARMATURE_Q300H_L,
            },
            friction={
                ".*_HIP_PITCH.*": 0.1,
                ".*_HIP_ROLL.*": 0.1,
                ".*_HIP_YAW.*": 0.05,
                ".*_KNEE_PITCH.*": 0.1,
            },
            dynamic_friction={
                ".*_HIP_PITCH.*": 0.08,
                ".*_HIP_ROLL.*": 0.08,
                ".*_HIP_YAW.*": 0.04,
                ".*_KNEE_PITCH.*": 0.08,
            },
            viscous_friction={
                ".*_HIP_PITCH.*": 0.01,
                ".*_HIP_ROLL.*": 0.01,
                ".*_HIP_YAW.*": 0.005,
                ".*_KNEE_PITCH.*": 0.01,
            },
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ANKLE_PITCH.*", ".*_ANKLE_ROLL.*"],
            effort_limit_sim={
                ".*_ANKLE_PITCH.*": EFFORT_LIMIT_Q50H,
                ".*_ANKLE_ROLL.*": EFFORT_LIMIT_Q50H,
            },
            velocity_limit_sim={
                ".*_ANKLE_PITCH.*": VELOCITY_LIMIT_Q50H,
                ".*_ANKLE_ROLL.*": VELOCITY_LIMIT_Q50H,
            },
            stiffness={
                ".*_ANKLE_PITCH.*": STIFFNESS_Q50H,
                ".*_ANKLE_ROLL.*": STIFFNESS_Q50H,
            },
            damping={
                ".*_ANKLE_PITCH.*": DAMPING_Q50H,
                ".*_ANKLE_ROLL.*": DAMPING_Q50H,
            },
            armature={
                ".*_ANKLE_ROLL.*": ARMATURE_Q50H,
                ".*_ANKLE_PITCH.*": ARMATURE_Q50H,
            },
            friction={
                ".*_ANKLE_PITCH.*": 0.15,
                ".*_ANKLE_ROLL.*": 0.12,
            },
            dynamic_friction={
                ".*_ANKLE_PITCH.*": 0.12,
                ".*_ANKLE_ROLL.*": 0.1,
            },
            viscous_friction={
                ".*_ANKLE_PITCH.*": 0.015,
                ".*_ANKLE_ROLL.*": 0.012,
            },
        ),
        "torso_yaw": ImplicitActuatorCfg(
            joint_names_expr=["J12_TORSO_YAW"],
            effort_limit_sim=EFFORT_LIMIT_Q200H,
            velocity_limit_sim=VELOCITY_LIMIT_Q200H,
            stiffness=STIFFNESS_Q200H,
            damping=DAMPING_Q200H,
            armature=ARMATURE_Q200H,
            friction=0.08,
            dynamic_friction=0.06,
            viscous_friction=0.008,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_SHOULDER_PITCH.*",
                ".*_SHOULDER_ROLL.*",
                ".*_SHOULDER_YAW.*",
                ".*_ELBOW_PITCH.*",
                ".*_ELBOW_YAW.*",
            ],
            effort_limit_sim={
                ".*_SHOULDER_PITCH.*": EFFORT_LIMIT_Q50H,
                ".*_SHOULDER_ROLL.*": EFFORT_LIMIT_Q50H,
                ".*_SHOULDER_YAW.*": EFFORT_LIMIT_Q50H,
                ".*_ELBOW_PITCH.*": EFFORT_LIMIT_Q50H,
                ".*_ELBOW_YAW.*": EFFORT_LIMIT_Q25H,
            },
            velocity_limit_sim={
                ".*_SHOULDER_PITCH.*": VELOCITY_LIMIT_Q50H,
                ".*_SHOULDER_ROLL.*": VELOCITY_LIMIT_Q50H,
                ".*_SHOULDER_YAW.*": VELOCITY_LIMIT_Q50H,
                ".*_ELBOW_PITCH.*": VELOCITY_LIMIT_Q50H,
                ".*_ELBOW_YAW.*": VELOCITY_LIMIT_Q25H,
            },
            stiffness={
                ".*_SHOULDER_PITCH.*": STIFFNESS_Q50H,
                ".*_SHOULDER_ROLL.*": STIFFNESS_Q50H,
                ".*_SHOULDER_YAW.*": STIFFNESS_Q50H,
                ".*_ELBOW_PITCH.*": STIFFNESS_Q50H,
                ".*_ELBOW_YAW.*": STIFFNESS_Q25H,
            },
            damping={
                ".*_SHOULDER_PITCH.*": DAMPING_Q50H,
                ".*_SHOULDER_ROLL.*": DAMPING_Q50H,
                ".*_SHOULDER_YAW.*": DAMPING_Q50H,
                ".*_ELBOW_PITCH.*": DAMPING_Q50H,
                ".*_ELBOW_YAW.*": DAMPING_Q25H,
            },
            armature={
                ".*_SHOULDER_PITCH.*": ARMATURE_Q50H,
                ".*_SHOULDER_ROLL.*": ARMATURE_Q50H,
                ".*_SHOULDER_YAW.*": ARMATURE_Q50H,
                ".*_ELBOW_PITCH.*": ARMATURE_Q50H,
                ".*_ELBOW_YAW.*": ARMATURE_Q25H,
            },
            friction={
                ".*_SHOULDER_PITCH.*": 0.12,
                ".*_SHOULDER_ROLL.*": 0.1,
                ".*_SHOULDER_YAW.*": 0.08,
                ".*_ELBOW_PITCH.*": 0.08,
                ".*_ELBOW_YAW.*": 0.06,
            },
            dynamic_friction={
                ".*_SHOULDER_PITCH.*": 0.1,
                ".*_SHOULDER_ROLL.*": 0.08,
                ".*_SHOULDER_YAW.*": 0.06,
                ".*_ELBOW_PITCH.*": 0.06,
                ".*_ELBOW_YAW.*": 0.05,
            },
            viscous_friction={
                ".*_SHOULDER_PITCH.*": 0.012,
                ".*_SHOULDER_ROLL.*": 0.01,
                ".*_SHOULDER_YAW.*": 0.008,
                ".*_ELBOW_PITCH.*": 0.008,
                ".*_ELBOW_YAW.*": 0.006,
            },
        ),
        "head": ImplicitActuatorCfg(
            joint_names_expr=["J23_HEAD_PITCH", "J24_HEAD_YAW"],
            effort_limit_sim={
                "J23_HEAD_PITCH": EFFORT_LIMIT_Q25H,
                "J24_HEAD_YAW": EFFORT_LIMIT_Q25H,
            },
            velocity_limit_sim={
                "J23_HEAD_PITCH": VELOCITY_LIMIT_Q25H,
                "J24_HEAD_YAW": VELOCITY_LIMIT_Q25H,
            },
            stiffness={
                "J23_HEAD_PITCH": STIFFNESS_Q25H,
                "J24_HEAD_YAW": STIFFNESS_Q25H,
            },
            damping={
                "J23_HEAD_PITCH": DAMPING_Q25H,
                "J24_HEAD_YAW": DAMPING_Q25H,
            },
            armature={
                "J23_HEAD_PITCH": ARMATURE_Q25H,
                "J24_HEAD_YAW": ARMATURE_Q25H,
            },
            friction={
                "J23_HEAD_PITCH": 0.05,
                "J24_HEAD_YAW": 0.05,
            },
            dynamic_friction={
                "J23_HEAD_PITCH": 0.04,
                "J24_HEAD_YAW": 0.04,
            },
            viscous_friction={
                "J23_HEAD_PITCH": 0.005,
                "J24_HEAD_YAW": 0.005,
            },
        ),
    },
    joint_sdk_names=[
        "J00_HIP_PITCH_L",
        "J01_HIP_ROLL_L",
        "J02_HIP_YAW_L",
        "J03_KNEE_PITCH_L",
        "J04_ANKLE_PITCH_L",
        "J05_ANKLE_ROLL_L",
        "J06_HIP_PITCH_R",
        "J07_HIP_ROLL_R",
        "J08_HIP_YAW_R",
        "J09_KNEE_PITCH_R",
        "J10_ANKLE_PITCH_R",
        "J11_ANKLE_ROLL_R",
        "J12_TORSO_YAW",
        "J13_SHOULDER_PITCH_L",
        "J14_SHOULDER_ROLL_L",
        "J15_SHOULDER_YAW_L",
        "J16_ELBOW_PITCH_L",
        "J17_ELBOW_YAW_L",
        "J18_SHOULDER_PITCH_R",
        "J19_SHOULDER_ROLL_R",
        "J20_SHOULDER_YAW_R",
        "J21_ELBOW_PITCH_R",
        "J22_ELBOW_YAW_R",
        "J23_HEAD_PITCH",
        "J24_HEAD_YAW",
    ],    
)

# T800_ACTION_SCALE = {}
# for a in T800_CFG.actuators.values():
#     e = a.effort_limit_sim
#     s = a.stiffness
#     if not isinstance(e, dict):
#         e = {n: e for n in a.joint_names_expr}
#     if not isinstance(s, dict):
#         s = {n: s for n in a.joint_names_expr}
#     for n in e.keys():  # 改用 e.keys() 而不是 joint_names_expr
#         # if "ANKLE" in n:
#         #     continue
#         if n in s and s[n]:
#             T800_ACTION_SCALE[n] = 0.25 * e[n] / s[n]


