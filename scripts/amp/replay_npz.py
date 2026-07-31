"""This script demonstrates how to use the interactive scene interface to setup a scene with multiple prims.

python scripts/amp/replay_npz.py --robot pm01 --input_file <path_to_motion.npz>
python scripts/amp/replay_npz.py --robot t800 --input_file <path_to_motion.npz>
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import numpy as np
import torch

from isaaclab.app import AppLauncher

DEFAULT_INPUT_FILE = "datasets/amp/walk_pm_test.npz"

# add argparse arguments
parser = argparse.ArgumentParser(description="Replay converted motions.")
parser.add_argument("--registry_name", type=str, default=None, help="The name of the wandb registry.")
parser.add_argument("--input_file", type=str, default=DEFAULT_INPUT_FILE, help="Path to a local .npz motion file.")
parser.add_argument(
    "--robot", type=str, default="pm01", choices=["pm01", "t800"], help="Robot type to use."
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Pre-defined configs
##
from engineai_rl_lab.tasks.locomotion.amp.robots.pm01 import PM01_CFG, PM_WAIST_DFS_JOINT_NAMES
from engineai_rl_lab.tasks.locomotion.amp.robots.t800 import T800_CFG
from engineai_rl_lab.tasks.locomotion.amp.utils.AMP_data_loader import AMPDataLoader


def _without_head_joints(joint_names: list[str]) -> list[str]:
    """Return the AMP motion joints, excluding head joints from replayed joint arrays."""
    return [name for name in joint_names if "HEAD" not in name]


# Joint-name synonyms across robot variants (e.g. PM01 "WAIST_YAW" vs T800 "TORSO_YAW").
# Motions exported with one variant's joint labels can then be replayed on the other.
_JOINT_NAME_SYNONYMS = [("WAIST_YAW", "TORSO_YAW")]


def _match_joint_names_to_robot(joint_names: list[str], robot: Articulation) -> list[str]:
    """Map motion joint names onto the robot's actual joint names, resolving known synonyms."""
    available = set(robot.joint_names)
    resolved: list[str] = []
    for name in joint_names:
        if name in available:
            resolved.append(name)
            continue
        matched = None
        for left, right in _JOINT_NAME_SYNONYMS:
            for candidate in (name.replace(left, right), name.replace(right, left)):
                if candidate != name and candidate in available:
                    matched = candidate
                    break
            if matched is not None:
                break
        resolved.append(matched if matched is not None else name)
    return resolved

ROBOT_CFGS = {
    "pm01": PM01_CFG,
    "t800": T800_CFG,
}

ROBOT_JOINT_NAMES = {
    "pm01": _without_head_joints(PM_WAIST_DFS_JOINT_NAMES),
    "t800": _without_head_joints(T800_CFG.joint_sdk_names),
}


@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    """Configuration for a replay motions scene."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    robot: ArticulationCfg = ROBOT_CFGS[args_cli.robot].replace(prim_path="{ENV_REGEX_NS}/Robot")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    # Extract scene entities
    robot: Articulation = scene["robot"]
    if args_cli.registry_name is not None:
        registry_name = args_cli.registry_name
        if ":" not in registry_name:
            registry_name += ":latest"
        import pathlib

        import wandb

        api = wandb.Api()
        artifact = api.artifact(registry_name)
        motion_file = str(pathlib.Path(artifact.download()) / "motion.npz")
    else:
        motion_file = args_cli.input_file

    motion = AMPDataLoader(motion_file=motion_file, device=sim.device, history_length=1)
    sim_dt = 1.0 / float(motion.fps[0].item())
    time_steps = torch.full((scene.num_envs,), -1, dtype=torch.long, device=sim.device)
    num_motion_joints = motion.joint_pos.shape[-1]
    num_sim_joints = robot.data.default_joint_pos.shape[-1]
    motion_joint_names = motion.joint_names
    fallback_joint_names = ROBOT_JOINT_NAMES[args_cli.robot]

    if motion_joint_names is not None and num_motion_joints == len(motion_joint_names):
        resolved_joint_names = _match_joint_names_to_robot(motion_joint_names, robot)
        robot_joint_indexes = robot.find_joints(resolved_joint_names, preserve_order=True)[0]
        num_robot_joints = len(resolved_joint_names)
        print(
            f"[INFO]: Motion has {num_motion_joints} named non-head joint columns. "
            "Replaying them in file joint-name order."
        )
    elif num_motion_joints == len(fallback_joint_names):
        robot_joint_indexes = robot.find_joints(fallback_joint_names, preserve_order=True)[0]
        num_robot_joints = len(fallback_joint_names)
        print(
            f"[INFO]: Motion has {num_motion_joints} non-head joint columns. "
            "Replaying them in configured AMP motion joint order."
        )
    elif num_motion_joints == num_sim_joints:
        robot_joint_indexes = slice(None)
        num_robot_joints = num_sim_joints
        print(
            f"[INFO]: Motion has {num_motion_joints} joint columns, matching the AMP robot joint order."
        )
    else:
        robot_joint_indexes = slice(None)
        num_robot_joints = num_sim_joints

    if num_motion_joints < num_robot_joints:
        raise RuntimeError(
            f"Motion has {num_motion_joints} joint columns, but robot '{args_cli.robot}' expects "
            f"{num_robot_joints} replay joints."
        )
    if num_motion_joints > num_robot_joints:
        print(
            f"[WARN]: Motion has {num_motion_joints} joint columns, but robot '{args_cli.robot}' matched "
            f"{num_robot_joints} joints. Extra joint columns will be ignored."
        )

    # Simulation loop
    while simulation_app.is_running():
        time_steps += 1
        reset_ids = time_steps >= motion.time_step_total
        time_steps[reset_ids] = 0

        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = motion.body_pos_w[time_steps, 0] + scene.env_origins
        root_states[:, 3:7] = motion.body_quat_w[time_steps, 0]
        root_states[:, 7:10] = motion.body_lin_vel_w[time_steps, 0]
        root_states[:, 10:] = motion.body_ang_vel_w[time_steps, 0]

        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        joint_pos[:, robot_joint_indexes] = motion.joint_pos[time_steps][:, :num_robot_joints]
        joint_vel[:, robot_joint_indexes] = motion.joint_vel[time_steps][:, :num_robot_joints]

        robot.write_root_state_to_sim(root_states)
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        scene.write_data_to_sim()
        sim.render()  # We don't want physic (sim.step())
        scene.update(sim_dt)

        pos_lookat = root_states[0, :3].cpu().numpy()
        sim.set_camera_view(pos_lookat + np.array([2.0, 2.0, 0.5]), pos_lookat)


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 0.02
    sim = SimulationContext(sim_cfg)

    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
