from __future__ import annotations

"""Replay compact AMP motions (GMR Skeleton1 NPZ) in Isaac Lab.

Playback is real-time by default (one motion second per wall-clock second).
Use --no-real-time to play as fast as rendering allows.

python scripts/amp/replay_npz.py --robot pm01 --input_file datasets/amp/pm01/data/walk_left_fast_001_Skeleton1.npz
python scripts/amp/replay_npz.py --robot t800 --input_file <path_to_motion.npz>

python scripts/amp/replay_npz.py --robot pm01 --input_folder datasets/amp/pm01/data
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from isaaclab.app import AppLauncher

DEFAULT_INPUT_FILE = "datasets/amp/pm01/data/walk_left_fast_001_Skeleton1.npz"

# add argparse arguments
parser = argparse.ArgumentParser(description="Replay converted motions.")
parser.add_argument("--registry_name", type=str, default=None, help="The name of the wandb registry.")
parser.add_argument(
    "--input_file",
    type=str,
    default=None,
    help="Path to a single local .npz motion file.",
)
parser.add_argument(
    "--input_folder",
    type=str,
    default=None,
    help="Path to a folder containing .npz motion files to play in sequence.",
)
parser.add_argument(
    "--robot", type=str, default="pm01", choices=["pm01", "t800"], help="Robot type to use."
)
parser.add_argument(
    "--real_time",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Play at the motion file's recorded real-world speed (default: True). Use --no-real-time to play as fast as rendering allows.",
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
from engineai_rl_lab.tasks.locomotion.amp.utils.AMP_data_loader import (
    _angular_velocity_world,
    _forward_difference,
    _validate_compact_motion,
)


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


@dataclass
class CompactMotion:
    """One compact Skeleton1 clip in Isaac Lab conventions."""

    fps: float
    time_step_total: int
    root_pos_w: torch.Tensor
    root_quat_w: torch.Tensor
    root_lin_vel_w: torch.Tensor
    root_ang_vel_w: torch.Tensor
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor


def _find_motion_files_in_folder(input_path: str) -> list[str]:
    """Return all .npz files in a folder, sorted by name."""
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Motion folder does not exist: {path}")
    if not path.is_dir():
        raise ValueError(f"--input_folder must be a directory, got: {path}")
    motion_files = sorted(path.glob("*.npz"))
    if not motion_files:
        raise FileNotFoundError(f"No .npz motion files found in folder: {path}")
    return [str(file) for file in motion_files]


def _load_compact_motion(motion_file: str, device: str) -> CompactMotion:
    """Load a GMR Skeleton1 NPZ: fps, root_pos, root_rot (xyzw), dof_pos."""
    with np.load(motion_file, allow_pickle=True) as data:
        required_keys = {"fps", "root_pos", "root_rot", "dof_pos"}
        missing_keys = required_keys.difference(data.files)
        if missing_keys:
            raise KeyError(
                f"Motion '{motion_file}' is not compact Skeleton1 format. "
                f"Missing required arrays: {sorted(missing_keys)}"
            )
        fps = float(data["fps"])
        root_pos = np.asarray(data["root_pos"])
        root_rot = np.asarray(data["root_rot"])
        dof_pos = np.asarray(data["dof_pos"])
        _validate_compact_motion(motion_file, fps, root_pos, root_rot, dof_pos)

    dt = 1.0 / fps
    root_pos_w = torch.as_tensor(root_pos, dtype=torch.float32, device=device)
    # Compact GMR files store quaternions as xyzw; Isaac Lab uses wxyz.
    root_quat_w = torch.as_tensor(root_rot[:, [3, 0, 1, 2]], dtype=torch.float32, device=device)
    root_quat_w = root_quat_w / torch.linalg.vector_norm(root_quat_w, dim=1, keepdim=True).clamp_min(1.0e-8)
    joint_pos = torch.as_tensor(dof_pos, dtype=torch.float32, device=device)
    return CompactMotion(
        fps=fps,
        time_step_total=joint_pos.shape[0],
        root_pos_w=root_pos_w,
        root_quat_w=root_quat_w,
        root_lin_vel_w=_forward_difference(root_pos_w, dt),
        root_ang_vel_w=_angular_velocity_world(root_quat_w, dt),
        joint_pos=joint_pos,
        joint_vel=_forward_difference(joint_pos, dt),
    )


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


def _load_motion(
    motion_file: str, robot: Articulation, sim: sim_utils.SimulationContext
) -> tuple[CompactMotion, float, list[int] | slice, int]:
    motion = _load_compact_motion(motion_file, sim.device)
    sim_dt = 1.0 / motion.fps
    print(f"[INFO]: Motion fps={motion.fps:.3f}, duration={motion.time_step_total / motion.fps:.2f}s")
    num_motion_joints = motion.joint_pos.shape[-1]
    num_sim_joints = robot.data.default_joint_pos.shape[-1]
    fallback_joint_names = ROBOT_JOINT_NAMES[args_cli.robot]
    resolved_joint_names = _match_joint_names_to_robot(fallback_joint_names, robot)

    if num_motion_joints >= len(resolved_joint_names):
        robot_joint_indexes = robot.find_joints(resolved_joint_names, preserve_order=True)[0]
        num_robot_joints = len(resolved_joint_names)
        print(
            f"[INFO]: Compact motion has {num_motion_joints} DOF columns "
            f"(J00..J22 plus optional head). Replaying {num_robot_joints} joints "
            "in configured AMP motion joint order."
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
            f"{num_robot_joints} joints. Extra joint columns (typically head) will be ignored."
        )

    return motion, sim_dt, robot_joint_indexes, num_robot_joints


def _resolve_motion_files() -> list[str]:
    if args_cli.input_file is not None and args_cli.input_folder is not None:
        raise ValueError("Specify only one of --input_file or --input_folder.")

    if args_cli.registry_name is not None:
        registry_name = args_cli.registry_name
        if ":" not in registry_name:
            registry_name += ":latest"

        import wandb

        api = wandb.Api()
        artifact = api.artifact(registry_name)
        return [str(Path(artifact.download()) / "motion.npz")]

    if args_cli.input_folder is not None:
        return _find_motion_files_in_folder(args_cli.input_folder)

    input_file = args_cli.input_file or DEFAULT_INPUT_FILE
    path = Path(input_file)
    if not path.exists():
        raise FileNotFoundError(f"Motion file does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"--input_file must be a .npz file, got: {path}")
    return [str(path)]


_DEFAULT_CAMERA_OFFSET = np.array([2.0, 2.0, 0.5], dtype=float)


def _get_viewport_camera_eye() -> np.ndarray | None:
    """Read the current viewport camera eye so user orbit/zoom is preserved."""
    try:
        from omni.kit.viewport.utility.camera_state import ViewportCameraState

        eye = ViewportCameraState().position_world
    except Exception:
        return None
    return np.array([float(eye[0]), float(eye[1]), float(eye[2])], dtype=float)


def _follow_robot_camera(
    sim: sim_utils.SimulationContext, robot_pos: np.ndarray, last_robot_pos: np.ndarray | None
) -> np.ndarray:
    """Keep the camera locked on the robot while preserving the current view direction."""
    robot_pos = np.asarray(robot_pos, dtype=float)
    if last_robot_pos is None:
        sim.set_camera_view(robot_pos + _DEFAULT_CAMERA_OFFSET, robot_pos)
        return robot_pos

    delta = robot_pos - last_robot_pos
    if float(np.linalg.norm(delta)) < 1.0e-8:
        return last_robot_pos

    eye = _get_viewport_camera_eye()
    if eye is None:
        eye = last_robot_pos + _DEFAULT_CAMERA_OFFSET
    sim.set_camera_view(eye + delta, robot_pos)
    return robot_pos


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    # Extract scene entities
    robot: Articulation = scene["robot"]
    motion_files = _resolve_motion_files()
    motion_index = 0
    motion, sim_dt, robot_joint_indexes, num_robot_joints = _load_motion(motion_files[motion_index], robot, sim)
    print(f"[INFO]: Replaying motion 1/{len(motion_files)}: {motion_files[motion_index]}")
    if args_cli.real_time:
        print("[INFO]: Real-time playback enabled (use --no-real-time to disable).")
    time_steps = torch.full((scene.num_envs,), -1, dtype=torch.long, device=sim.device)
    next_frame_time = time.perf_counter()
    last_robot_pos: np.ndarray | None = None

    # Simulation loop
    while simulation_app.is_running():
        time_steps += 1
        reset_ids = time_steps >= motion.time_step_total
        if torch.any(reset_ids):
            motion_index = (motion_index + 1) % len(motion_files)
            motion, sim_dt, robot_joint_indexes, num_robot_joints = _load_motion(
                motion_files[motion_index], robot, sim
            )
            print(f"[INFO]: Replaying motion {motion_index + 1}/{len(motion_files)}: {motion_files[motion_index]}")
            time_steps[:] = 0
            next_frame_time = time.perf_counter()

        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = motion.root_pos_w[time_steps] + scene.env_origins
        root_states[:, 3:7] = motion.root_quat_w[time_steps]
        root_states[:, 7:10] = motion.root_lin_vel_w[time_steps]
        root_states[:, 10:] = motion.root_ang_vel_w[time_steps]

        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        joint_pos[:, robot_joint_indexes] = motion.joint_pos[time_steps][:, :num_robot_joints]
        joint_vel[:, robot_joint_indexes] = motion.joint_vel[time_steps][:, :num_robot_joints]

        robot.write_root_state_to_sim(root_states)
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        last_robot_pos = _follow_robot_camera(sim, root_states[0, :3].cpu().numpy(), last_robot_pos)
        scene.write_data_to_sim()
        sim.render()  # We don't want physic (sim.step())
        scene.update(sim_dt)

        if args_cli.real_time:
            next_frame_time += sim_dt
            remaining = next_frame_time - time.perf_counter()
            if remaining > 0.0:
                time.sleep(remaining)
            else:
                next_frame_time = time.perf_counter()


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
