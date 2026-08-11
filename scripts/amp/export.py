"""Export the latest policy checkpoint from one or more AMP training runs."""

"""Launch Isaac Sim Simulator first."""

import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

# Ensure the repository root is on Python path for Hydra registry imports.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

parser = argparse.ArgumentParser(description="Export AMP policies from RSL-RL checkpoints.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments used to create the runner.")
parser.add_argument("--experiment_name", type=str, default=None, help="Override the experiment log directory.")
parser.add_argument(
    "--load_run",
    type=str,
    nargs="+",
    required=True,
    help="One or more run directories whose latest checkpoints will be exported.",
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default="model_.*.pt",
    help="Checkpoint filename or regular expression. Defaults to the latest checkpoint in each run.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks.
import engineai_rl_lab.tasks  # noqa: F401


def export_checkpoint(ppo_runner: OnPolicyRunner, checkpoint_path: Path) -> None:
    """Export a loaded policy into a run-specific directory."""
    export_dir = checkpoint_path.parent / "export"
    export_dir.mkdir(parents=True, exist_ok=True)

    onnx_filename = f"{checkpoint_path.stem}.onnx"
    ppo_runner.export_policy_to_onnx(path=str(export_dir), filename=onnx_filename)

    onnx_path = export_dir / onnx_filename
    if not onnx_path.exists():
        raise RuntimeError(f"ONNX export did not create '{onnx_path}'.")
    print(f"[INFO] Exported ONNX policy: {onnx_path}")

    if importlib.util.find_spec("MNN") is None:
        print("[WARN] MNN is not installed in the current Python environment. Skipping MNN conversion.")
        return

    mnn_path = export_dir / f"{checkpoint_path.stem}.mnn"
    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "MNN.tools.mnnconvert",
                "-f",
                "ONNX",
                "--modelFile",
                str(onnx_path),
                "--MNNModel",
                str(mnn_path),
                "--bizCode",
                "MNN",
            ],
            check=True,
        )
        print(f"[INFO] Exported MNN policy: {mnn_path}")
    except subprocess.CalledProcessError as err:
        print(f"[WARN] Failed to convert ONNX to MNN: {err}")


def validate_checkpoint(checkpoint_path: Path) -> None:
    """Reject non-PyTorch files before passing them to torch.load."""
    if not checkpoint_path.is_file():
        raise ValueError(f"Checkpoint is not a file: '{checkpoint_path}'.")
    with checkpoint_path.open("rb") as checkpoint_file:
        header = checkpoint_file.read(2)
    # torch.save uses either a ZIP container (PK) or the legacy pickle format.
    if header != b"PK" and not header.startswith(b"\x80"):
        raise ValueError(
            f"Checkpoint is not a PyTorch checkpoint: '{checkpoint_path}' "
            f"(file starts with {header!r})."
        )


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Load and export each requested AMP checkpoint."""
    if args_cli.experiment_name is not None:
        agent_cfg.experiment_name = args_cli.experiment_name
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs

    log_root = (REPO_ROOT / "logs" / "rsl_rl" / agent_cfg.experiment_name).resolve()
    checkpoint_paths = [
        Path(get_checkpoint_path(str(log_root), run, args_cli.checkpoint)) for run in args_cli.load_run
    ]
    for checkpoint_path in checkpoint_paths:
        validate_checkpoint(checkpoint_path)
        print(f"[INFO] Latest checkpoint selected: {checkpoint_path}")

    env_cfg.log_dir = str(checkpoint_paths[0].parent)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env)

    try:
        ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        load_cfg = {"actor": True, "critic": False, "optimizer": False, "iteration": False, "rnd": False}
        for checkpoint_path in checkpoint_paths:
            # Only the actor is needed, so checkpoints with stale critic layouts still export.
            ppo_runner.load(str(checkpoint_path), load_cfg=load_cfg)
            export_checkpoint(ppo_runner, checkpoint_path)
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
