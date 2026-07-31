"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import importlib.util
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# ensure repository root is on Python path for Hydra registry imports
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

# ==========================================================================
# 全局参数：command（速度指令）范围设置
#   command[0] = 线速度 vx (m/s)
#   command[1] = 线速度 vy (m/s)
#   command[2] = 角速度 wz (rad/s)
# 通过终端输入的指令会被裁剪到以下范围内。
# ==========================================================================
CMD_LIN_VEL_X_RANGE = (-1.0, 1.0)
CMD_LIN_VEL_Y_RANGE = (-0.5, 0.5)
CMD_ANG_VEL_Z_RANGE = (-1.0, 1.0)

# 命令管理器中速度指令项的名称（与 env cfg 中的定义保持一致）
COMMAND_TERM_NAME = "base_velocity"

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# command 范围可通过命令行覆盖上面的全局默认值
parser.add_argument(
    "--cmd_x_range", type=float, nargs=2, default=None, metavar=("MIN", "MAX"),
    help="线速度 vx 的允许范围，例如 --cmd_x_range -1.0 1.0",
)
parser.add_argument(
    "--cmd_y_range", type=float, nargs=2, default=None, metavar=("MIN", "MAX"),
    help="线速度 vy 的允许范围，例如 --cmd_y_range -0.5 0.5",
)
parser.add_argument(
    "--cmd_z_range", type=float, nargs=2, default=None, metavar=("MIN", "MAX"),
    help="角速度 wz 的允许范围，例如 --cmd_z_range -1.0 1.0",
)

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# 用命令行参数覆盖全局 command 范围（若提供）
if args_cli.cmd_x_range is not None:
    CMD_LIN_VEL_X_RANGE = tuple(args_cli.cmd_x_range)
if args_cli.cmd_y_range is not None:
    CMD_LIN_VEL_Y_RANGE = tuple(args_cli.cmd_y_range)
if args_cli.cmd_z_range is not None:
    CMD_ANG_VEL_Z_RANGE = tuple(args_cli.cmd_z_range)

# always enable cameras to record video
# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import pathlib
import subprocess
import threading
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import engineai_rl_lab.tasks  # noqa: F401


def _clamp(value: float, value_range: tuple[float, float]) -> float:
    """将数值裁剪到给定范围内。"""
    low, high = min(value_range), max(value_range)
    return max(low, min(high, value))


class TerminalCommandController:
    """通过终端输入来实时控制机器人速度指令 (vx, vy, wz)。

    在后台线程中读取标准输入，不会阻塞仿真主循环。
    输入格式:
        - "vx vy wz"     例如 "0.5 0 0.3"   分别设置三个分量
        - "vx"           只设置 vx，vy/wz 置 0
        - "s" 或 "stop"  停止 (全部置 0)
        - "q" 或 "quit"  退出程序
    输入的指令会被裁剪到全局范围内。
    """

    def __init__(
        self,
        x_range: tuple[float, float],
        y_range: tuple[float, float],
        z_range: tuple[float, float],
    ):
        self._x_range = x_range
        self._y_range = y_range
        self._z_range = z_range
        self._lock = threading.Lock()
        self._command = [0.0, 0.0, 0.0]
        self._should_quit = False
        self._thread = threading.Thread(target=self._read_loop, daemon=True)

    def start(self):
        self._print_help()
        self._thread.start()

    def _print_help(self):
        print("\n" + "=" * 60)
        print("[Terminal Command Controller] 通过终端输入控制机器人运动")
        print(f"  vx 范围: {self._x_range}")
        print(f"  vy 范围: {self._y_range}")
        print(f"  wz 范围: {self._z_range}")
        print("  输入格式: 'vx vy wz'，例如 '0.5 0 0.3'")
        print("  输入 's' 停止, 'q' 退出")
        print("=" * 60 + "\n")

    def _read_loop(self):
        while not self._should_quit:
            try:
                line = input()
            except (EOFError, KeyboardInterrupt):
                break
            line = line.strip().lower()
            if line == "":
                continue
            if line in ("q", "quit", "exit"):
                self._should_quit = True
                break
            if line in ("s", "stop"):
                with self._lock:
                    self._command = [0.0, 0.0, 0.0]
                print("[Command] 已停止 (0, 0, 0)")
                continue

            parts = line.replace(",", " ").split()
            try:
                values = [float(p) for p in parts]
            except ValueError:
                print(f"[Command] 无法解析输入: '{line}'，请输入数字，如 '0.5 0 0.3'")
                continue

            vx = _clamp(values[0], self._x_range) if len(values) > 0 else 0.0
            vy = _clamp(values[1], self._y_range) if len(values) > 1 else 0.0
            wz = _clamp(values[2], self._z_range) if len(values) > 2 else 0.0
            with self._lock:
                self._command = [vx, vy, wz]
            print(f"[Command] 设置指令 -> vx={vx:.3f}, vy={vy:.3f}, wz={wz:.3f}")

    @property
    def should_quit(self) -> bool:
        return self._should_quit

    def get_command(self) -> list[float]:
        with self._lock:
            return list(self._command)


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with RSL-RL agent."""
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)


    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    file_basename = os.path.basename(resume_path).split(".")[0]
    onnx_filename = file_basename + ".onnx"
    ppo_runner.export_policy_to_onnx(path=export_model_dir, filename=onnx_filename)

    onnx_file = os.path.join(export_model_dir, onnx_filename)
    mnn_file = os.path.join(export_model_dir, file_basename + ".mnn")
    if os.path.exists(onnx_file):
        if importlib.util.find_spec("MNN") is None:
            print("[WARN] MNN is not installed in the current Python environment. Skipping MNN conversion.")
        else:
            try:
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "MNN.tools.mnnconvert",
                        "-f",
                        "ONNX",
                        "--modelFile",
                        onnx_file,
                        "--MNNModel",
                        mnn_file,
                        "--bizCode",
                        "MNN",
                    ],
                    check=True,
                )
                print(f"Successfully converted to MNN: {mnn_file}")
            except subprocess.CalledProcessError as err:
                print(f"[WARN] Failed to convert ONNX to MNN: {err}")
    else:
        print(f"ONNX file not found: {onnx_file}")


    # 获取速度指令项，并禁用其自动重采样，以便完全由终端输入控制
    command_term = env.unwrapped.command_manager.get_term(COMMAND_TERM_NAME)
    # 将重采样时间设为极大值，避免仿真过程中自动生成随机指令
    command_term.cfg.resampling_time_range = (1.0e9, 1.0e9)

    # 启动终端指令控制器
    controller = TerminalCommandController(
        CMD_LIN_VEL_X_RANGE, CMD_LIN_VEL_Y_RANGE, CMD_ANG_VEL_Z_RANGE
    )
    controller.start()

    def apply_terminal_command():
        """把终端输入的指令写入命令管理器，覆盖自动采样结果。"""
        cmd = controller.get_command()
        cmd_tensor = torch.tensor(cmd, device=env.unwrapped.device, dtype=torch.float32)
        # vel_command_b 形状为 [num_envs, 3]，广播到所有环境
        command_term.vel_command_b[:] = cmd_tensor
        # 关闭 standing 标记，避免指令被清零
        if hasattr(command_term, "is_standing_env"):
            command_term.is_standing_env[:] = False

    # reset environment
    obs = env.get_observations()
    apply_terminal_command()
    timestep = 0
    # simulate environment
    while simulation_app.is_running() and not controller.should_quit:
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, _, _ = env.step(actions)
            # 用终端指令覆盖命令（含 reset 后被重采样的环境）
            apply_terminal_command()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
