# engineai_rl_lab
[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.1.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.3.2-silver)](https://isaac-sim.github.io/IsaacLab)

[中文](README.md) | English

## Overview
This project provides reinforcement learning environments based on Isaac Lab. It currently supports the EngineAI PM01 and T800 robots, with tasks including whole-body tracking and AMP-based humanoid walking.

[Demo Video](https://www.bilibili.com/video/BV14GTk6UE9o/?share_source=copy_web&vd_source=74098b4d7182a602fcb3a63bb054e83f)

|Robot|Training|Sim2Sim|Deployment|
|:--------:|:--------:|:--------:|:--------:|
|||**Whole-Body Tracking**|||
|**T800**|<img src="./docs/train.gif" height="180"/>|<img src="./docs/sim2sim.gif" height="180"/>|<img src="./docs/deploy.gif" height="180"/>|
|**PM01**|<img src="./docs/train_pm.gif" height="180"/>|<img src="./docs/sim2sim_pm.gif" height="180"/>|<img src="./docs/deploy_pm.gif" height="180"/>|
|||**AMP**|||
|**PM01**|<img src="./docs/amp-pm-train.gif" height="180"/>|<img src="./docs/amp-pm-mujoco.gif" height="180"/>|<img src="./docs/amp-pm-deploy.gif" height="180"/>|

## Installation
### Install Isaac Lab
This repository is developed against Isaac Lab 2.3.2 at commit `c22775241e28f465fe345fa1a482ad6d29d712b0`. The code may not be compatible with other Isaac Lab versions. Refer to the official [Isaac Lab installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html) for detailed instructions.

### Install engineai_rl_lab
1. Clone the repository and download the large assets:

```bash
# 1. Clone the repository.
git clone https://github.com/engineai-robotics/engineai_rl_lab.git

# 2. Download the large assets.
cd engineai_rl_lab
git lfs install
git lfs pull
```

2. Install `engineai_rl_lab`:

```bash
# Make sure the Isaac Lab environment is active.
cd engineai_rl_lab
pip install -e source/engineai_rl_lab

# Install MNN.
pip install mnn
```

## Training
### Whole-Body Tracking
1. Convert CSV motion files to NPZ and replay the generated motions.

> The robot joint positions in the first and last frames of a whole-body tracking motion should be close to the PD stand pose. This helps keep the motion smooth when switching policies or modes.

```bash
# Convert CSV files to NPZ files. The NPZ files are written to the same directory.
# PM01
python scripts/tracking/csv_to_npz.py --robot pm01 --input_fps 30 -f datasets/tracking/pm01/dance.csv

# T800
python scripts/tracking/csv_to_npz.py --robot t800 --input_fps 30 -f datasets/tracking/t800/dance_t800.csv

# Replay NPZ files.
# PM01
python scripts/tracking/replay_npz.py --robot pm01 --input_file datasets/tracking/pm01/dance.npz

# T800
python scripts/tracking/replay_npz.py --robot t800 --input_file datasets/tracking/t800/dance_t800.npz
```

2. Train a policy:

```bash
# PM01
python scripts/tracking/train.py --task Tracking-Flat-PM01-Wo-State-Estimation-v0 --headless --num_envs 4096 --motion_file datasets/tracking/pm01/dance.npz

# T800
python scripts/tracking/train.py --task Tracking-Flat-T800-Wo-State-Estimation-v0 --headless --num_envs 4096 --motion_file datasets/tracking/t800/dance_t800.npz

# View the training logs.
python -m tensorboard.main --logdir logs
```

3. Evaluate and export the trained policy:

```bash
# PM01
python scripts/tracking/play.py --task Tracking-Flat-PM01-Wo-State-Estimation-v0 --num_envs 1 --motion_file datasets/tracking/pm01/dance.npz --load_run 2026-06-23_09-58-43 --checkpoint dance.pt

# T800
python scripts/tracking/play.py --task Tracking-Flat-T800-Wo-State-Estimation-v0 --num_envs 1 --motion_file datasets/tracking/t800/dance_t800.npz --load_run 2026-06-28_20-47-15 --checkpoint dance.pt
```

### AMP
1. Inspect the reference motions.

Before training, replay the data in Isaac Lab and verify the root pose, joint order, frame rate, and foot-ground contacts. Use `--input_file` to inspect one motion. Use `--input_folder` to play every NPZ file in a directory in filename order.

```bash
# PM01: replay every motion in the dataset.
python scripts/amp/replay_npz.py --robot pm01 --input_folder datasets/amp/pm01/data

# T800: replay every motion in the dataset.
python scripts/amp/replay_npz.py --robot t800 --input_folder datasets/amp/t800/data

# Example: replay one motion.
python scripts/amp/replay_npz.py --robot pm01 --input_file datasets/amp/pm01/data/walk_001_Skeleton9.npz
```

2. Train a policy:

```bash
# PM01
python scripts/amp/train.py --task AMP-Flat-PM01-v1 --headless --num_envs 4096

# T800
python scripts/amp/train.py --task AMP-Flat-T800-v1 --headless --num_envs 4096

# View the training curves.
python -m tensorboard.main --logdir logs/rsl_rl
```

3. Evaluate and export the trained policy:

```bash
# PM01
python scripts/amp/play.py --task AMP-Flat-PM01-v1 --num_envs 1 \
  --load_run RUN_DIRECTORY --checkpoint model_19999.pt

# T800
python scripts/amp/play.py --task AMP-Flat-T800-v1 --num_envs 1 \
  --load_run RUN_DIRECTORY --checkpoint model_19999.pt
```

## Deployment
### Install engineai_robotics_native_sdk
Both simulation and real-robot deployment require `engineai_robotics_native_sdk`. Refer to [engineai_robotics_native_sdk](https://github.com/engineai-robotics/engineai_robotics_native_sdk) for installation instructions.

### Sim2Sim
1. Prepare the deployment files:

- Copy `logs/rsl_rl/xx_flat/xxxx-xx-xx/exported/policy.mnn` to `assets/config/xxx/rl_dance_example(rl_lab)/policies`.
- Copy the NPZ motion file to `assets/config/xxx/rl_dance_example/trajectories`.
- Update `policy_file` and `trajectory_file_npz` in `assets/config/xxx/rl_dance_example(rl_lab)/default.yaml`.

2. Run the simulation:

```bash
# Terminal 1: run the MuJoCo simulation environment.
# Enter the container.
engineai_robotics_env
./scripts/run_mujoco.sh pm01_edu
# ./scripts/run_mujoco.sh t800

# Terminal 2: run the controller.
# Enter the container.
engineai_robotics_env
./run.sh pm01_edu
# ./run.sh t800

# Terminal 3: start the virtual gamepad or use a remote controller.
# Enter the container before starting the Python program.
engineai_robotics_env
python3 tools/virtual_gamepad/virtual_gamepad.py
```

![Gamepad control interface](docs/gamepad_en.png)

Press `LB+X` to enter AMP walking mode, or `RB+X` to enter whole-body tracking mode.

For remote-controller key mappings, refer to [engineai_robotics_native_sdk](https://github.com/engineai-robotics/engineai_robotics_native_sdk).

> When MuJoCo starts, the robot falls automatically. Switch to PD stand mode to restore the initial joint pose; this procedure is only recommended in simulation. Press Enter to reset the MuJoCo environment and place the robot in a standing state. You can then enter dance or AMP mode. After the motion finishes, the robot automatically switches to walk mode.

### Sim2Real
1. Edit the target robot settings in `install.sh` from [engineai_robotics_native_sdk](https://github.com/engineai-robotics/engineai_robotics_native_sdk):

```bash
remote_user="user"
remote_host="192.168.0.163"
remote_dir="~/projects/engineai_robotics"
```

2. Run the installation:

```bash
# Enter the container.
engineai_robotics_env

# Install the program.
./install.sh pm01_edu robot
# ./install.sh t800 robot
```

3. Run on the real robot.

> **Safety notice:**
> - Make sure the area is clear and everyone maintains a safe distance from the robot.
> - If the robot behaves abnormally, stop it immediately by pressing the emergency-stop button or switching back to `passive` mode.
> - Suspend the robot from a support frame first. Enter `pd_stand` mode, place the robot on the ground, and then switch to walking mode.

**Before running:**

- Enable the PM01 motor system with the emergency-stop button, or enable the T800 motor system with the remote controller.
- Connect to the robot hotspot.

**Startup steps:**

```bash
# 1. SSH into the robot (Nezha).
ssh user@192.168.0.163

# 2. Stop the auto-started motion-control program; otherwise native_sdk cannot start.
sudo systemctl stop robotics.service

# 3. Start native_sdk after ensuring that the motor system is enabled.
cd ~/projects/engineai_robotics
sudo ./run_robot.sh pm01_edu
# sudo ./run_robot.sh t800

# 4. Use the remote controller to switch modes and enter dance or AMP mode.
```

## License
This project is open-sourced under the **BSD 3-Clause License**. See [LICENSE](LICENSE.txt) for details.

## Acknowledgements
This project benefits from the support and contributions of the following open-source projects. We sincerely thank them:

- **[IsaacLab](https://github.com/isaac-sim/IsaacLab)** — The base framework for training and running simulation experiments.
- **[rsl_rl](https://github.com/leggedrobotics/rsl_rl)** — A high-performance reinforcement learning library for legged robots.
- **[BeyondMimic](https://github.com/HybridRobotics/whole_body_tracking)** — Inspiration for the project structure and a valuable implementation reference.
- **[MNN](https://github.com/alibaba/mnn)** — A lightweight, high-performance inference engine for edge deployment.
- **[Legged Lab](https://github.com/zitongbai/legged_lab)** — An Isaac Lab extension for legged-robot reinforcement learning that informed the structure and implementation of the AMP tasks in this project.
- **[MimicKit](https://github.com/xbpeng/MimicKit)** — A lightweight motion-imitation library that provided implementation references for AMP discriminator training, experience replay, and reference-motion sampling.
- **[AMP for Hardware](https://github.com/escontra/AMP_for_hardware)** — The implementation accompanying *Adversarial Motion Priors Make Good Substitutes for Complex Reward Functions*, used as a reference for deploying AMP on real robots.
