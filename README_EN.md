# engineai_rl_lab
[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.1.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.3.2-silver)](https://isaac-sim.github.io/IsaacLab)

[中文](README.md)| English

## Overview
This project provides a set of reinforcement learning environments based on Isaac Lab. It currently supports the EngineAI PM01 and T800 robots and includes whole-body tracking tasks.

[Demo Video](https://www.bilibili.com/video/BV14GTk6UE9o/?share_source=copy_web&vd_source=74098b4d7182a602fcb3a63bb054e83f)

|Robot|Training| Sim2Sim |Deploy|
|:--------:|:--------:|:--------:|:--------:|
|||**whole body tracking**|||
|**T800**|<img src="./docs/train.gif" height="180"/>|<img src="./docs/sim2sim.gif" height="180"/>|<img src="./docs/deploy.gif" height="180"/>|
|**PM01**|<img src="./docs/train_pm.gif" height="180"/>|<img src="./docs/sim2sim_pm.gif" height="180"/>|<img src="./docs/deploy_pm.gif" height="180"/>|
|||**amp**|||
|**T800**|<img src="./docs/amp-t800-train.gif" height="180"/>|<img src="./docs/amp-t800-mujoco.gif" height="180"/>|<img src="./docs/amp-t800-deploy.gif" height="180"/>|
|**PM01**|<img src="./docs/amp-pm-train.gif" height="180"/>|<img src="./docs/amp-pm-mujoco.gif" height="180"/>|<img src="./docs/amp-pm-deploy.gif" height="180"/>|

## Installation
### Install Isaac Lab
This repository is developed against Isaac Lab 2.3.2 commit `c22775241e28f465fe345fa1a482ad6d29d712b0`. Code may not be compatible across different versions. For detailed installation instructions, refer to the official [Isaac Lab installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).

### Install engineai_rl_lab
1. Clone engineai_rl_lab from GitHub:
```bash
# 1. Clone the repository
git clone https://github.com/engineai-robotics/engineai_rl_lab.git
# 2. Download large assets
cd engineai_rl_lab
git lfs install
git lfs pull
```
2. Install engineai_rl_lab:
```bash
# Make sure the Isaac Lab environment is activated
cd engineai_rl_lab
pip install -e source/engineai_rl_lab
# Install MNN
pip install mnn
```

## Training
### Whole-Body Tracking
1. Convert CSV files to NPZ files

> The robot joint positions in the first and last frames of the whole-body tracking motion data should be approximately the same as those in PD stand. This helps ensure smooth motion when switching policies or modes.

```bash
# Convert CSV files to NPZ files; the NPZ files are saved in the same directory
# PM01
python scripts/csv_to_npz.py --robot pm01 --input_fps 30 -f datasets/tracking/pm01/dance.csv
# T800
python scripts/csv_to_npz.py --robot t800 --input_fps 30 -f datasets/tracking/t800/dance_t800.csv

# Replay NPZ files
# PM01
python scripts/replay_npz.py --robot pm01 --input_file datasets/tracking/pm01/dance.npz
# T800
python scripts/replay_npz.py --robot t800 --input_file datasets/tracking/t800/dance_t800.npz
```

2. Train
```bash
# PM01
python scripts/tracking/train.py --task Tracking-Flat-PM01-Wo-State-Estimation-v0 --headless --num_envs 4096 --motion_file datasets/tracking/pm01/dance.npz

# T800
python scripts/tracking/train.py --task Tracking-Flat-T800-Wo-State-Estimation-v0 --headless --num_envs 4096 --motion_file datasets/tracking/t800/dance_t800.npz

# View training logs
python -m tensorboard.main --logdir logs
```

3. Evaluate the trained policy and export it
```bash
# PM01
python scripts/tracking/play.py --task Tracking-Flat-PM01-Wo-State-Estimation-v0 --num_envs 1 --motion_file datasets/tracking/pm01/dance.npz --load_run 2026-06-23_09-58-43 --checkpoint dance.pt

# T800
python scripts/tracking/play.py --task Tracking-Flat-T800-Wo-State-Estimation-v0 --num_envs 1 --motion_file datasets/tracking/t800/dance_t800.npz --load_run 2026-06-28_20-47-15 --checkpoint dance.pt
```

### AMP

1. Check the reference motions

Before training, it is recommended to replay the data in Isaac Lab to verify the root pose, joint order, frame rate, and foot-ground contact. Use `--input_file` to check a single motion; `--input_folder` plays all NPZ files in a directory in filename order.

```bash
# PM01: play all motions in the dataset
python scripts/amp/replay_npz.py --robot pm01 --input_folder datasets/amp/pm01/data

# T800: play all motions in the dataset
python scripts/amp/replay_npz.py --robot t800 --input_folder datasets/amp/t800/data

# Single-motion example
python scripts/amp/replay_npz.py --robot pm01 --input_file datasets/amp/pm01/data/walk_001_Skeleton9.npz
```

2. Train

```bash
# PM01
python scripts/amp/train.py --task AMP-Flat-PM01-v1 --headless --num_envs 4096

# T800
python scripts/amp/train.py --task AMP-Flat-T800-v1 --headless --num_envs 4096

# View training curves
python -m tensorboard.main --logdir logs/rsl_rl
```

3. Evaluate the trained policy and export it

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
Both simulation and real-robot deployment require `engineai_robotics_native_sdk`. For installation instructions, refer to [engineai_robotics_native_sdk](https://github.com/engineai-robotics/engineai_robotics_native_sdk).

### Sim2Sim
1. Prepare the data
- Copy `logs/rsl_rl/xx_flat/xxxx-xx-xx/exported/policy.mnn` to the `assets/config/xxx/rl_dance_example(rl_lab)/policies` directory.
- Copy the NPZ motion file to the `assets/config/xxx/rl_dance_example/trajectories` directory.
- Edit the `policy_file` and `trajectory_file_npz` filenames in `assets/config/xxx/rl_dance_example(rl_lab)/default.yaml`.

2. Run the simulation
```bash
# Terminal 1: run the MuJoCo simulation environment
# Enter the container
engineai_robotics_env
./scripts/run_mujoco.sh pm01_edu
# ./scripts/run_mujoco.sh t800

# Terminal 2: run the controller
# Enter the container
engineai_robotics_env
./run.sh pm01_edu
# ./run.sh t800

# Terminal 3: start the virtual gamepad or use a remote controller
# Enter the container before starting the Python program
engineai_robotics_env
python3 tools/virtual_gamepad/virtual_gamepad.py
```
![Gamepad control interface](docs/gamepad.png)

Press LB+X to enter AMP walking test mode, or RB+X to enter whole-body tracking test mode.

For remote controller key mappings, refer to [engineai_robotics_native_sdk](https://github.com/engineai-robotics/engineai_robotics_native_sdk).
> After entering MuJoCo, the robot will automatically fall down. Switch to PD stand mode to restore the robot joints to their initial positions (this operation is only suitable for simulation), then press Enter to reset the MuJoCo environment and place the robot in a standing state. You can then enter dance mode. After the robot finishes the motion, it automatically switches to walk mode.

### Sim2Real
#### Whole-Body Tracking
1. Edit the target robot parameters in `install.sh` from [engineai_robotics_native_sdk](https://github.com/engineai-robotics/engineai_robotics_native_sdk):
```bash
remote_user="user"
remote_host="192.168.0.163"
remote_dir="~/projects/engineai_robotics"
```

2. Run the installation:
```bash
# Enter the container
engineai_robotics_env
# Install the program
./install.sh pm01_edu robot
# ./install.sh t800 robot
```

3. Run on the real robot
> **Safety Notice:**
> - Make sure the area is clear and that everyone keeps a safe distance from the robot.
> - If the robot behaves abnormally, stop it immediately by pressing the emergency stop button or switching back to `passive` mode.
> - It is recommended to suspend the robot with a support frame first, place it on the ground after entering `pd_stand` mode, and then switch to walking mode.

**Preparation:**
- Enable the robot motor system using the emergency stop button (PM01) or remote controller (T800).
- Connect to the robot hotspot.

**Startup Steps:**
```bash
# 1. SSH into the robot (Nezha)
ssh user@192.168.0.163

# 2. Stop the auto-started motion-control program; otherwise native_sdk cannot start
sudo systemctl stop robotics.service

# 3. Start native_sdk (make sure the motor system is enabled)
cd ~/projects/engineai_robotics
sudo ./run_robot.sh pm01_edu
# sudo ./run_robot.sh t800

# 4. Use the remote controller to switch modes and enter dance or AMP mode
```

## License
This project is open-sourced under the **BSD 3-Clause License**. See the [LICENSE](LICENSE.txt) file for details.

## Acknowledgements

This project benefits from the support and contributions of the following open-source projects. We sincerely thank them:

- **[IsaacLab](https://github.com/isaac-sim/IsaacLab)** — The base framework for training and running simulation experiments.
- **[rsl_rl](https://github.com/leggedrobotics/rsl_rl)** — A high-performance reinforcement learning library for legged robots.
- **[BeyondMimic](https://github.com/HybridRobotics/whole_body_tracking)** — Inspiration for the project structure and a valuable implementation reference.
- **[MNN](https://github.com/alibaba/mnn)** — A lightweight, high-performance inference engine for edge deployment.
- **[Legged Lab](https://github.com/zitongbai/legged_lab)** — An Isaac Lab-based reinforcement learning extension for legged robots that provided a reference for the structure and implementation of this project's AMP tasks.
- **[MimicKit](https://github.com/xbpeng/MimicKit)** — A lightweight motion imitation library that provided implementation references for AMP discriminator training, replay buffers, and reference motion sampling.
- **[AMP for Hardware](https://github.com/escontra/AMP_for_hardware)** — The code for the paper *Adversarial Motion Priors Make Good Substitutes for Complex Reward Functions*, which provided a reference for applying AMP to real robots.
