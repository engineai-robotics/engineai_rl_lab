# engineai_rl_lab
[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.1.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.3.2-silver)](https://isaac-sim.github.io/IsaacLab)

中文|[English](README_EN.md)

## 概览
本项目提供了一套基于Isaac Lab的强化学习环境，目前已经支持众擎PM01、T800机器人，实现的任务包括whole body tracking。

[演示视频](https://www.bilibili.com/video/BV14GTk6UE9o/?share_source=copy_web&vd_source=74098b4d7182a602fcb3a63bb054e83f)

|Robot|Training| Sim2Sim |Deploy|
|:--------:|:--------:|:--------:|:--------:|
|||**whole body tracking**|||
|**T800**|<img src="./docs/train.gif" height="180"/>|<img src="./docs/sim2sim.gif" height="180"/>|<img src="./docs/deploy.gif" height="180"/>|
|**PM01**|<img src="./docs/train_pm.gif" height="180"/>|<img src="./docs/sim2sim_pm.gif" height="180"/>|<img src="./docs/deploy_pm.gif" height="180"/>|
|||**amp**|||
|**T800**|<img src="./docs/amp-t800-train.gif" height="180"/>|<img src="./docs/amp-t800-mujoco.gif" height="180"/>|<img src="./docs/amp-t800-deploy.gif" height="180"/>|
|**PM01**|<img src="./docs/amp-pm-train.gif" height="180"/>|<img src="./docs/amp-pm-mujoco.gif" height="180"/>|<img src="./docs/amp-pm-deploy.gif" height="180"/>|

## 安装
### 安装Isaac Lab
本仓库基于Isaac Lab2.3.2的commit `c22775241e28f465fe345fa1a482ad6d29d712b0`进行开发,不同版本之间代码可能不通用。关于Isaac Lab的详细安装步骤，请参阅其官方安装指南[Isaac Lab](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)

### 安装engineai_rl_lab
1. 从github克隆engineai_rl_lab
```bash
# 1. 克隆仓库
git clone https://github.com/engineai-robotics/engineai_rl_lab.git
# 2.下载大型资产
cd engineai_rl_lab
git lfs install
git lfs pull
```
2. 安装engineai_rl_lab
```bash
# 请确保已经激活isaaclab环境
cd engineai_rl_lab
pip install -e source/engineai_rl_lab
# 安装mnn库
pip install mnn
```

## 训练
### whole body tracking
1. 将csv文件转换npz文件

> wholoe body tracking的动作数据的第一帧和最后一帧的机器人关节位置需要和pd stand下的基本一致，这有利于切换策略(模式)时的动作流畅性。

```bash
# csv文件转换为npz文件,npz文件在同一目录下
# PM01
python scripts/csv_to_npz.py --robot pm01 --input_fps 30 -f datasets/tracking/pm01/dance.csv
# T800
python scripts/csv_to_npz.py --robot t800 --input_fps 30 -f datasets/tracking/t800/dance_t800.csv

# 重放npz文件
# PMm01
python scripts/replay_npz.py --robot pm01 --input_file datasets/tracking/pm01/dance.npz
# T800
python scripts/replay_npz.py --robot t800 --input_file datasets/tracking/t800/dance_t800.npz
```

2. 训练
```bash
# PM01
python scripts/tracking/train.py --task Tracking-Flat-PM01-Wo-State-Estimation-v0 --headless --num_envs 4096 --motion_file datasets/tracking/pm01/dance.npz

# T800
python scripts/tracking/train.py --task Tracking-Flat-T800-Wo-State-Estimation-v0 --headless --num_envs 4096 --motion_file datasets/tracking/t800/dance_t800.npz

# 查看训练日志
python -m tensorboard.main --logdir logs
```

3. 验证训练效果并导出策略
```bash
# PM01
python scripts/tracking/play.py --task Tracking-Flat-PM01-Wo-State-Estimation-v0 --num_envs 1 --motion_file datasets/tracking/pm01/dance.npz --load_run 2026-06-23_09-58-43 --checkpoint dance.pt

# T800
python scripts/tracking/play.py --task Tracking-Flat-T800-Wo-State-Estimation-v0 --num_envs 1 --motion_file datasets/tracking/t800/dance_t800.npz --load_run 2026-06-28_20-47-15 --checkpoint dance.pt
```

### AMP

1. 检查参考动作

训练前建议先在 Isaac Lab 中重放数据，检查根节点位姿、关节顺序、帧率及脚底与地面的接触是否正常。`--input_file` 用于检查单个动作，`--input_folder` 会按文件名顺序播放目录中的所有 NPZ 文件。

```bash
# PM01：播放数据集中的所有动作
python scripts/amp/replay_npz.py --robot pm01 --input_folder datasets/amp/pm01/data

# T800：播放数据集中的所有动作
python scripts/amp/replay_npz.py --robot t800 --input_folder datasets/amp/t800/data

# 单个动作示例
python scripts/amp/replay_npz.py --robot pm01 --input_file datasets/amp/pm01/data/walk_001_Skeleton9.npz
```

2. 训练

```bash
# PM01
python scripts/amp/train.py --task AMP-Flat-PM01-v1 --headless --num_envs 4096

# T800
python scripts/amp/train.py --task AMP-Flat-T800-v1 --headless --num_envs 4096

# 查看训练曲线
python -m tensorboard.main --logdir logs/rsl_rl
```

3. 验证训练效果并导出策略

```bash
# PM01
python scripts/amp/play.py --task AMP-Flat-PM01-v1 --num_envs 1 \
  --load_run RUN_DIRECTORY --checkpoint model_19999.pt

# T800
python scripts/amp/play.py --task AMP-Flat-T800-v1 --num_envs 1 \
  --load_run RUN_DIRECTORY --checkpoint model_19999.pt
```

## 部署
### 安装engineai_robotics_native_sdk
仿真与真机部署均依赖安装engineai_robotics_native_sdk，具体安装教程请参阅[engineai_robotics_native_sdk](https://github.com/engineai-robotics/engineai_robotics_native_sdk)

### Sim2Sim
1. 数据准备
- 将`logs/rsl_rl/xx_flat/xxxx-xx-xx/exported/policy.mnn`复制到`assets/config/xxx/rl_dance_example(rl_lab)/policies`目录下
- 将npz动作文件复制到`assets/config/xxx/rl_dance_example/trajectories`目录下
- 修改`assets/config/xxx/rl_dance_example(rl_lab)/default.yaml`文件内的`policy_file`、`trajectory_file_npz`的文件名

2. 运行仿真
```bash
# 终端1：运行mujoco仿真环境
# 进入容器
engineai_robotics_env
./scripts/run_mujoco.sh pm01_edu
# ./scripts/run_mujoco.sh t800

# 终端2：运行控制程序
# 进入容器
engineai_robotics_env
./run.sh pm01_edu
# ./run.sh t800

# 终端3：启动虚拟手柄或使用遥控器
# 进入容器后再启动python程序
engineai_robotics_env
python3 tools/virtual_gamepad/virtual_gamepad.py
```
![手柄控制界面](docs/gamepad.png)
LB+X进入amp行走测试，RB+X进入whole body tracking测试

遥控器操作请参阅[engineai_robotics_native_sdk](https://github.com/engineai-robotics/engineai_robotics_native_sdk)中的键位。
> 进入mujoco后机器人会自动倒地，此时应切换到pd stand模式使机器人关节恢复到初始位置(仅仿真环境可以这样操作),按下键盘中的Enter即可重置mujoco环境，使机器人处于站立状态，这时可进入dance模式。机器人执行完动作之后自动会切换到walk状态。

### Sim2Real
#### whole body tracking
1. 编辑[engineai_robotics_native_sdk](https://github.com/engineai-robotics/engineai_robotics_native_sdk)中的`install.sh` 中的要部署目标机器人参数：
```bash
remote_user="user"
remote_host="192.168.0.163"
remote_dir="~/projects/engineai_robotics"
```

2. 执行安装：
```bash
# 进入容器
engineai_robotics_env
# 安装程序
./install.sh pm01_edu robot
# ./install.sh t800 robot
```

3. 真机运行
>  **安全提示：** 
> - 确保场地空旷，所有人员与机器人保持安全距离
> - 若机器人动作异常，随时快速停止（按急停键或切回 `passive` 模式）
> - 建议先用吊架吊起机器人，在进入 `pd_stand` 模式之后放到地上，再切入行走模式

**运行前准备：**
- 利用急停按键(pm01)或遥控器(t800)使能机器人的电机系统
- 连接机器人热点

**启动步骤：**
```bash
# 1. SSH 连接机器人（Nezha）
ssh user@192.168.0.163

# 2. 暂停自启动的运控程序，否则无法启动native_sdk
sudo systemctl stop robotics.service

# 3. 启动 native_sdk(确保已经使能电机系统)
cd ~/projects/engineai_robotics
sudo ./run_robot.sh pm01_edu
# sudo ./run_robot.sh t800

# 4. 使用遥控器切换模式，进入dance或amp模式
```

## 许可证
本项目采用 **BSD 3-Clause License** 开源协议。详见 [LICENSE](LICENSE.txt) 文件。

## 致谢

本项目得益于以下开源项目的支持和贡献，在此表示衷心的感谢：

- **[IsaacLab](https://github.com/isaac-sim/IsaacLab)** — 训练和运行仿真实验的基础框架。
- **[rsl_rl](https://github.com/leggedrobotics/rsl_rl)** — 适用于足式机器人的高性能强化学习库。
- **[BeyondMimic](https://github.com/HybridRobotics/whole_body_tracking)** — 项目结构启发及有价值的功能实现参考。
- **[MNN](https://github.com/alibaba/mnn)** — 轻量级高性能推理引擎，用于端侧部署。
- **[Legged Lab](https://github.com/zitongbai/legged_lab)** — 基于 Isaac Lab 的足式机器人强化学习扩展，为本项目的 AMP 任务结构与实现提供了参考。
- **[MimicKit](https://github.com/xbpeng/MimicKit)** — 轻量级动作模仿方法库，为 AMP 判别器训练、经验回放和参考动作采样提供了实现参考。
- **[AMP for Hardware](https://github.com/escontra/AMP_for_hardware)** — 论文 *Adversarial Motion Priors Make Good Substitutes for Complex Reward Functions* 的代码，为 AMP 在真实机器人上的应用提供了参考。
