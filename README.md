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
```bash
# csv文件转换为npz文件,npz文件在同一目录下
# PM01
python scripts/csv_to_npz.py --robot pm01 --input_fps 30 -f datasets/tracking/pm01/dance.csv
# T800
python scripts/csv_to_npz.py --robot t800 --input_fps 30 -f datasets/tracking/t800/dance_t800.csv
# T800pro
python scripts/csv_to_npz.py --robot t800pro --input_fps 30 -f datasets/tracking/t800pro/dance_t800pro.csv

# 重放npz文件
# PMm01
python scripts/replay_npz.py --robot pm01 --input_file datasets/tracking/pm01/dance.npz
# T800
python scripts/replay_npz.py --robot t800 --input_file datasets/tracking/t800/dance_t800.npz
# T800pro
python scripts/replay_npz.py --robot t800pro --input_file datasets/tracking/t800pro/dance_t800.npz
```

2. 训练
```bash
# PM01
python scripts/tracking/train.py --task Tracking-Flat-PM01-Wo-State-Estimation-v0 --headless --num_envs 4096 --motion_file datasets/tracking/pm01/dance.npz

# T800
python scripts/tracking/train.py --task Tracking-Flat-T800-Wo-State-Estimation-v0 --headless --num_envs 4096 --motion_file datasets/tracking/t800/dance_t800.npz

# T800pro
python scripts/tracking/train.py --task Tracking-Flat-T800pro-Wo-State-Estimation-v0 --headless --num_envs 4096 --motion_file datasets/tracking/t800pro/dance_t800pro.npz

# 查看训练日志
python -m tensorboard.main --logdir logs
```

3. 验证训练效果并导出策略
```bash
# PM01
python scripts/tracking/play.py --task Tracking-Flat-PM01-Wo-State-Estimation-v0 --num_envs 1 --motion_file datasets/tracking/pm01/dance.npz --load_run 2026-06-23_09-58-43 --checkpoint dance.pt

# T800
python scripts/tracking/play.py --task Tracking-Flat-T800-Wo-State-Estimation-v0 --num_envs 1 --motion_file datasets/tracking/t800/dance_t800.npz --load_run 2026-06-28_20-47-15 --checkpoint dance.pt

# T800pro
```

## 部署
### 安装engineai_robotics_native_sdk
仿真与真机部署均依赖安装engineai_robotics_native_sdk，具体安装教程请参阅[engineai_robotics_native_sdk](https://github.com/engineai-robotics/engineai_robotics_native_sdk)

### Sim2Sim
#### whole body tracking
1. 数据准备
- 将`logs/rsl_rl/xx_flat/xxxx-xx-xx/exported/policy.mnn`复制到`assets/config/xxx/rl_dance_example/policies`目录下
- 将npz动作文件复制到`assets/config/xxx/rl_dance_example/trajectories`目录下
- 修改`assets/config/xxx/rl_dance_example/default.yaml`文件内的`policy_file`、`trajectory_file_npz`的文件名

> 请保证动作数据的第一帧和最后一帧的机器人关节位置和pd stand下的基本一致，这有利于切换策略(模式)时的动作流畅性。

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
./install pm01_edu robot
# ./install t800 robot
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

# 4. 使用遥控器切换模式，进入dance模式
```
## TODO
- [ ] 添加AMP拟人行走
- [ ] 添加基于Direct RL Environment的行走训练
- [ ] 完善文档与教程

## 技术支持
如果您在使用本项目过程中遇到任何问题，请在本项目的GitHub仓库中提交Issue，我们会尽快回复。同时，也欢迎您加入我们的众擎机器人开发者交流微信群。
<div align="center"> <img src="docs/weixin.png" height="300" alt="微信交流群"/> <br> <em>扫码加入众擎机器人开发者交流微信群</em> </div>

## 许可证
本项目采用 **BSD 3-Clause License** 开源协议。详见 [LICENSE](LICENSE.txt) 文件。

## 致谢
本项目得益于以下开源项目的支持和贡献，在此表示衷心的感谢：
- **[IsaacLab](https://github.com/isaac-sim/IsaacLab)** — 训练和运行仿真实验的基础框架。
- **[rsl_rl](https://github.com/leggedrobotics/rsl_rl)** — 适用于足式机器人的高性能强化学习库。
- **[BeyondMimic](https://github.com/HybridRobotics/whole_body_tracking)** — 项目结构启发及有价值的功能实现参考。
- **[MNN](https://github.com/alibaba/mnn)** — 轻量级高性能推理引擎，用于端侧部署。
