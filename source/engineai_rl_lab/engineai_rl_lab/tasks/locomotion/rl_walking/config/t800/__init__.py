import gymnasium as gym
from isaaclab.utils import configclass

from engineai_rl_lab.tasks.locomotion.rl_walking.agents import WalkingPPORunnerCfg

from .flat_env_cfg import T800FlatWalkingEnvCfg


@configclass
class T800FlatWalkingPPORunnerCfg(WalkingPPORunnerCfg):
    experiment_name = "velocity_flat_t800"


gym.register(
    id="RL-Walking-Flat-T800-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": T800FlatWalkingEnvCfg,
        "rsl_rl_cfg_entry_point": T800FlatWalkingPPORunnerCfg,
    },
)
