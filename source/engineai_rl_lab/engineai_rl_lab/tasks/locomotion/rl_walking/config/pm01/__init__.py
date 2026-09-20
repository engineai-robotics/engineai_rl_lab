import gymnasium as gym
from isaaclab.utils import configclass

from engineai_rl_lab.tasks.locomotion.rl_walking.agents import WalkingPPORunnerCfg

from .flat_env_cfg import PM01FlatWalkingEnvCfg


@configclass
class PM01FlatWalkingPPORunnerCfg(WalkingPPORunnerCfg):
    experiment_name = "velocity_flat_pm01"


gym.register(
    id="RL-Walking-Flat-PM01-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": PM01FlatWalkingEnvCfg,
        "rsl_rl_cfg_entry_point": PM01FlatWalkingPPORunnerCfg,
    },
)
