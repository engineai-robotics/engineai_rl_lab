import gymnasium as gym

from . import agents

gym.register(
    id="AMP-Flat-PM01-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_amp_env_cfg:PM01AMPFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.amp_ppo_cfg:PM01FlatAMPPPORunnerCfg",
    },
)
