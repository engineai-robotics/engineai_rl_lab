import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

# gym.register(
#     id="AMP-Flat-T800-v0",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.flat_amp_env_cfg:T800AMPFlatEnvCfg",
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.amp_ppo_cfg:T800FlatAMPPPORunnerCfg",
#     },
# )

gym.register(
    id="AMP-Flat-T800-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_amp_env_cfg:T800AMPFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.amp_ppo_cfg:T800FlatAMPPPORunnerCfg_V1",
    },
)