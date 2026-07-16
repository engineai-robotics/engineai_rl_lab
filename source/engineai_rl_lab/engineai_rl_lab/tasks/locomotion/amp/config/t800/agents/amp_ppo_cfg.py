from isaaclab.utils import configclass

from .rsl_rl_ppo_cfg import T800BasePPORunnerCfg


@configclass
class T800FlatAMPPPORunnerCfg(T800BasePPORunnerCfg):
    max_iterations: int = 20000
    save_interval: int = 2000
    # AMP parameters
    # effective style reward = 0.01 * style_reward_weight * r_style (see amp_ppo.py).
    # raise this so the style reward is comparable to the task reward, otherwise the
    # policy ignores the motion data entirely.
    style_reward_weight = 50.0
    frame_length = 5
    frame_dim = 26
    frame_normalization = True
    discriminator_hidden_dims = [512, 256, 128]
    experiment_name = "amp_velocity_flat_t800"
    dataset_path = "datasets/amp/t800/config/dataset.yaml"

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.class_name = "engineai_rl_lab.tasks.locomotion.amp.algorithms.amp_ppo:AMPPPO"

@configclass
class T800FlatAMPPPORunnerCfg_V1(T800BasePPORunnerCfg):
    max_iterations: int = 20000
    save_interval: int = 2000
    # AMP parameters
    style_reward_weight = 2.0
    frame_length = 5
    frame_dim = 26
    frame_normalization = True
    discriminator_hidden_dims = [512, 256, 128]
    experiment_name = "amp_velocity_flat_t800"
    dataset_path = "datasets/amp/t800/config/dataset.yaml"

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.class_name = "engineai_rl_lab.tasks.locomotion.amp.algorithms.amp_ppo_v2:AMPPPOV2"
