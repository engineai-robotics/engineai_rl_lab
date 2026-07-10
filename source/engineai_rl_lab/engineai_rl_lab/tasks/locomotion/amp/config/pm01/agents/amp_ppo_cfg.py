from isaaclab.utils import configclass

from .rsl_rl_ppo_cfg import PM01BasePPORunnerCfg


@configclass
class PM01FlatAMPPPORunnerCfg(PM01BasePPORunnerCfg):
    max_iterations: int = 20000
    save_interval: int = 2000
    # AMP parameters
    style_reward_weight = 2.0
    frame_length = 5
    frame_dim = 26
    frame_normalization = True
    discriminator_hidden_dims = [512, 256, 128]
    experiment_name = "amp_velocity_flat_pm01"
    dataset_path = "datasets/amp/pm01/config/dataset.yaml"

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.class_name = "engineai_rl_lab.tasks.locomotion.amp.algorithms.amp_ppo:AMPPPO"
