from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class T800BasePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 2000
    experiment_name = "amp_velocity_flat_t800"
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.008,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            init_std=1.0,
            std_type="scalar",
        ),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
    )

    def __post_init__(self):
        super().__post_init__()

        # Remove deprecated fields for rsl-rl >= 5.0.0.
        deprecated_keys = {"stochastic", "init_noise_std", "noise_std_type", "state_dependent_std"}

        def _remove_deprecated_keys(cfg_obj):
            if cfg_obj is None:
                return None
            return {key: value for key, value in vars(cfg_obj).items() if key not in deprecated_keys}

        self.actor = _remove_deprecated_keys(self.actor)
        self.critic = _remove_deprecated_keys(self.critic)

@configclass
class T800FlatAMPPPORunnerCfg_V1(T800BasePPORunnerCfg):
    max_iterations: int = 20000
    save_interval: int = 2000
    # AMP parameters
    style_reward_scale: float = 2.0
    task_style_lerp: float = 0.4
    frame_length: int = 5
    frame_dim: int = 32
    frame_normalization: bool = True
    discriminator_hidden_dims: list[int] = [512, 256, 128]
    experiment_name: str = "amp_velocity_flat_t800"
    dataset_path: str = "datasets/amp/t800/config/dataset.yaml"

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.class_name = "engineai_rl_lab.tasks.locomotion.amp.algorithms.amp_ppo_v2:AMPPPOV2"


@configclass
class T800FlatPPORunnerCfg(T800BasePPORunnerCfg):
    max_iterations = 20000
    experiment_name = "amp_velocity_flat_t800"

    def __post_init__(self):
        super().__post_init__()
        self.actor["hidden_dims"] = [128, 128, 128]
        self.critic["hidden_dims"] = [128, 128, 128]
