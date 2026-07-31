from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg,RslRlMLPModelCfg,RslRlPpoAlgorithmCfg


@configclass
class PM01BasePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 2000
    experiment_name = "amp_velocity_flat_pm01"
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
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
        distribution_cfg=
            RslRlMLPModelCfg.GaussianDistributionCfg(
                init_std=1.0,
                std_type="scalar"
            )
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
    )


    def __post_init__(self):
        super().__post_init__()
        
        # 2. deprecated key words for rsl-rl >= 5.0.0  
        deprecated_keys = {"stochastic", "init_noise_std", "noise_std_type", "state_dependent_std"}
        
        def _remove_deprecated_keys(cfg_obj):
            if cfg_obj is None:
                return None
            return {k: v for k, v in vars(cfg_obj).items() if k not in deprecated_keys}
        
        self.actor = _remove_deprecated_keys(self.actor)
        self.critic = _remove_deprecated_keys(self.critic)
