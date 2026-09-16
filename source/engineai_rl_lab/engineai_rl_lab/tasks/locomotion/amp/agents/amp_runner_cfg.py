"""Runner configuration shared by the AMP locomotion agents.

All AMP hyperparameters live in :class:`RslRlAmpCfg` and are consumed by
``AMP.construct_algorithm`` through the nested ``amp`` entry of the runner config, so a
robot agent config only needs to override the values it cares about.
"""

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

AMP_ALGORITHM_CLASS = "engineai_rl_lab.tasks.locomotion.amp.algorithms.amp:AMP"


@configclass
class RslRlAmpCfg:
    """Adversarial Motion Priors hyperparameters."""

    # Reference dataset

    dataset_path: str = MISSING
    """Path to the expert motion dataset YAML, relative to the repository root."""

    include_joint_vel: bool = True
    """Whether the discriminator frame contains joint velocities."""

    include_base_lin_vel: bool = False
    """Whether the discriminator frame contains the base linear velocity."""

    include_projected_gravity: bool = False
    """Whether the discriminator frame contains the projected gravity vector."""

    include_foot_features: bool = False
    """Whether the discriminator frame contains foot positions and velocities.

    Requires ``foot_body_indices`` in the dataset YAML.
    """

    # Discriminator observation and network

    amp_obs_group: str = "disc"
    """Observation group holding the agent's discriminator observation."""

    frame_length: int = 10
    """Number of physical frames per discriminator observation.

    Must match the ``history_length`` of the ``amp_obs_group`` observation group.
    """

    frame_dim: int = MISSING
    """Feature dimension of a single discriminator frame.

    Must match the per-frame width produced by the ``amp_obs_group`` observation group and
    by the dataset ``include_*`` flags above.
    """

    flatten_history_dim: bool = False
    """Whether the frame history is flattened before entering the discriminator.

    Must match ``flatten_history_dim`` of the ``amp_obs_group`` observation group.
    """

    frame_normalization: bool = True
    """Whether the discriminator normalizes its input features with running statistics."""

    frame_normalization_clip: float = 10.0
    """Clamp applied to the normalized discriminator features. 0 disables the clamp."""

    discriminator_hidden_dims: list[int] = [512, 256, 128]
    """Hidden layer sizes of the discriminator trunk."""

    # Reward shaping

    style_reward_scale: float = 2.0
    """Scale applied to the style reward before blending it with the task reward."""

    task_style_lerp: float = 0.5
    """Task reward weight in ``lerp(style, task)``. 1.0 disables the style reward."""

    step_dt: float | None = None
    """Control step used to scale the style reward. ``None`` resolves it from the env."""

    # Discriminator optimization

    loss_type: str = "GAN"
    """Adversarial objective. One of ``"GAN"``, ``"LSGAN"`` or ``"WGAN"``."""

    grad_penalty_scale: float = 10.0
    """Weight of the gradient penalty, applied to both the expert and the agent samples."""

    disc_logit_reg: float = 0.01
    """Weight of the L2 penalty on the discriminator output layer. 0 disables it."""

    disc_learning_rate: float = 2.5e-4
    """Learning rate of the discriminator optimizer."""

    disc_weight_decay: float = 1.0e-4
    """Weight decay of the discriminator optimizer."""

    disc_max_grad_norm: float = 1.0
    """Gradient clipping norm of the discriminator update."""

    disc_epochs: int = 2
    """Number of passes over the rollout per discriminator update phase."""

    disc_batch_size: float = 2.0
    """Discriminator batch size, as a *multiple of the number of environments*.

    Following MimicKit, the gradient step count of one update phase is
    ``ceil(num_samples / ceil(disc_batch_size * num_envs)) * disc_epochs``.
    """

    disc_update_interval: int = 2
    """Number of policy iterations between two discriminator update phases.

    The discriminator is decoupled from the PPO mini-batch loop, so this throttles the whole
    update phase: with the defaults, 24 gradient steps run every 2 iterations.
    """

    # Discriminator replay

    disc_replay_buffer_size: int = 200000
    """Capacity, in frames, of the policy replay buffer feeding negative samples.

    Costs ``capacity * frame_length * frame_dim * 4`` bytes of device memory, i.e. roughly
    390 MB at the default capacity with a ``10 x 49`` discriminator observation.
    """

    disc_replay_samples: int = 1000
    """Frames injected into the replay buffer per iteration once it is full.

    Before it fills up the whole rollout is inserted.
    """

    # Conditional AMP

    condition_dim: int = 0
    """Command conditioning dimension. Either 0 (unconditional) or 3 (``[vx, vy, wz]``)."""

    amp_condition_group: str = "amp_command"
    """Observation group holding the conditioning command. Unused when ``condition_dim`` is 0."""


@configclass
class AmpBasePpoRunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner defaults shared by the AMP locomotion agents."""

    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 2000
    experiment_name = MISSING
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
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
    amp: RslRlAmpCfg = RslRlAmpCfg()

    def __post_init__(self):
        super().__post_init__()

        self.algorithm.class_name = AMP_ALGORITHM_CLASS

        # Remove deprecated fields for rsl-rl >= 5.0.0.
        deprecated_keys = {"stochastic", "init_noise_std", "noise_std_type", "state_dependent_std"}

        def _remove_deprecated_keys(cfg_obj):
            if cfg_obj is None:
                return None
            return {key: value for key, value in vars(cfg_obj).items() if key not in deprecated_keys}

        self.actor = _remove_deprecated_keys(self.actor)
        self.critic = _remove_deprecated_keys(self.critic)
