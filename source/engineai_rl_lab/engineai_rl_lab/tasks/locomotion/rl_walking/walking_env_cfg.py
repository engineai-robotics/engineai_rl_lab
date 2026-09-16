"""Shared observation configuration for non-AMP velocity-tracking tasks."""

from isaaclab.utils import configclass

from engineai_rl_lab.tasks.locomotion.amp.amp_env_cfg import AmpObservationsCfg


@configclass
class WalkingObservationsCfg:
    """Actor and critic observations without the AMP discriminator group."""

    policy: AmpObservationsCfg.PolicyCfg = AmpObservationsCfg.PolicyCfg()
    critic: AmpObservationsCfg.CriticCfg = AmpObservationsCfg.CriticCfg()
