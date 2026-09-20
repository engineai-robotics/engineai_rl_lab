"""T800 flat-ground velocity-tracking environment without AMP."""

from isaaclab.utils import configclass

from engineai_rl_lab.tasks.locomotion.amp.config.t800.flat_amp_env_cfg import T800AMPFlatEnvCfg
from engineai_rl_lab.tasks.locomotion.rl_walking.walking_env_cfg import WalkingObservationsCfg


@configclass
class T800FlatWalkingEnvCfg(T800AMPFlatEnvCfg):
    """T800 task rewards and dynamics with no discriminator observation."""

    observations: WalkingObservationsCfg = WalkingObservationsCfg()
