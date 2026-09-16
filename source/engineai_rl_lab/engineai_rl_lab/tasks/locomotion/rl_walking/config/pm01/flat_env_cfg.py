"""PM01 flat-ground velocity-tracking environment without AMP."""

from isaaclab.utils import configclass

from engineai_rl_lab.tasks.locomotion.amp.config.pm01.flat_amp_env_cfg import PM01AMPFlatEnvCfg
from engineai_rl_lab.tasks.locomotion.rl_walking.walking_env_cfg import WalkingObservationsCfg


@configclass
class PM01FlatWalkingEnvCfg(PM01AMPFlatEnvCfg):
    """PM01 task rewards and dynamics with no discriminator observation."""

    observations: WalkingObservationsCfg = WalkingObservationsCfg()
