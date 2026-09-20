from __future__ import annotations

from isaaclab.actuators import ActuatorBaseCfg

from engineai_rl_lab.tasks.tracking.robots.actuator import DelayedImplicitActuatorCfg


def build_delayed_actuators(
    actuators: dict[str, ActuatorBaseCfg],
    min_delay: int,
    max_delay: int,
) -> dict[str, DelayedImplicitActuatorCfg]:
    """Rebuild implicit actuator configurations as delayed actuators.

    Args:
        actuators: The actuator configurations of the source articulation.
        min_delay: Minimum action delay in physics steps.
        max_delay: Maximum action delay in physics steps.
    """
    delayed_actuators = {}
    for name, cfg in actuators.items():
        delayed_actuators[name] = DelayedImplicitActuatorCfg(
            joint_names_expr=cfg.joint_names_expr,
            effort_limit=cfg.effort_limit,
            effort_limit_sim=cfg.effort_limit_sim,
            velocity_limit=cfg.velocity_limit,
            velocity_limit_sim=cfg.velocity_limit_sim,
            stiffness=cfg.stiffness,
            damping=cfg.damping,
            armature=cfg.armature,
            friction=cfg.friction,
            dynamic_friction=cfg.dynamic_friction,
            viscous_friction=cfg.viscous_friction,
            min_delay=min_delay,
            max_delay=max_delay,
        )
    return delayed_actuators
