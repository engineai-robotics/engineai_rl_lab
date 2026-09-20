from isaaclab.utils import configclass

from engineai_rl_lab.tasks.locomotion.amp.agents import AmpBasePpoRunnerCfg, RslRlAmpCfg


@configclass
class PM01FlatAMPPPORunnerCfg(AmpBasePpoRunnerCfg):
    experiment_name = "amp_velocity_flat_pm01"
    max_iterations = 20000
    save_interval = 2000

    amp = RslRlAmpCfg(
        dataset_path="datasets/amp/pm01/config/dataset.yaml",
        frame_length=10,
        frame_dim=55,
        include_joint_vel=True,
        include_base_lin_vel=True,
        include_projected_gravity=True,
        amp_obs_group="disc",
        flatten_history_dim=False,
        frame_normalization=True,
        discriminator_hidden_dims=[512, 256, 128],
        style_reward_scale=7.5,
        task_style_lerp=0.4,
        loss_type="LSGAN",
        grad_penalty_scale=10.0,
        disc_epochs=1,
        disc_update_interval=4,
        disc_replay_samples=4096,
    )
