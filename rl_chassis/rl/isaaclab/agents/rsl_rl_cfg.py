"""RSL-RL configuration for the wheel-legged flat-ground task.

Actor/critic sizes follow ``plane_v11/wheel_legged_gym`` 的 v4 配置
（actor [128, 64, 32]、critic [256, 128, 64]、ELU、初始动作标准差 0.5）。
这一轮不接 history encoder：Stage 1 先用 25 维当前观测把平地站稳，
sim2sim 侧再按单输入策略处理。
"""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class WheelLeggedPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 48
    max_iterations = 600
    save_interval = 50
    experiment_name = "wheel_legged_isaaclab"
    clip_actions = 100.0

    actor = RslRlMLPModelCfg(
        hidden_dims=[128, 64, 32],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.5),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[256, 128, 64],
        activation="elu",
        obs_normalization=False,
    )
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
        desired_kl=0.005,
        max_grad_norm=1.0,
    )
