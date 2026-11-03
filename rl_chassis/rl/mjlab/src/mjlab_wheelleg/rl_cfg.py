"""RSL-RL / PPO 配置：与 Isaac Lab 版 v12 stage1 相同的网络与超参。"""

from __future__ import annotations

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def wheelleg_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      # 输入 = 25 维观测 × 5 帧历史 = 125；隐藏层对齐 plane_v11 的 256³。
      hidden_dims=(256, 256, 256),
      activation="elu",
      obs_normalization=False,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 0.5,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(256, 256, 256),
      activation="elu",
      obs_normalization=False,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,  # 与 plane_v11 原版一致
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",  # 与 plane_v11 原版一致（desired_kl=0.005）
      gamma=0.99,
      lam=0.95,
      desired_kl=0.005,
      max_grad_norm=1.0,
    ),
    experiment_name="wheelleg_v12_mjlab",
    save_interval=50,
    num_steps_per_env=48,
    max_iterations=600,
    clip_actions=100.0,
    logger="tensorboard",  # wandb 0.31 与 rsl-rl 5.4 的 Settings 不兼容
  )


def wheelleg_jump_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """跳跃残差 actor：42 维输入 -> 6 维残差（冻结平地策略另算）。

  探索 σ=1.0：蹬伸需要残差幅度 ±2~4，σ=0.3 时探索永远够不到（700 轮
  line_z 恒 0）。残差只在 ~1s 跳跃窗口内生效（FLAT 相位缩放为 0），
  大 σ 不会污染站立/走行。
  """
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(256, 256, 256),
      activation="elu",
      obs_normalization=False,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(256, 256, 256),
      activation="elu",
      obs_normalization=False,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.02,  # 维持残差探索（蹬伸动作幅度大，靠大 σ 随机发现）
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.005,
      max_grad_norm=1.0,
    ),
    experiment_name="wheelleg_jump_v12",
    save_interval=50,
    num_steps_per_env=48,
    max_iterations=3000,
    clip_actions=100.0,
    logger="tensorboard",
  )
