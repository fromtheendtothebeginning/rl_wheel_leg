"""v12 轮腿 mjlab 任务注册。import 本模块即注册（train.py 会 import 它）。"""

from __future__ import annotations

from mjlab.tasks.registry import register_mjlab_task

from .env_cfg import V12_SOLID_XML, wheelleg_flat_env_cfg, wheelleg_jump_env_cfg
from .rl_cfg import wheelleg_jump_ppo_runner_cfg, wheelleg_ppo_runner_cfg

TASK_ID = "WheelLegged-Flat-v12-Closed"
TASK_ID_JUMP = "WheelLegged-Jump-v12-Closed"
TASK_ID_SOLID = "WheelLegged-Flat-v12-Solid"

# resume 会把 checkpoint 里的 optimizer lr 一并带回（adaptive 时代被踩到 1e-5），
# 且 PPO.load 会把 self.learning_rate 覆盖成加载值；fixed 调度下 rsl-rl 不会再改。
# 加载后强制重置回**配置里的目标 lr**（不是加载值）。
from rsl_rl.algorithms.ppo import PPO

from .rl_cfg import wheelleg_ppo_runner_cfg as _runner_cfg_fn

_TARGET_LR = float(_runner_cfg_fn().algorithm.learning_rate)
_orig_ppo_load = PPO.load


def _ppo_load_with_lr_reset(self, *args, **kwargs):
  out = _orig_ppo_load(self, *args, **kwargs)
  self.learning_rate = _TARGET_LR
  for group in self.optimizer.param_groups:
    group["lr"] = _TARGET_LR
  return out


PPO.load = _ppo_load_with_lr_reset

register_mjlab_task(
  task_id=TASK_ID,
  env_cfg=wheelleg_flat_env_cfg(),
  play_env_cfg=wheelleg_flat_env_cfg(play=True),
  rl_cfg=wheelleg_ppo_runner_cfg(),
)

register_mjlab_task(
  task_id=TASK_ID_JUMP,
  env_cfg=wheelleg_jump_env_cfg(),
  play_env_cfg=wheelleg_jump_env_cfg(play=True),
  rl_cfg=wheelleg_jump_ppo_runner_cfg(),
)

# 实心碰撞变体（hybrid：运动 link 实心凸包，base 保留腿坑凸分解）。
# 同一套奖励/课程/观测，只换 MJCF 与日志目录；额外加 0.07 m 低伏终止。
# 15.641 kg 模型零动作会自然沉到 0.073 m，0.12 m 会把早期探索全部判死；
# 0.07 m 仍能挡住全凸包版 0.056~0.066 m 的折腿贴地解。
_solid_runner = wheelleg_ppo_runner_cfg()
_solid_runner.experiment_name = "wheelleg_v12_solid"
# 重质量从零训练时 adaptive 调度会因高 KL 把 LR 踩到 1e-5（实测
# iter 12~36 卡在 1e-5，ep_len 328→64、站高 0.153→0.098）。solid
# 任务改用 fixed 1e-3；与 __init__ 里的 _TARGET_LR 一致，resume 也安全。
_solid_runner.algorithm.schedule = "fixed"
_solid_runner.algorithm.learning_rate = 1.0e-3

register_mjlab_task(
  task_id=TASK_ID_SOLID,
  env_cfg=wheelleg_flat_env_cfg(
    xml_path=V12_SOLID_XML, min_base_height=0.07,
    # 15.641 kg + 现有 kp=40 的实测上限约 0.165 m：0.19 指令站不到，
    # 站高曲线会一直有 2.5 cm 残差且 settle 不达标。目标改 0.16，
    # stage1 放宽到 0.12~0.18（更低站姿可达，0.24 不可达）。
    height_ranges=((0.16, 0.16), (0.12, 0.18)),
    cmd_height_range=(0.12, 0.18),
    # 治 POM 导轮支撑的 27° 伪站姿（v8 ep_len 2000 但 bad_contact -0.975）。
    bad_contact_termination=True),
  play_env_cfg=wheelleg_flat_env_cfg(
    play=True, xml_path=V12_SOLID_XML, min_base_height=0.07,
    height_ranges=((0.16, 0.16), (0.12, 0.18)),
    cmd_height_range=(0.12, 0.18),
    bad_contact_termination=True),
  rl_cfg=_solid_runner,
)
