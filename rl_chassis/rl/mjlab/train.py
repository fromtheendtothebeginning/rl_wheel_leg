#!/usr/bin/env python3
"""训练入口：注册任务后复用 mjlab 的训练管线。

    uv run python train.py --num-envs 1024 --max-iterations 600 --run-name flat_v1
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--num-envs", type=int, default=1024)
  parser.add_argument("--max-iterations", type=int, default=600)
  parser.add_argument("--seed", type=int, default=None)
  parser.add_argument("--run-name", type=str, default=None)
  parser.add_argument("--log-root", type=str, default="logs/rsl_rl")
  parser.add_argument("--task", choices=("flat", "jump", "solid"), default="flat",
                      help="flat=平地任务（凸分解碰撞）；solid=平地任务（每 link "
                           "实心凸包碰撞）；jump=冻结平地策略+跳跃残差")
  parser.add_argument("--resume", action="store_true",
                      help="从 experiment 下最新 run 的最新 checkpoint 续训"
                           "（含优化器状态；每 50 轮存一次，最多丢 50 轮进度）")
  parser.add_argument("--load-run", type=str, default=None,
                      help="显式指定 load_run（如 distill_wxk 从蒸馏基座重开）；"
                           "缺省仍自动挑最新含 checkpoint 的 run")
  parser.add_argument("--start-stage", type=int, default=None,
                      help="覆盖课程的起始 stage（curriculum 不进 checkpoint，"
                           "从 stage 1 续训时传 1，避免回到 stage 0 分布）")
  args = parser.parse_args()

  import mjlab_wheelleg  # noqa: F401  注册任务

  from mjlab.scripts.train import TrainConfig, launch_training
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
  import mjlab_wheelleg as _wl

  TASK_ID = {"jump": _wl.TASK_ID_JUMP, "solid": _wl.TASK_ID_SOLID}.get(
    args.task, _wl.TASK_ID)

  env_cfg = load_env_cfg(TASK_ID)
  if args.start_stage is not None:
    env_cfg.curriculum["command_standing"].params["start_stage"] = args.start_stage
  agent_cfg = load_rl_cfg(TASK_ID)
  env_cfg.scene.num_envs = args.num_envs
  agent_cfg.max_iterations = args.max_iterations
  if args.seed is not None:
    agent_cfg.seed = args.seed
  if args.run_name:
    agent_cfg.run_name = args.run_name
  if args.resume:
    agent_cfg.resume = True
    # load_run 默认 ".*" 会命中刚创建的新 run 目录（空、无 checkpoint），
    # 这里显式指向最新的**含有 checkpoint** 的 run 目录。
    import glob as _glob
    import os as _os
    exp_dir = _os.path.join(args.log_root, agent_cfg.experiment_name)
    if args.load_run:
      agent_cfg.load_run = args.load_run
      agent_cfg.load_checkpoint = "model_.*.pt"
    else:
      runs_with_ckpt = [d for d in _glob.glob(_os.path.join(exp_dir, "*"))
                        if _glob.glob(_os.path.join(d, "model_*.pt"))]

      def _newest_model_time(d):
        # 用 run 内最新 model 文件的 mtime：改写文件不会更新目录 mtime，
        # 只看目录会把"旧目录里刚被替换的 checkpoint"排在后面（踩过）。
        return max(_os.path.getmtime(p) for p in _glob.glob(_os.path.join(d, "model_*.pt")))

      if runs_with_ckpt:
        agent_cfg.load_run = _os.path.basename(max(runs_with_ckpt, key=_newest_model_time))

  cfg = TrainConfig(
    env=env_cfg,
    agent=agent_cfg,
    log_root=str(Path(args.log_root).resolve()),
  )
  launch_training(task_id=TASK_ID, args=cfg)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
