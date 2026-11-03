#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""训练观测窗：MuJoCo viewer 实时跟随最新训练 checkpoint。

一边训练一边看：本脚本只占 CPU（单环境仿真 + CPU 推理），完全不碰训练的 GPU。
每 30 s 扫一遍 logs/rsl_rl/wheelleg_v12_mjlab/<experiment>/<run>/model_*.pt，
发现更新的 checkpoint 就热替换策略权重并重置场景，从出生姿态看新策略。

用法（在 mjlab/ 目录下，用 .venv 的 python 跑，有 torch+mujoco）：
    .venv/bin/python ../mujoco/watch_training.py --run flat_curri2
    .venv/bin/python ../mujoco/watch_training.py --run flat_curri2 --interval 15
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
import time

import numpy as np
import torch
from torch import nn

HERE = os.path.dirname(os.path.abspath(__file__))
MJLAB_DIR = os.path.abspath(os.path.join(HERE, "..", "mjlab"))
sys.path.insert(0, HERE)
sys.path.insert(0, MJLAB_DIR)

from onnx_mj_v11 import CLIP_ACTIONS, POLICY_DT, SIM_DT, build_parser  # noqa: E402
from onnx_mj_v12_hist import V12HistSim2Sim  # noqa: E402

V12_SCENE = os.path.abspath(os.path.join(HERE, "..", "..", "mjcf", "rl_chassis_v12_scene.xml"))


class TorchPolicy:
    """onnxruntime 兼容壳：.run(None, {'obs': x}) -> [actions]，支持热替换权重。"""

    def __init__(self, num_obs: int = 125, hidden=(256, 256, 256), num_actions: int = 6):
        dims = [num_obs, *hidden, num_actions]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ELU())
        self.net = nn.Sequential(*layers)
        self.net.eval()
        self.in_names = ["obs"]

    def load(self, ckpt_path: str) -> int:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = ckpt.get("actor_state_dict", ckpt.get("model_state_dict", ckpt))
        mapped = {}
        for k, v in state.items():
            if k.startswith("mlp."):
                mapped[k[len("mlp."):]] = v.float()  # Sequential 键无 mlp. 前缀
        self.net.load_state_dict(mapped, strict=True)
        return int(ckpt.get("iter", -1))

    @torch.no_grad()
    def run(self, _ignored, feed):
        x = torch.from_numpy(feed["obs"])
        return [self.net(x).numpy()]


def find_latest(run_name: str, experiment: str = "wheelleg_v12_mjlab"):
  root = os.path.join(MJLAB_DIR, "logs", "rsl_rl", experiment)
  runs = sorted(glob.glob(os.path.join(root, f"*_{run_name}")), key=os.path.getmtime)
  if not runs:
    return None, -1
  run_dir = runs[-1]
  best, best_it = None, -1
  for p in glob.glob(os.path.join(run_dir, "model_*.pt")):
    m = re.search(r"model_(\d+)\.pt$", p)
    if m and int(m.group(1)) > best_it:
      best, best_it = p, int(m.group(1))
  return best, best_it


def current_train_iter(run_name: str, experiment: str) -> int:
  """从最新 run 目录的 tensorboard 事件文件读当前训练轮数（读不到返回 -1）。"""
  root = os.path.join(MJLAB_DIR, "logs", "rsl_rl", experiment)
  runs = sorted(glob.glob(os.path.join(root, f"*_{run_name}")), key=os.path.getmtime)
  if not runs:
    return -1
  try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea = EventAccumulator(runs[-1])
    ea.Reload()
    for tag in ("Train/mean_episode_length", "Train/mean_reward"):
      if tag in ea.Tags()["scalars"]:
        return ea.Scalars(tag)[-1].step
  except Exception:
    return -1
  return -1


class Watcher(V12HistSim2Sim):
  """在标准 sim2sim 循环里加 checkpoint 轮询 + 热替换。

  刷新模式二选一：``every_iters > 0`` 按训练轮数（每前进 N 轮刷新一次，
  轮数从 tensorboard 事件文件读）；否则按墙钟 ``interval`` 秒扫。
  """

  def __init__(self, args, run_name: str, interval: float,
               every_iters: int = 0, scan_wall: float = 15.0):
    super().__init__(args)  # args.onnx=None，不会建 onnxruntime 会话
    self.sess = TorchPolicy()
    self._run_name = run_name
    self._experiment = getattr(args, "experiment", "wheelleg_v12_mjlab")
    self._interval = interval
    self._every = int(every_iters)
    self._scan_wall = float(scan_wall)
    self._loaded_iter = -1
    self._last_scan = 0.0
    self._last_poll = 0.0
    self._refresh_base = None
    self._scan(False)
    if self._loaded_iter < 0:
      print("[watch] 还没有可用 checkpoint，先用零动作看机器人自由落体", flush=True)

  def _scan(self, swap_and_reset: bool = True) -> None:
    path, it = find_latest(self._run_name, self._experiment)
    if path is None or it <= self._loaded_iter:
      return
    try:
      got = self.sess.load(path)
      if got < 0:
        got = it
    except Exception as exc:  # 文件可能正写到一半
      print(f"[watch] 跳过 {os.path.basename(path)}: {exc}", flush=True)
      return
    self._loaded_iter = got
    print(f"[watch] 热替换策略 -> iter {got}（{os.path.basename(os.path.dirname(path))}）", flush=True)
    if swap_and_reset:
      self.reset()

  def run(self):
    args = self.args
    import mujoco
    import mujoco.viewer
    viewer = mujoco.viewer.launch_passive(self.m, self.d)
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    viewer.cam.distance = 2.2
    viewer.cam.azimuth = 140.0
    viewer.cam.elevation = -18.0
    self.cmd.start_listener()
    print("[KEY] 1/2/3/4 前后左右，5/6 升降，0 停，b 失能，f 姿态保持，q 退出", flush=True)
    if self._every > 0:
      print(f"[watch] 监视 run={self._run_name}，训练每前进 {self._every} 轮刷新一次"
            f"（轮询间隔 {self._scan_wall}s）", flush=True)
    else:
      print(f"[watch] 监视 run={self._run_name}，scan 间隔 {self._interval}s", flush=True)
    t0 = time.time()
    next_t = time.perf_counter()
    last_sync = 0.0
    fall_time = 0.0
    n_resets = 0
    last_dbg = -1.0
    while viewer.is_running() and self.cmd.running:
      if self._every > 0:
        if time.time() - self._last_poll >= self._scan_wall:
          self._last_poll = time.time()
          it = current_train_iter(self._run_name, self._experiment)
          if it >= 0:
            if self._refresh_base is None:
              self._refresh_base = it
              print(f"[watch] 训练当前轮数 {it}", flush=True)
            elif it - self._refresh_base >= self._every:
              self._refresh_base = it
              self._scan()
      elif time.time() - self._last_scan >= self._interval:
        self._last_scan = time.time()
        self._scan()
      self.cmd.step_ramp(POLICY_DT)
      obs, q, qd, jac = self.read()
      flat = self.push_obs(obs)
      if self._loaded_iter < 0:
        act = np.zeros(6, dtype=np.float32)
      else:
        act = np.clip(self.sess.run(None, {"obs": flat.reshape(1, -1).astype(np.float32)})[0]
                      .reshape(-1), -CLIP_ACTIONS, CLIP_ACTIONS).astype(np.float32)
      self.last_actions = act
      self.action_fifo.append(act.astype(np.float32))
      act_ctrl = self.action_fifo.pop(0) if len(self.action_fifo) > self.n_delay else np.zeros(6, dtype=np.float32)
      for _ in range(int(round(POLICY_DT / SIM_DT))):
        tau = self.compute_torques(act_ctrl, q, qd, jac)
        self.apply(tau)
        mujoco.mj_step(self.m, self.d)
      # 翻倒自动复位：让窗口持续展示"站立尝试"而不是永远躺着
      _, x, y, _ = self.d.qpos[3:7]
      tilt = 1.0 - 2.0 * (x * x + y * y)  # = cos(tilt)
      if tilt < 0.5:  # >60°
        fall_time += POLICY_DT
        if fall_time > 2.0:
          n_resets += 1
          print(f"[watch] 翻倒复位 #{n_resets}（policy iter={self._loaded_iter}）", flush=True)
          self.reset()
          fall_time = 0.0
      else:
        fall_time = 0.0
      if self.d.time - last_dbg >= 1.0:
        last_dbg = self.d.time
        print(f"[watch] t={self.d.time:5.1f}s z={float(self.d.qpos[2]):.3f} "
              f"tilt={tilt:.3f} cmd=({self.cmd.lin:+.2f},{self.cmd.yaw:+.2f},"
              f"{self.cmd.height:.3f}) mode={self.cmd.mode} fall_t={fall_time:.2f}",
              flush=True)
      now = time.perf_counter()
      if now - last_sync >= 1.0 / 60.0:
        viewer.cam.lookat[:] = self.d.qpos[:3]
        viewer.sync()
        last_sync = now
      next_t += POLICY_DT
      wait = next_t - now
      if wait > 0:
        time.sleep(wait)
      else:
        next_t = time.perf_counter()
    print(f"[watch] 退出（跑了 {time.time() - t0:.0f}s，最后策略 iter={self._loaded_iter}）", flush=True)
    return 0


def main():
    parser = build_parser()
    parser.set_defaults(xml=V12_SCENE, onnx=None, duration=0)
    parser.add_argument("--run", default="flat_bc3", help="要跟随的训练 run 名")
    parser.add_argument("--experiment", default="wheelleg_v12_mjlab")
    parser.add_argument("--interval", type=float, default=30.0,
                        help="墙钟扫描间隔（秒），--every-iters 0 时生效")
    parser.add_argument("--every-iters", type=int, default=20,
                        help="训练每前进多少轮刷新一次；0 = 改用墙钟 --interval")
    parser.add_argument("--scan-wall", type=float, default=15.0,
                        help="轮数轮询间隔（秒）")
    args = parser.parse_args()
    return Watcher(args, args.run, args.interval,
                   every_iters=args.every_iters, scan_wall=args.scan_wall).run()


if __name__ == "__main__":
    raise SystemExit(main())
