#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""跳跃训练观测窗：MuJoCo viewer 实时跟随最新残差 checkpoint。

组合与 onnx_mj_v12_jump_mjlab.py 同口径：冻结平地策略（v12_bc2.onnx）
+ 训练中的 42 维残差 actor（直接热载 .pt，省一步 ONNX 导出）
+ 相位状态机 + 气弹簧。纯 CPU（单环境 + CPU 推理），不碰训练 GPU。

刷新规则：每 --scan-wall 秒读一次 run 目录的 tensorboard 事件文件取当前
训练轮数，每前进 --every-iters 轮刷新一次——有新 checkpoint 就热替换权重，
然后重置场景从出生姿态重新演示（checkpoint 每 50 轮存盘，两次存盘之间
的刷新 = 同权重重新演示）。翻倒 >60° 持续 2 s 自动复位。

用法（mjlab/ 目录，.venv 的 python 跑）：
    .venv/bin/python ../mujoco/watch_jump_training.py --run jump_v2
    .venv/bin/python ../mujoco/watch_jump_training.py --run jump_v2 --no-auto-jump
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MJLAB_DIR = os.path.abspath(os.path.join(HERE, "..", "mjlab"))
sys.path.insert(0, HERE)
sys.path.insert(0, MJLAB_DIR)

from onnx_mj_v11 import CLIP_ACTIONS, POLICY_DT, SIM_DT, build_parser, quat_rotate_inverse  # noqa: E402
from onnx_mj_v12_jump_mjlab import DIMS, FLAT_ONNX, MjlabJumpSim, V12_GAS_SCENE  # noqa: E402
from jump_manager import JumpPhase  # noqa: E402
from watch_training import TorchPolicy, find_latest  # noqa: E402

PHASE_NAMES = {0: "FLAT", 1: "CROUCH", 2: "THRUST", 3: "FLIGHT",
               4: "EXTEND", 5: "LAND", 6: "RECOVER"}


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


class JumpWatcher(MjlabJumpSim):
  """在跳跃 sim2sim 循环里加训练轮数轮询 + 残差 .pt 热替换 + 翻倒复位。"""

  def __init__(self, args, run_name: str, experiment: str,
               every_iters: int, scan_wall: float):
    args.jump_onnx = None  # 残差不用固定 ONNX，走 .pt 热替换
    super().__init__(args)
    self._pol = TorchPolicy(num_obs=42)  # 42->256^3->6，与训练 residual actor 同构
    self.jsess = self._pol  # onnxruntime 兼容壳：.run(None, {"obs": x})
    self.jinput = "obs"
    self._run_name = run_name
    self._experiment = experiment
    self._every = int(every_iters)
    self._scan_wall = float(scan_wall)
    self._auto_jump = bool(getattr(args, "auto_jump", True))
    self._loaded_ckpt = -1
    self._last_poll = 0.0
    self._refresh_base = None
    self._scan(initial=True)
    if self._loaded_ckpt < 0:
      print("[watch] 还没有可用残差 checkpoint，先看纯平地策略+状态机", flush=True)

  def _apply_warmup_policy(self):
    """--no-auto-jump 时压制相位机的自动起跳（reset 会把 warmup 重置回 0.5~1.5）。"""
    if not self._auto_jump:
      self.manager.warmup.fill_(1e9)

  def _scan(self, initial: bool = False) -> None:
    path, it = find_latest(self._run_name, self._experiment)
    if path is None:
      return
    if it <= self._loaded_ckpt:
      if not initial:
        print(f"[watch] 刷新（权重不变，仍是 iter {self._loaded_ckpt}），重置场景", flush=True)
        self.reset()
        self._apply_warmup_policy()
      return
    try:
      got = self._pol.load(path)
      if got < 0:
        got = it
    except Exception as exc:  # checkpoint 可能正写到一半
      print(f"[watch] 跳过 {os.path.basename(path)}: {exc}", flush=True)
      return
    self._loaded_ckpt = got
    print(f"[watch] 热替换残差策略 -> iter {got}（{os.path.basename(os.path.dirname(path))}）",
          flush=True)
    self.reset()
    self._apply_warmup_policy()

  def run(self):
    import mujoco
    import mujoco.viewer

    viewer = mujoco.viewer.launch_passive(self.m, self.d)
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    viewer.cam.distance = 2.2
    viewer.cam.azimuth = 140.0
    viewer.cam.elevation = -18.0
    self.cmd.start_listener()
    mode = "自动连续起跳" if self._auto_jump else "空格起跳"
    print(f"[KEY] {mode}；1/2/3/4 前后左右，5/6 升降，0 停，b 失能，q 退出；"
          f"训练每前进 {self._every} 轮刷新一次", flush=True)
    print(f"[watch] 监视 run={self._run_name}，轮询间隔 {self._scan_wall}s", flush=True)

    t0 = time.time()
    next_t = time.perf_counter()
    last_sync = 0.0
    fall_time = 0.0
    n_resets = 0
    step = 0
    while viewer.is_running() and self.cmd.running:
      # ---- 训练轮数轮询：每前进 _every 轮刷新一次 ----
      now = time.time()
      if now - self._last_poll >= self._scan_wall:
        self._last_poll = now
        it = current_train_iter(self._run_name, self._experiment)
        if it >= 0:
          if self._refresh_base is None:
            self._refresh_base = it
            print(f"[watch] 训练当前轮数 {it}", flush=True)
          elif it - self._refresh_base >= self._every:
            self._refresh_base = it
            self._scan()

      self.cmd.step_ramp(POLICY_DT)
      obs25, q, qd, jac = self.read()
      # 平地策略：term-major 125 维历史
      i = 0
      for buf, d in zip(self.term_bufs, DIMS):
        buf[:-1] = buf[1:]
        buf[-1] = obs25[i:i + d]
        i += d
      flat_in = np.concatenate([b.reshape(-1) for b in self.term_bufs]).reshape(1, -1)
      a_flat = np.clip(self.sess.run(None, {self.in_names[0]: flat_in.astype(np.float32)})[0]
                       .reshape(-1), -CLIP_ACTIONS, CLIP_ACTIONS)

      # 相位状态机推进（可部署信号）
      contact = self._contact()
      root_z = float(self.d.qpos[2])
      base_vz = float(self.d.qvel[2])
      grav_z = float(quat_rotate_inverse(self.d.qpos[3:7], np.array([0, 0, -1.0]))[2])
      ang_xy = float(np.linalg.norm(self.d.qvel[3:5]))
      self.manager.advance(
          POLICY_DT, self.torch.tensor([contact]),
          self.torch.tensor([root_z]), self.torch.tensor([base_vz]),
          self.torch.tensor([grav_z]), self.torch.tensor([ang_xy]))
      if not self._auto_jump:
        # RECOVER->FLAT 时 _to(reset_warmup=True) 会塞回 0.5~1.5s 自动起跳预热，
        # 这里逐拍压掉，只留空格触发
        self.manager.warmup[0] = 1e9
        if getattr(self.cmd, "jump_request", False):
          self.manager.warmup[0] = 0.0
          self.cmd.jump_request = False
      elif getattr(self.cmd, "jump_request", False):
        self.manager.warmup[0] = 0.0
        self.cmd.jump_request = False

      # 接触置信度 / 力矩反馈（可部署近似）
      qd_arr = np.array(qd)
      tau_w = np.array(self.tau_applied[[2, 5]]) if hasattr(self, "tau_applied") else np.zeros(2)
      acc_w = np.abs(qd_arr[[2, 5]]) / POLICY_DT
      conf = np.clip(0.5 * np.clip(acc_w / 50.0, 0, 1) + 0.3 * np.clip(ang_xy * 0.4, 0, 1)
                     + 0.2 * np.clip(np.abs(tau_w) / 20.0, 0, 1), 0, 1)
      # 残差推理（.pt 热替换壳）
      o42 = self._obs42(obs25, conf, tau_w / 20.0)
      res = np.clip(self.jsess.run(None, {self.jinput: o42.astype(np.float32)})[0]
                    .reshape(-1), -CLIP_ACTIONS, CLIP_ACTIONS)
      if self.residual_delay:
        self.residual_delay.append(res)
        res_d = self.residual_delay.pop(0)
      else:
        res_d = res
      scale = float(self.manager.residual_scale()[0])
      self.last_residual = res
      act = np.clip(a_flat + scale * res_d, -CLIP_ACTIONS, CLIP_ACTIONS).astype(np.float32)
      self.last_total = act

      # 气弹簧 ctrl（FLAT 20N / 跳跃窗口 350N）
      gas = float(self.args.gas_force_flat) if int(self.manager.phase[0]) == int(JumpPhase.FLAT) \
          else float(self.args.gas_force_jump)
      for gid in self.gas_ids:
        self.d.ctrl[gid] = gas

      self.last_actions = act
      self.action_fifo.append(act.astype(np.float32))
      act_ctrl = self.action_fifo.pop(0) if len(self.action_fifo) > self.n_delay else np.zeros(6, np.float32)
      for _ in range(int(round(POLICY_DT / SIM_DT))):
        tau = self.compute_torques(act_ctrl, q, qd, jac)
        self.apply(tau)
        mujoco.mj_step(self.m, self.d)

      # 翻倒自动复位：让窗口持续展示而不是永远躺着
      _, qx, qy, _ = self.d.qpos[3:7]
      tilt = 1.0 - 2.0 * (qx * qx + qy * qy)  # = cos(tilt)
      if tilt < 0.5:  # >60°
        fall_time += POLICY_DT
        if fall_time > 2.0:
          n_resets += 1
          print(f"[watch] 翻倒复位 #{n_resets}（残差 iter={self._loaded_ckpt}）", flush=True)
          self.reset()
          self._apply_warmup_policy()
          fall_time = 0.0
      else:
        fall_time = 0.0

      # 记录/打印（每秒一行）
      if self.start_z is None and self.manager.phase[0] != int(JumpPhase.FLAT):
        self.start_z = root_z
      if not contact:
        self.peak_z = max(self.peak_z, root_z - float(self.start_z or root_z))
      now_sync = time.perf_counter()
      if now_sync - last_sync >= 1.0 / 60.0:
        viewer.cam.lookat[:] = self.d.qpos[:3]
        viewer.sync()
        last_sync = now_sync
      if step % max(1, int(1.0 / POLICY_DT)) == 0:
        print(f"[{self.d.time:6.1f}s] {PHASE_NAMES[int(self.manager.phase[0])]:7s} "
              f"z={root_z:.3f} vz={base_vz:+.2f} gas={gas:3.0f}N "
              f"peak={self.peak_z:.3f} contact={int(contact)} "
              f"res={np.abs(res).max():.1f}", flush=True)
      step += 1
      next_t += POLICY_DT
      wait = next_t - time.perf_counter()
      if wait > 0:
        time.sleep(wait)
      else:
        next_t = time.perf_counter()
    print(f"[watch] 退出（跑了 {time.time() - t0:.0f}s，最后残差 iter={self._loaded_ckpt}）",
          flush=True)
    return 0


def main():
  parser = build_parser()
  parser.set_defaults(xml=V12_GAS_SCENE, onnx=None, duration=0)
  parser.add_argument("--jump-onnx", default=None)  # 占位：与 onnx_mj_v12_jump_mjlab 参数对齐
  parser.add_argument("--target-height", type=float, default=0.10)
  parser.add_argument("--residual-delay", type=int, default=1)
  parser.add_argument("--gas-force-flat", type=float, default=20.0)
  parser.add_argument("--gas-force-jump", type=float, default=350.0)
  parser.add_argument("--run", default="jump_v2", help="要跟随的训练 run 名")
  parser.add_argument("--experiment", default="wheelleg_jump_v12")
  parser.add_argument("--every-iters", type=int, default=20, help="训练每前进多少轮刷新一次")
  parser.add_argument("--scan-wall", type=float, default=15.0, help="轮数轮询间隔（秒）")
  parser.add_argument("--auto-jump", dest="auto_jump", action="store_true", default=True)
  parser.add_argument("--no-auto-jump", dest="auto_jump", action="store_false")
  args = parser.parse_args()
  if args.onnx is None:
    args.onnx = FLAT_ONNX
  return JumpWatcher(args, args.run, args.experiment, args.every_iters, args.scan_wall).run()


if __name__ == "__main__":
  raise SystemExit(main())
