#!/usr/bin/env python3
"""v12 闭链 MJCF 在 MuJoCo Warp (GPU) 上的保真度对拍。

同一开环力矩序列下，CPU MuJoCo 与 GPU mujoco-warp 各跑 N 步，
逐步比较 qpos / 关键 site 位置，报告最大偏差。
这决定训练资产能否直接用闭链模型（mjlab 仿真层就是 mujoco-warp）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np
import warp as wp

SCENE = Path(__file__).resolve().parents[3] / "mjcf" / "rl_chassis_v12_scene.xml"


def make_ctrl(model: mujoco.MjModel, nstep: int, dof: int) -> np.ndarray:
  """确定性开环控制：髋正弦力矩 + 轮恒力矩，激励闭链 + 接触。"""
  t = np.arange(nstep) * model.opt.timestep
  ctrl = np.zeros((nstep, model.nu))
  for name, amp, freq in (
    ("Left_front_joint_motor", 3.0, 1.3),
    ("Left_rear_joint_motor", -3.0, 1.7),
    ("Left_Wheel_joint_motor", 1.5, 0.0),
    ("Right_front_joint_motor", -3.0, 1.1),
    ("Right_rear_joint_motor", 3.0, 1.9),
    ("Right_Wheel_joint_motor", -1.5, 0.0),
  ):
    i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    ctrl[:, i] = amp * np.sin(2 * np.pi * freq * t)
  return ctrl


def run_cpu(model: mujoco.MjModel, data: mujoco.MjData, ctrl: np.ndarray) -> np.ndarray:
  mujoco.mj_resetDataKeyframe(model, data, 0)
  n = ctrl.shape[0]
  out = np.zeros((n, model.nq))
  for i in range(n):
    data.ctrl[:] = ctrl[i]
    mujoco.mj_step(model, data)
    out[i] = data.qpos
  return out


def run_gpu(model: mujoco.MjModel, ctrl: np.ndarray) -> np.ndarray:
  import mujoco_warp as mjwarp

  mjd = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, mjd, 0)
  wm = mjwarp.put_model(model)
  wd = mjwarp.put_data(model, mjd, nworld=1, nconmax=128, njmax=1024)
  n = ctrl.shape[0]
  out = np.zeros((n, model.nq))
  for i in range(n):
    wd.ctrl.assign(wp.from_numpy(np.ascontiguousarray(ctrl[i]), dtype=wp.float32))
    mjwarp.step(wm, wd)
    if i % 50 == 49 or i == n - 1:
      mjwarp.get_data_into(mjd, model, wd)
      out[i] = mjd.qpos
  # 上面只在采样点回读；为公平对比，CPU 输出也只在同样点取。
  return out


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--steps", type=int, default=1000)
  parser.add_argument("--xml", type=str, default=str(SCENE))
  args = parser.parse_args()

  model = mujoco.MjModel.from_xml_path(args.xml)
  print(
    f"nq={model.nq} nu={model.nu} neq={model.neq} ngeom={model.ngeom}"
    f" timestep={model.opt.timestep} solver={model.opt.solver}"
  )
  if model.neq != 4:
    print("!! 闭链 equality 数量不对", file=sys.stderr)

  ctrl = make_ctrl(model, args.steps, model.nu)
  cpu_full = run_cpu(model, mujoco.MjData(model), ctrl)

  gpu = run_gpu(model, ctrl)

  idx = [i for i in range(args.steps) if i % 50 == 49 or i == args.steps - 1]
  cpu = cpu_full[idx]
  dq = np.abs(gpu[idx] - cpu)
  jn = [model.joint(i).name for i in range(model.njnt)]

  print("\n偏差随时间（每 0.25 s 采样一次的 max|Δqpos|）:")
  for k, i in enumerate(idx):
    t = (i + 1) * model.opt.timestep
    print(f"  t={t:5.2f}s  max|Δq|={dq[k].max():.3e}  (argmax={dq[k].argmax()})")

  base = dq[:, :7]
  print(f"\nbase 悬浮位姿最大偏差: pos {base[:, :3].max():.3e} m, quat {base[:, 3:].max():.3e}")
  print("\n结论:", "GPU 可用" if dq.max() < 5e-3 else "偏差大，需检查")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
