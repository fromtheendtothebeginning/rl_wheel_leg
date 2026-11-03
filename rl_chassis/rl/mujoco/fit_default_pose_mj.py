#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在 MuJoCo 闭链模型里**实测**出训练环境该用的默认站姿（默认角）。

为什么不用解析估计：PD 的静平衡满足 kp*(q_default - q_eq) = τ_gravity(q_eq)，解析算
τ_gravity 时容易漏掉腿自重/闭链约束力矩（实测差了约 1.5 倍），静平衡点就会偏，
后果是腿撑过头——轮子跑到髋后方，整车往后倒在尾巴上（底盘跟地面接触）。

固定点迭代：
    q_default <- q_default + (q_des - q_settled)
其中 q_settled 是零动作 PD 静置后的实测姿态，q_des 是想要的目标站姿。

用法:
  python fit_default_pose_mj.py --L0 0.205 --theta0 0 --iters 4
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import onnx_mj_v11 as sim2sim  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", default=sim2sim.DEFAULT_XML)
    ap.add_argument("--L0", type=float, default=0.205)
    ap.add_argument("--theta0", type=float, default=0.0, help="deg")
    ap.add_argument("--iters", type=int, default=4)
    ap.add_argument("--settle", type=float, default=3.0, help="每次静置仿真时长 [s]")
    args = ap.parse_args()

    base_args = sim2sim.build_parser().parse_args(["--zero-action", "--no-viewer",
                                                   "--duration", str(args.settle), "--xml", args.xml])
    _, leg_l, leg_r = sim2sim.load_legs(args.xml)
    legs = {"L": leg_l, "R": leg_r}
    q_des = {s: np.array(legs[s].ik_virtual(args.L0, math.radians(args.theta0))) for s in ("L", "R")}
    q_def = {s: q_des[s].copy() for s in ("L", "R")}
    print(f"目标 L0={args.L0:.4f} m, theta0={args.theta0:+.1f} deg -> "
          f"q_des L={np.round(q_des['L'],4)} R={np.round(q_des['R'],4)}")
    for it in range(args.iters):
        sim2sim.DEFAULT_DOF = np.array([*q_def["L"][:1], q_def["L"][1], 0.0,
                                        *q_def["R"][:1], q_def["R"][1], 0.0])
        sim = sim2sim.V11Sim2Sim(base_args)
        for _ in range(int(args.settle / sim2sim.POLICY_DT)):
            obs, q, qd, jac = sim.read()
            sim.apply(sim.compute_torques(np.zeros(6), q, qd, jac))
            for _ in range(int(round(sim2sim.POLICY_DT / sim2sim.SIM_DT))):
                mujoco.mj_step(sim.m, sim.d)
        d = sim.d
        w, x, y, z = d.qpos[3:7]
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, 1 - 2 * (x * x + y * y)))))
        settled = {}
        for i, s in enumerate(("L", "R")):
            adr = sim.dof_adr if s == "L" else sim.dof_adr_R
            settled[s] = np.array([float(d.qpos[adr["rear"][0]]), float(d.qpos[adr["shank"][0]])])
        L0L, th0L = legs["L"].virtual_leg(*settled["L"])
        L0R, th0R = legs["R"].virtual_leg(*settled["R"])
        print(f"[iter {it}] 静置: h={float(d.qpos[2]):.4f} m  倾斜={tilt:5.1f}°  "
              f"L0=({L0L:.4f},{L0R:.4f})  theta0=({math.degrees(th0L):+.1f},{math.degrees(th0R):+.1f})°  "
              f"接地={sim.floor_contacts()}  "
              f"settled L={np.round(settled['L'],4)} R={np.round(settled['R'],4)}")
        for s in ("L", "R"):
            q_def[s] = q_def[s] + (q_des[s] - settled[s])
    print("\n# 建议的 default_joint_angles（左腿 / 右腿，髋与虚拟膝）：")
    for s in ("L", "R"):
        pre = "Left" if s == "L" else "Right"
        print(f'            "{pre}_rear_joint": {q_def[s][0]:.6f},')
        print(f'            "{pre}_rear_child1_joint": {q_def[s][1]:.6f},')
        print(f'            "{pre}_Wheel_joint": 0.0,')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
