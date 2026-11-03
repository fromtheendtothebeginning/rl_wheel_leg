#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""给训练环境算"重力静平衡默认站姿"。

为什么需要：PD 在零动作时输出 kp*(default - q)，腿在重力下会静差下沉。v11 的腿比较
折（L0=0.205 m 时膝上需要 ~7 N·m），kp=20 时静差 ≈ 0.35 rad，腿会一路折到轮舱里、
底盘凸包顶到轮子（冒烟测试里 base_height 掉到 0.06 m 就是这个原因）。

做法：让 default = 目标姿态 + τ_gravity/kp，于是零动作时恰好停在目标姿态。
力矩由虚功原理算：τ = J^T f，J = d(轮心位置)/d(q_rear, q_shank)，f = (0, -F)。

用法:
  python3 tune_default_pose.py --L0 0.205 --kp 20
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from leg_model import load  # noqa: E402

DEFAULT_XML = "../../mjcf/rl_chassis_v12_weld.xml"
MASS = 7.3956  # 训练模型总质量 [kg]（前链已删）
G = 9.81


def leg_jacobian(leg, q_rear, q_shank, eps=1e-6):
    def vec(qr, qs):
        L0, th0 = leg.virtual_leg(qr, qs)
        return np.array([L0 * np.sin(th0), -L0 * np.cos(th0)])

    J = np.column_stack([
        (vec(q_rear + eps, q_shank) - vec(q_rear - eps, q_shank)) / (2 * eps),
        (vec(q_rear, q_shank + eps) - vec(q_rear, q_shank - eps)) / (2 * eps),
    ])
    return J  # [dx/dq; dz/dq] 行=方向, 列=关节


def hold_torque(leg, q_rear, q_shank, F):
    J = leg_jacobian(leg, q_rear, q_shank)
    return J.T @ np.array([0.0, -F])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", default=DEFAULT_XML)
    ap.add_argument("--L0", type=float, default=0.205)
    ap.add_argument("--theta0", type=float, default=0.0, help="deg")
    ap.add_argument("--kp", type=float, default=20.0)
    args = ap.parse_args()

    _, L, R = load(args.xml)
    F = MASS * G / 2.0  # 单腿承担的重力 [N]
    print(f"目标站姿 L0={args.L0:.4f} m, theta0={args.theta0:+.1f} deg, kp={args.kp:g}, "
          f"单腿重力 {F:.2f} N\n")
    out = {}
    for leg in (L, R):
        q_des = np.array(leg.ik_virtual(args.L0, np.radians(args.theta0)))
        tau = hold_torque(leg, *q_des, F)
        q_def = q_des + tau / args.kp
        sag = np.array(leg.ik_virtual(*[float(v) for v in [args.L0, np.radians(args.theta0)]]))
        del sag
        print(f"{leg.side:5s}: 目标 q={np.round(q_des,4)} rad  需保持力矩 τ={np.round(tau,2)} N·m")
        print(f"       默认角 q_default=q_des+τ/kp = {np.round(q_def,6)} rad "
              f"({np.round(np.degrees(q_def),2)} deg)")
        print(f"       不补偿时的静差 ≈ {np.round(tau/args.kp,4)} rad "
              f"({np.degrees(tau/args.kp).round(2)} deg)")
        out[leg.side] = q_def
    print("\n# 填进 wheel_legged_config.py 的 default_joint_angles：")
    for leg in (L, R):
        s = leg.side.capitalize()
        print(f'            "{s}_rear_joint": {out[leg.side][0]:.6f},')
        print(f'            "{s}_rear_child1_joint": {out[leg.side][1]:.6f},')
        print(f'            "{s}_Wheel_joint": 0.0,')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
