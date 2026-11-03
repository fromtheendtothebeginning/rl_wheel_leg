#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""站姿工具：(L0, theta0) <-> 关节角，顺带给出真机曲柄角与静态力矩预算。

用途：
  * 挑训练环境的 default_joint_angles / base_height_target；
  * 给 MuJoCo sim2sim 生成初始 keyframe（需要真机的曲柄角）；
  * 估算各关节在该姿态下需要多少力矩（重力静态分量），用来判断力矩上限是否够。

用法:
  python3 pose_tool.py --L0 0.205 --theta0 0          # 一个姿态
  python3 pose_tool.py --scan                          # 常用姿态表
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from leg_model import load  # noqa: E402

DEFAULT_XML = "../../mjcf/rl_chassis_v12_weld.xml"
# base_link 原点相对髋关节的高度：base_z = L0（腿竖直时）+ R_wheel - HIP_OFFSET
HIP_OFFSET = 0.11508
WHEEL_R = 0.06
HIP_X = -0.001951
MASS = 7.3956  # 训练模型总质量
G = 9.81


def report(leg, L0, theta0, verbose=True):
    q_rear, q_shank = leg.ik_virtual(L0, theta0)
    q, res, seed = leg.solve_virtual(q_rear, q_shank)
    L0_chk, th0_chk = leg.virtual_leg(q_rear, q_shank)
    if verbose:
        print(f"  {leg.side:5s}: q_rear={q_rear:+.4f} q_shank={q_shank:+.4f} rad "
              f"({np.degrees(q_rear):+7.2f}, {np.degrees(q_shank):+7.2f} deg)  "
              f"真机曲柄 q_front={q['front']:+.4f} rad ({np.degrees(q['front']):+7.2f} deg)  "
              f"销残差 {res * 1e6:.3f} um  校验 L0={L0_chk:.4f} theta0={np.degrees(th0_chk):+.2f}deg")
        print(f"         真机全关节: child1={q['child1']:+.4f} child2={q['child2']:+.4f} "
              f"child3={q['child3']:+.4f}")
    return dict(q_rear=q_rear, q_shank=q_shank, q_front=q["front"], q=q, res=res)


def static_torque(leg, q_rear, q_shank):
    """只考虑整车重力摊到这条腿上的竖直力，估算 (后髋, 前髋) 静态力矩。"""
    F = MASS * G / 2.0  # 单腿承担的竖直力
    tau_virtual_rear, tau_virtual_shank = 0.0, 0.0
    # 数值求 (L0, theta0) 对 (q_rear, q_shank) 的雅可比，再换算虚拟关节力矩
    eps = 1e-6
    def vec(qr, qs):
        L, th = leg.virtual_leg(qr, qs)
        return np.array([L * np.sin(th), -L * np.cos(th)])
    J = np.column_stack([
        (vec(q_rear + eps, q_shank) - vec(q_rear - eps, q_shank)) / (2 * eps),
        (vec(q_rear, q_shank + eps) - vec(q_rear, q_shank - eps)) / (2 * eps),
    ])  # [dx/dq, dz/dq]
    f = np.array([0.0, -F])
    tau = J.T @ f  # 两个虚拟关节所需力矩 (N·m)
    return tau


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", default=DEFAULT_XML)
    ap.add_argument("--L0", type=float, default=0.205)
    ap.add_argument("--theta0", type=float, default=0.0, help="deg，轮在前为正（我们口径）")
    ap.add_argument("--scan", action="store_true")
    args = ap.parse_args()

    _, L, R = load(args.xml)
    if args.scan:
        print("常用站姿表（theta0 = 0 即腿竖直，L0 越大站得越高）")
        for L0 in (0.15, 0.18, 0.205, 0.23, 0.26):
            print(f"\nL0={L0:.3f} m  -> base_link 高度 {L0 + WHEEL_R - HIP_OFFSET:+.4f} m, "
                  f"髋高 {L0 + WHEEL_R:.4f} m")
            for leg in (L, R):
                report(leg, L0, 0.0)
        print("\n倾角扫描（L0=0.205 固定）")
        for th in (-20, -10, 0, 10, 20):
            print(f"\ntheta0={th:+d} deg")
            for leg in (L, R):
                report(leg, 0.205, np.radians(th), verbose=False)
        return 0

    print(f"目标 L0={args.L0:.4f} m, theta0={args.theta0:+.1f} deg -> "
          f"base_link 高度 {args.L0 + WHEEL_R - HIP_OFFSET:+.4f} m（腿竖直时）")
    for leg in (L, R):
        r = report(leg, args.L0, np.radians(args.theta0))
        tau = static_torque(leg, r["q_rear"], r["q_shank"])
        print(f"         单腿静态重力力矩（虚拟 髋/膝）: {tau[0]:+.2f} / {tau[1]:+.2f} N·m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
