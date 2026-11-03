#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""标定 + 校验 v11 单腿闭链：

1) 后链等效串联参数 (l1/l2/a1/a2/符号)，供训练环境复刻复旦的 L0/theta0 公式；
2) 闭环逆解 vs 体检报告里真机动力学生成态（左腿有 ground truth）；
3) 虚拟腿公式 vs 真机闭环正解（应当逐点吻合）；
4) 工作站姿可达域扫描：从装配位姿连续外推，看 (L0, theta0) 能覆盖到哪里、雅可比量级。

用法: python3 calibrate_leg.py [--xml ...] [--plot out.png]
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from leg_model import V11Leg, load, wrap_to_pi  # noqa: E402

DEFAULT_XML = "../../mjcf/rl_chassis_v12_weld.xml"

# 体检报告 verify_report.txt 里，焊死基座自由衰减 2 s 后的真机状态（左腿 ground truth）
GROUND_TRUTH = dict(front=-0.4976, rear=-0.7726, child1=-0.1521, child2=+0.0292,
                    child3=-0.1521, shank=+0.1521)


def check_constants(leg: V11Leg) -> None:
    print(f"  l1 = {leg.l1:.6f} m   l2 = {leg.l2:.6f} m   R_wheel = {leg.wheel_radius:.5f} m "
          f"(半宽 {leg.wheel_half_width:.5f})")
    print(f"  a1 = {np.degrees(leg.a1):+8.3f} deg   a2 = {np.degrees(leg.a2):+8.3f} deg"
          f"   sgn_rear = {leg.sgn_rear:+.0f}   sgn_shank = {leg.sgn_shank:+.0f}")
    L0, th0 = leg.virtual_leg(0.0, 0.0)
    print(f"  装配位姿 q=0: L0 = {L0:.4f} m, theta0 = {np.degrees(th0):+7.2f} deg"
          f"（轮心到髋: 水平 {(leg._fk(rear=0.0, shank=0.0)[2] - leg.hip_base)[0]:+.4f} m, "
          f"垂直 {(leg._fk(rear=0.0, shank=0.0)[2] - leg.hip_base)[2]:+.4f} m）")


def check_ground_truth(leg: V11Leg) -> None:
    # verify_report 给的是左腿原始值；右腿是镜像（同值反号）
    sgn = 1.0 if leg.side == "left" else -1.0
    qf, qr = sgn * GROUND_TRUTH["front"], sgn * GROUND_TRUTH["rear"]
    q, res, seed = leg.solve_loop(qf, qr)
    print(f"  输入 (front={qf:+.4f}, rear={qr:+.4f})  销残差 {res * 1e6:.4f} um")
    ok = True
    for k in ("child1", "child2", "child3", "shank"):
        exp = sgn * GROUND_TRUTH[k]
        err = q[k] - exp
        ok &= abs(err) < 1e-4
        print(f"    {k:7s} 解 {q[k]:+.4f}  真值 {exp:+.4f}  差 {err:+.2e} rad")
    print(f"  => 逆解{'与真机一致' if ok else '不一致'}")
    return ok


def check_formula(leg: V11Leg, rng: np.random.Generator) -> float:
    """虚拟腿公式 vs 真机闭环正解（轮心相对髋的真实位置）。"""
    worst = 0.0
    for _ in range(200):
        qf, qr = rng.uniform(-0.6, 0.6, 2)
        q, res, _ = leg.solve_loop(qf, qr)
        if res > 1e-9:
            continue
        _, _, wheel = leg._fk(rear=qr, shank=q["shank"])
        ex = float(wheel[0] - leg.hip_base[0])
        ey = float(wheel[2] - leg.hip_base[2])
        true_L0 = float(np.hypot(ex, ey))
        true_th0 = float(np.arctan2(ex, -ey))
        L0, th0 = leg.virtual_leg(qr, q["shank"])
        worst = max(worst, abs(L0 - true_L0), abs(wrap_to_pi(th0 - true_th0)) * true_L0)
    return worst


def sweep(leg: V11Leg, span: float, n: int):
    """从装配位姿出发，按细网格栅格连续求解（种子继承）。

    只把「轮心在髋下方」的解算作物理分支（站姿可用），其余记为离支。
    """
    grid = np.linspace(-span, span, n)
    seed = None
    L0 = np.full((n, n), np.nan)
    th0 = np.full((n, n), np.nan)
    jac = np.full((n, n), np.nan)
    bad, jumps, off_branch = [], [], 0
    for i, qf in enumerate(grid):
        for j, qr in enumerate(grid):
            prev = None if seed is None else seed.copy()
            q, res, seed = leg.solve_loop(qf, qr, seed=seed)
            if res > 1e-9:
                bad.append((qf, qr, res))
                continue
            if prev is not None and np.max(np.abs(seed - prev)) > 0.15:
                jumps.append((qf, qr, float(np.max(np.abs(seed - prev)))))
            _, _, wheel = leg._fk(rear=qr, shank=q["shank"])
            if wheel[2] > leg.hip_base[2]:  # 轮心高于髋 -> 非站姿分支
                off_branch += 1
                continue
            L0[j, i], th0[j, i] = leg.virtual_leg(qr, q["shank"])
            jac[j, i] = abs(leg.jacobian(qf, qr, seed=seed)[0])
    return grid, L0, th0, jac, bad, jumps, off_branch


def check_branches(leg: V11Leg, span: float) -> None:
    """枚举若干输入下的所有装配分支，看看多解有多普遍。"""
    print("  多解分支枚举（残差<1e-9 的解个数 / shank 角）:")
    seed = None
    for qf, qr in ((0.0, 0.0), (0.3, 0.3), (-0.3, -0.3), (0.5, -0.5), (span, span)):
        sols = leg.solutions(qf, qr, n_start=10, seed=seed)
        seed = sols[0][1] if sols else None
        txt = ", ".join(f"shank={s[2]['shank']:+.4f}" for s in sols)
        print(f"    (front={qf:+.2f}, rear={qr:+.2f}): {len(sols)} 解  [{txt}]")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", default=DEFAULT_XML)
    ap.add_argument("--span", type=float, default=0.6, help="扫描的半幅 [rad]")
    ap.add_argument("--n", type=int, default=13)
    ap.add_argument("--plot", default=None)
    args = ap.parse_args()

    m, L, R = load(args.xml)
    print(f"== 模型 {args.xml}")
    print(f"   总质量 {sum(m.body_mass):.4f} kg, nq={m.nq}, nv={m.nv}, nu={m.nu}")
    rng = np.random.default_rng(0)
    ok_all = True
    for leg in (L, R):
        print(f"\n== {leg.side} 腿")
        check_constants(leg)
        ok_all &= check_ground_truth(leg)
        worst = check_formula(leg, rng)
        print(f"  虚拟腿公式 vs 闭环正解: 最大偏差 {worst * 1e6:.4f} um")
        grid, L0, th0, jac, bad, jumps, off_branch = sweep(leg, args.span, args.n)
        fin = np.isfinite(L0)
        print(f"  扫描 q_front/q_rear ∈ ±{args.span} rad（{args.n}×{args.n}）: "
              f"站姿分支可解 {fin.sum()}/{L0.size}，求解失败 {len(bad)}，"
              f"离支 {off_branch}，大幅跳变 {len(jumps)}")
        if fin.any():
            print(f"    L0 范围 {np.nanmin(L0):.4f} ~ {np.nanmax(L0):.4f} m")
            print(f"    theta0 范围 {np.degrees(np.nanmin(th0)):+.1f} ~ "
                  f"{np.degrees(np.nanmax(th0)):+.1f} deg")
            print(f"    |dq_shank/dq_rear| 最大 {np.nanmax(jac):.3f}")
        check_branches(leg, args.span)
        L0, th0 = np.where(fin, L0, 0.0), np.where(fin, th0, 0.0)
        if args.plot:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 2, figsize=(10, 4))
            X, Y = np.meshgrid(grid, grid)
            cs = ax[0].contourf(X, Y, L0, levels=20)
            ax[0].set_title(f"{leg.side}: L0 [m]"); plt.colorbar(cs, ax=ax[0])
            cs = ax[1].contourf(X, Y, np.degrees(th0), levels=20)
            ax[1].set_title(f"{leg.side}: theta0 [deg]"); plt.colorbar(cs, ax=ax[1])
            for a in ax:
                a.set_xlabel("q_front (crank)"); a.set_ylabel("q_rear (thigh)")
            fig.tight_layout()
            fig.savefig(args.plot, dpi=120)
            print(f"  图已保存 {args.plot}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
