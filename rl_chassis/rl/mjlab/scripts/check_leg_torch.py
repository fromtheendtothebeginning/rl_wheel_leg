#!/usr/bin/env python3
"""leg_torch 数值校验：

1. 常数（l1/l2/a1/a2/sgn）与仓库参考实现 rl/tools/leg_kinematics_ref.py 一致；
2. 默认姿态的 residual ≈ 0；
3. shank_jacobian 与（a）参考实现的数值牛顿雅可比、（b）本实现残差的
   有限差分一致；
4. theta0 与参考实现一致。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

REPO = Path("/home/end/RoboMaster/KaiWuRoboMaster/rl_chassis/rl")
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "mjlab" / "src"))

from leg_kinematics_ref import LegRef  # noqa: E402

from mjlab_wheelleg.leg_torch import (  # noqa: E402
    CLOSED_DEFAULT,
    LegTorch,
    VIRTUAL_DEFAULT,
)

torch.set_default_dtype(torch.float64)

ok = True
for side in ("left", "right"):
    lt = LegTorch(side)
    ref = LegRef(side)
    dc = max(
        abs(lt.l1 - ref.l1),
        abs(lt.l2 - ref.l2),
        abs(lt.a1 - ref.a1),
        abs(lt.a2 - ref.a2),
        abs(lt.sgn_rear - ref.sgn_rear),
        abs(lt.sgn_shank - ref.sgn_shank),
    )
    print(f"[{side}] 串联常数最大偏差 {dc:.2e}")
    ok &= dc < 1e-9

    q0 = torch.tensor(CLOSED_DEFAULT[side]).reshape(1, 6)
    r = lt.residual(q0).abs().max().item()
    print(f"[{side}] 默认姿态 residual = {r:.2e} m")
    ok &= r < 1e-9  # 常数只保留了 9 位小数，1e-10 量级属正常

# —— 有限差分 + 牛顿投影（不依赖 functorch）——
def fd_jac(lt, q: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    J = torch.zeros(4, 6)
    for j in range(6):
        qp, qm = q.clone(), q.clone()
        qp[0, j] += eps
        qm[0, j] -= eps
        J[:, j] = (lt.residual(qp) - lt.residual(qm)).squeeze(0) / (2 * eps)
    return J


def project(lt, q: torch.Tensor, iters: int = 60) -> torch.Tensor:
    """牛顿投影到闭链流形（固定 front/rear，解 4 个被动关节）。"""
    fixed = q[:, [0, 4]].clone()
    q = q.clone()
    idx = [1, 2, 3, 5]  # c1, c2, c3, shank
    for _ in range(iters):
        r = lt.residual(q).squeeze(0)
        if r.norm() < 1e-13:
            break
        Jfd = fd_jac(lt, q)
        Jx = Jfd[:, idx]
        dx = torch.linalg.solve(Jx, -r)
        s = dx.abs().max().item()
        if s > 0.35:
            dx *= 0.35 / s
        q[0, idx] += dx
        q[0, [0, 4]] = fixed[0]
    return q


for side in ("left", "right"):
    lt = LegTorch(side)
    ref = LegRef(side)
    torch.manual_seed(1)
    worst_ref = worst_th = worst_res = 0.0
    seed = None
    for _ in range(100):
        # 以参考实现的解为基准状态（同一装配分支），再对比雅可比
        qf_, qr_ = torch.clamp(torch.randn(2) * 0.35, -0.6, 0.6)
        qq, res, seed = ref.solve(qf_.item(), qr_.item(), seed=seed)
        if res > 1e-9:
            continue
        q = torch.tensor(
            [[qq[k] for k in ("front", "child1", "child2", "child3", "rear", "shank")]],
            dtype=torch.get_default_dtype(),
        )
        worst_res = max(worst_res, lt.residual(q).norm().item())

        d_front, d_rear = lt.shank_jacobian(q)
        dr, df = ref.jacobian(qf_.item(), qr_.item(), seed=seed)
        worst_ref = max(
            worst_ref, abs(dr - d_rear.item()), abs(df - d_front.item())
        )

        th = lt.theta0(q[:, 4], q[:, 5]).item()
        th_ref = ref.virtual_leg(qr_.item(), q[0, 5].item())[1]
        worst_th = max(worst_th, abs(th - th_ref))

    # 另一路：随机状态投影到流形后，重投影有限差分校验（不经过参考实现，
    # 避免装配分支歧义）
    worst_fd = 0.0
    for _ in range(30):
        q = project(lt, torch.clamp(torch.randn(1, 6) * 0.35, -0.6, 0.6))
        d_front, d_rear = lt.shank_jacobian(q)
        eps = 1e-6
        for name, var, ana in (("front", 0, d_front), ("rear", 4, d_rear)):
            qp, qm = q.clone(), q.clone()
            qp[0, var] += eps
            qm[0, var] -= eps
            num = (project(lt, qp)[0, 5] - project(lt, qm)[0, 5]) / (2 * eps)
            worst_fd = max(worst_fd, abs(num.item() - ana.item()))

    print(
        f"[{side}] 流形残差 {worst_res:.1e} | 解析vs数值(重投影FD) {worst_fd:.1e} | "
        f"vs参考实现 {worst_ref:.1e} | theta0 {worst_th:.1e}"
    )
    ok &= worst_fd < 5e-4 and worst_ref < 5e-3 and worst_th < 1e-9

print("\nPASS" if ok else "\nFAIL")
raise SystemExit(0 if ok else 1)
