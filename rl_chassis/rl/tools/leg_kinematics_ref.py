#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v11 单腿闭链运动学 —— **不依赖 MuJoCo 的纯 numpy 参考实现**（单片机移植照这个）。

真机每侧只有两个驱动关节：前髋(曲柄 front) 和后髋(大腿 rear)；膝(rear_child1, 下面叫
shank) 是被两条销轴约束"带"出来的。策略是在串联模型上训的，它嘴里的"膝角"就是这个
虚拟膝角，所以真机上必须先把 (q_front, q_rear) 解算成虚拟膝角，再把策略给的
虚拟 (髋, 膝) 力矩换回真机 (前髋, 后髋) 力矩。

这里把整条链路写清楚，并且提供与 MuJoCo 版（leg_model.py）一致的数值接口：

    leg = LegRef("left")
    q   = leg.solve(q_front, q_rear, seed)        # 闭环正/逆解，得到全部关节角
    d_rear, d_front = leg.jacobian(q_front, q_rear, seed)   # dq_shank/d(q_rear, q_front)
    tau_rear, tau_front = leg.torque_map(tau_hip_v, tau_knee_v, d_rear, d_front)

坐标系：与 URDF 一致（base: x 前 / y 左 / z 上）。每侧腿是平面机构，两条销轴的
(x, z) 分量构成 4 个约束方程，解 4 个被动关节 (child1, child2, child3, shank)。

自检：python leg_kinematics_ref.py            # 与 leg_model.py(MuJoCo) 逐点对比
"""

from __future__ import annotations

import math

import numpy as np

PI = math.pi

# ---------------------------------------------------------------------------
# 几何常数：全部直接抄自 urdf/rl_chassis_v11_std.urdf（单位 m；关节轴都是局部 +z）
#   mount_*  : 该关节在 base_link 系里的原点与姿态(rpy)
#   A_B      : 关节 B 相对关节 A 的原点（写在 A 的子连杆坐标系里）
#   pin*_A/B : 销轴点在该连杆自身坐标系里的位置
# ---------------------------------------------------------------------------
LEGS = {
    "left": dict(
        mount_front=((-0.001951, 0.1755, 0.11508), (-PI / 2, 0.0, 0.0)),
        mount_rear=((-0.001951, 0.1515, 0.11508), (-PI / 2, 0.0, 0.0)),
        front_child1=(0.0999999789, 1.56497905e-07, -0.0209999983),
        child1_child2=(-0.100071446, 0.0627908435, 0.0134999983),
        child2_child3=(-0.136668405, -0.110450469, -0.012499876),
        rear_shank=(-0.21499951, -0.00048882211, 0.017999212),
        shank_wheel=(0.21484219, 0.13549473, 0.017550293),
        pin1_A=(-0.099927571748, -0.063019390865, -0.0028752271278),   # 在 child2 上
        pin1_B=(-0.099999038848, -0.00022839022379, 0.013624772872),   # 在 rear_link 上
        pin2_A=(-0.11500040187, -0.00026465136291, 0.014699563538),    # 在 child3 上
        pin2_B=(-0.036740763971, -0.047435297612, 0.00070047553825),   # 在 shank 上
    ),
    "right": dict(
        mount_front=((-0.001951, -0.1755, 0.11508), (PI / 2, 0.0, 0.0)),
        mount_rear=((-0.001951, -0.1515, 0.11508), (PI / 2, 0.0, 0.0)),
        front_child1=(0.100000046, -1.67484869e-07, -0.0209999983),
        child1_child2=(-0.100071359, -0.0627908325, 0.0134999983),
        child2_child3=(-0.136668391, 0.11045049, -0.012500124),
        rear_shank=(-0.21499938, 0.00048895405, 0.018000788),
        shank_wheel=(0.21484232, -0.1354946, 0.017549707),
        pin1_A=(-0.099927638948, 0.063019410142, -0.0028747712562),
        pin1_B=(-0.099998951948, 0.00022840951442, 0.013625228744),
        pin2_A=(-0.11500033997, 0.00026483636101, 0.0097004366532),
        pin2_B=(-0.036740663968, 0.047435371683, -0.0043004753468),
    ),
}

FREE = ("child1", "child2", "child3", "shank")   # 被动关节（闭环解出来的）


def rpy_to_R(rpy):
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class LegRef:
    def __init__(self, side):
        self.side = side
        self.c = LEGS[side]
        self._serial_constants()

    # ---- 正运动学：返回两个销轴在世界系(base)的位置 ----
    def pins(self, q):
        c = self.c
        out = {}
        # 前链: base -> crank -> child1 -> child2 -> child3
        # 注意：关节原点是写在「父连杆」坐标系里的，所以先取父系姿态放原点，
        #       再把关节角叠到子连杆姿态上。
        p = np.array(c["mount_front"][0])
        R_crank = rpy_to_R(c["mount_front"][1]) @ rot_z(q["front"])
        p1 = p + R_crank @ np.array(c["front_child1"])
        R1 = R_crank @ rot_z(q["child1"])
        p2 = p1 + R1 @ np.array(c["child1_child2"])
        R2 = R1 @ rot_z(q["child2"])
        out["pin1_A"] = p2 + R2 @ np.array(c["pin1_A"])
        p3 = p2 + R2 @ np.array(c["child2_child3"])
        R3 = R2 @ rot_z(q["child3"])
        out["pin2_A"] = p3 + R3 @ np.array(c["pin2_A"])
        # 后链: base -> rear(thigh) -> shank
        p = np.array(c["mount_rear"][0])
        Rr = rpy_to_R(c["mount_rear"][1]) @ rot_z(q["rear"])
        ps = p + Rr @ np.array(c["rear_shank"])
        out["pin1_B"] = p + Rr @ np.array(c["pin1_B"])
        Rs = Rr @ rot_z(q["shank"])
        out["pin2_B"] = ps + Rs @ np.array(c["pin2_B"])
        out["wheel_center"] = ps + Rs @ np.array(c["shank_wheel"])
        out["hip"] = p
        out["knee"] = ps
        return out

    def residual(self, q):
        p = self.pins(q)
        # 平面机构：只取 base 系 (x, z) 分量
        return np.array([
            p["pin1_A"][0] - p["pin1_B"][0], p["pin1_A"][2] - p["pin1_B"][2],
            p["pin2_A"][0] - p["pin2_B"][0], p["pin2_A"][2] - p["pin2_B"][2],
        ])

    def _solve(self, fixed, unknowns, seed=None, tol=1e-14, iters=60, max_step=0.35):
        """通用闭环求解：fixed 里的关节固定，解 unknowns 里的关节。"""
        q = {k: 0.0 for k in ("front", "child1", "child2", "child3", "rear", "shank", "wheel")}
        q.update({k: float(v) for k, v in fixed.items()})
        x = np.zeros(len(unknowns)) if seed is None else np.asarray(seed, dtype=float).copy()
        for _ in range(iters):
            for i, k in enumerate(unknowns):
                q[k] = x[i]
            r = self.residual(q)
            if np.linalg.norm(r) < tol:
                break
            J = np.zeros((4, len(unknowns)))
            eps = 1e-7
            for i in range(len(unknowns)):
                xp = x.copy(); xp[i] += eps
                for j, k in enumerate(unknowns):
                    q[k] = xp[j]
                J[:, i] = (self.residual(q) - r) / eps
            try:
                dx = np.linalg.solve(J, -r)
            except np.linalg.LinAlgError:
                dx = -np.linalg.lstsq(J, r, rcond=None)[0]
            step = float(np.max(np.abs(dx)))
            if step > max_step:
                dx *= max_step / step
            x += dx
            if np.linalg.norm(dx) < 1e-15:
                break
        for i, k in enumerate(unknowns):
            q[k] = x[i]
        return q, float(np.linalg.norm(self.residual(q))), x

    def solve(self, q_front, q_rear, seed=None, **kw):
        """真机口径：给定 (前髋曲柄, 后髋大腿)，解 4 个被动关节。返回 (q dict, 残差, 新种子)。

        单片机建议改用解析解（每条销轴是一个二连杆求交，共 2 组二解，用连续性挑分支），
        这里的牛顿法只用了几行、跑 100 Hz 也没问题（若 MCU 有余量可直接照抄）。
        """
        return self._solve({"front": q_front, "rear": q_rear}, FREE, seed, **kw)

    def solve_virtual(self, q_rear, q_shank, seed=None, **kw):
        """虚拟口径：给定 (大腿, 虚拟膝)，解出真机曲柄角与三个被动关节。"""
        return self._solve({"rear": q_rear, "shank": q_shank},
                           ("front", "child1", "child2", "child3"), seed, **kw)

    def jacobian(self, q_front, q_rear, seed=None, eps=1e-6):
        """返回 (dq_shank/dq_rear, dq_shank/dq_front)。"""
        def f(qf, qr):
            return self.solve(qf, qr, seed=seed)[0]["shank"]
        d_rear = (f(q_front, q_rear + eps) - f(q_front, q_rear - eps)) / (2 * eps)
        d_front = (f(q_front + eps, q_rear) - f(q_front - eps, q_rear)) / (2 * eps)
        return d_rear, d_front

    @staticmethod
    def torque_map(tau_hip_v, tau_knee_v, d_rear, d_front):
        """虚拟 (髋, 膝) 力矩 -> 真机 (后髋大腿, 前髋曲柄) 力矩（虚功原理）。

        δq_knee = d_rear·δq_rear + d_front·δq_front  =>
        τ_rear  = τ_hip  + τ_knee·d_rear
        τ_front = τ_knee·d_front
        """
        return tau_hip_v + tau_knee_v * d_rear, tau_knee_v * d_front

    # ---- 后链等效串联参数（训练环境里 L0/theta0 公式用的，部署不需要）----
    def _serial_constants(self):
        p = self.pins({k: 0.0 for k in ("front", "child1", "child2", "child3", "rear", "shank")})

        def ang(u, v):
            return math.atan2(v[2] - u[2], v[0] - u[0])
        self.l1 = float(math.hypot(p["knee"][0] - p["hip"][0], p["knee"][2] - p["hip"][2]))
        self.l2 = float(math.hypot(p["wheel_center"][0] - p["knee"][0],
                                   p["wheel_center"][2] - p["knee"][2]))
        self.a1 = ang(p["hip"], p["knee"])
        self.a2 = ang(p["knee"], p["wheel_center"]) - self.a1
        q0 = {k: 0.0 for k in ("front", "child1", "child2", "child3", "rear", "shank")}
        q0["rear"] = 0.1
        p1 = self.pins(q0)
        self.sgn_rear = float(np.sign(((ang(p1["hip"], p1["knee"]) - self.a1 + PI) % (2 * PI) - PI) / 0.1))
        q0["rear"] = 0.0; q0["shank"] = 0.1
        p2 = self.pins(q0)
        d = (ang(p2["knee"], p2["wheel_center"]) - ang(p["knee"], p["wheel_center"]) + PI) % (2 * PI) - PI
        self.sgn_shank = float(np.sign(d / 0.1))

    def virtual_leg(self, q_rear, q_shank):
        """返回 (L0, theta0)：L0 = 髋到轮心距离；theta0 = 相对竖直的倾角（0 = 竖直向下）。"""
        t1 = self.a1 + self.sgn_rear * q_rear
        t2 = self.a2 + self.sgn_shank * q_shank
        ex = self.l1 * math.cos(t1) + self.l2 * math.cos(t1 + t2)
        ez = self.l1 * math.sin(t1) + self.l2 * math.sin(t1 + t2)
        return float(math.hypot(ex, ez)), float(math.atan2(ex, -ez))

    def ik_virtual(self, L0, theta0, hint=(0.0, 0.0)):
        """(L0, theta0) -> (q_rear, q_shank)，取靠近 hint 的膝弯分支。"""
        ex, ez = L0 * math.sin(theta0), -L0 * math.cos(theta0)
        c2 = (L0 * L0 - self.l1 ** 2 - self.l2 ** 2) / (2 * self.l1 * self.l2)
        if abs(c2) > 1:
            raise ValueError("L0 超出可达范围")
        best = None
        for t2 in (math.acos(c2), -math.acos(c2)):
            t1 = math.atan2(ez, ex) - math.atan2(self.l2 * math.sin(t2), self.l1 + self.l2 * math.cos(t2))
            q_rear = ((t1 - self.a1 + PI) % (2 * PI) - PI) / self.sgn_rear
            q_shank = ((t2 - self.a2 + PI) % (2 * PI) - PI) / self.sgn_shank
            d = abs(q_rear - hint[0]) + abs(q_shank - hint[1])
            if best is None or d < best[0]:
                best = (d, float(q_rear), float(q_shank))
        return best[1], best[2]


def _selftest():
    """与 MuJoCo 版 leg_model.py 对比（需要 mujoco）。"""
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        import mujoco
        from leg_model import V11Leg
    except Exception as exc:
        print(f"跳过对照（没有 mujoco）：{exc}")
        return 0
    xml = "../../mjcf/rl_chassis_v12_weld.xml"
    m = mujoco.MjModel.from_xml_path(xml)
    worst = dict(q=0.0, res=0.0, jac=0.0, const=0.0)
    for side in ("left", "right"):
        ref = LegRef(side)
        mj = V11Leg(m, side)
        worst["const"] = max(worst["const"], abs(ref.l1 - mj.l1), abs(ref.l2 - mj.l2),
                             abs(ref.a1 - mj.a1), abs(ref.a2 - mj.a2),
                             abs(ref.sgn_rear - mj.sgn_rear), abs(ref.sgn_shank - mj.sgn_shank))
        rng = np.random.default_rng(0)
        seed_r, seed_m = None, None
        for _ in range(300):
            qf, qr = rng.uniform(-0.7, 0.7, 2)
            qr_, res_r, seed_r = ref.solve(qf, qr, seed=seed_r)
            qm_, res_m, seed_m = mj.solve_loop(qf, qr, seed=seed_m)
            if res_r > 1e-9 or res_m > 1e-9:
                continue
            worst["q"] = max(worst["q"], abs(qr_["shank"] - qm_["shank"]))
            worst["res"] = max(worst["res"], abs(res_r), abs(res_m))
            dr_r, df_r = ref.jacobian(qf, qr, seed=seed_r)
            dr_m, df_m = mj.jacobian(qf, qr, seed=seed_m)
            worst["jac"] = max(worst["jac"], abs(dr_r - dr_m), abs(df_r - df_m))
    print("纯 numpy 版 vs MuJoCo 版：")
    print(f"  串联常数偏差 {worst['const']:.2e}")
    print(f"  虚拟膝角偏差 {worst['q']:.2e} rad")
    print(f"  销残余      {worst['res']:.2e} m")
    print(f"  雅可比偏差  {worst['jac']:.2e}")
    return 0 if (worst["q"] < 1e-6 and worst["jac"] < 1e-3) else 1


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        raise SystemExit(_selftest())
    for side in ("left", "right"):
        leg = LegRef(side)
        print(f"[{side}] l1={leg.l1:.6f} l2={leg.l2:.6f} a1={math.degrees(leg.a1):.3f}deg "
              f"a2={math.degrees(leg.a2):.3f}deg sgn=({leg.sgn_rear:+.0f},{leg.sgn_shank:+.0f})")
        # 演示：站立姿态 L0=0.205 m、theta0=0 -> 真机的三个驱动关节角
        q_rear, q_shank = leg.ik_virtual(0.205, 0.0)
        q_v, res, seed = leg.solve_virtual(q_rear, q_shank)
        d_rear, d_front = leg.jacobian(q_v["front"], q_rear, seed=seed)
        print(f"      站立: q_rear={q_rear:+.6f} q_shank={q_shank:+.6f} "
              f"前髋(曲柄)角 q_front={q_v['front']:+.6f} 销残差={res*1e6:.3f} um")
        print(f"      该姿态 d_rear={d_rear:+.4f} d_front={d_front:+.4f}")
