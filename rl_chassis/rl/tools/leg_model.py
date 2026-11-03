#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""rl_chassis_v11 单腿运动学模型（RL 训练前的标定 + 部署时的虚拟腿映射）。

每条腿有两条链:
  后链（复旦口径的“串联腿”）: base -> rear_link(大腿) -> rear_child1(小腿) -> Wheel
  前链（真机上驱动膝的曲柄）: base -> front_Link(曲柄) -> child1 -> child2 -> child3
两个销把前链闭到后链上:
  pin1: front_child2 <-> rear_link
  pin2: front_child3 <-> rear_child1

本模块提供:
  * 后链的等效串联参数 (l1, l2, 角度零点 a1/a2, 关节旋转符号)，用来在训练环境里
    复刻复旦的 L0 / theta0 公式；
  * 闭环正逆解: 真机输入 (q_front 曲柄, q_rear 大腿) -> 四个被约束关节
    (child1/child2/child3/rear_child1)，其中 rear_child1 就是“虚拟膝角”；
  * 虚拟腿雅可比 dq_shank/d(q_front, q_rear): 状态映射（膝角速度）和力矩映射
    (虚拟髋/膝力矩 -> 真机 后髋/前髋力矩) 都用它，依据是虚功原理。
"""

from __future__ import annotations

import numpy as np
import mujoco


def wrap_to_pi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class V11Leg:
    """一条腿的闭环运动学。side 取 'left' / 'right'。"""

    # v11 左右腿命名不对称，这里显式列出
    _NAMES = {
        "left": dict(
            front="Left_front_joint",
            child1="Left_front_child1_joint",
            child2="Left_front_child2_joint",
            child3="Left_front_child3_joint",
            rear="Left_rear_joint",
            shank="Left_rear_child1_joint",
            wheel="Left_Wheel_joint",
            pin1_a="pin1_A",
            pin1_b="pin1_B",
            pin2_a="pin2_A",
            pin2_b="pin2_B",
            wheel_body="Left_Wheel_link",
        ),
        "right": dict(
            front="Right_front_joint",
            child1="Right_front_child1_joint",
            child2="Right_front_joint2_joint",
            child3="Right_front_joint3_joint",
            rear="Right_rear_joint",
            shank="Right_rear_child1_joint",
            wheel="Right_Wheel_joint",
            pin1_a="pin3_A",
            pin1_b="pin3_B",
            pin2_a="pin4_A",
            pin2_b="pin4_B",
            wheel_body="Right_Wheel_link",
        ),
    }

    def __init__(self, model: mujoco.MjModel, side: str, base_body: str = "base_link"):
        self.m = model
        self.side = side
        self._data = mujoco.MjData(model)  # 复用，避免每次分配
        n = self._NAMES[side]
        self.names = n
        # 关节
        self.ids = {
            k: self._jid(n[k])
            for k in ("front", "child1", "child2", "child3", "rear", "shank", "wheel")
        }
        self.qadr = {k: int(model.jnt_qposadr[v]) for k, v in self.ids.items()}
        self.dofadr = {k: int(model.jnt_dofadr[v]) for k, v in self.ids.items()}
        self.site_ids = {k: self._sid(n[k]) for k in ("pin1_a", "pin1_b", "pin2_a", "pin2_b")}
        self.wheel_body_id = int(
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n["wheel_body"])
        )
        self.base_body_id = int(
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, base_body)
        )
        # 真机只有 front / rear / wheel 三个电机；child1..3 与 shank 是闭环里的被动关节
        self.active = ("front", "rear", "wheel")
        self.free = ("child1", "child2", "child3", "shank")
        self._wheel_geometry()
        self._serial_constants()

    # ---------------- 基础 ----------------
    def _jid(self, name):
        i = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, name)
        if i < 0:
            raise KeyError(f"joint {name} not found")
        return int(i)

    def _sid(self, name):
        i = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, name)
        if i < 0:
            raise KeyError(f"site {name} not found")
        return int(i)

    def _wheel_geometry(self):
        for g in range(self.m.ngeom):
            if self.m.geom_bodyid[g] != self.wheel_body_id:
                continue
            if self.m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CYLINDER:
                self.wheel_radius = float(self.m.geom_size[g][0])
                self.wheel_half_width = float(self.m.geom_size[g][1])
                return
        raise RuntimeError(f"{self.side}: wheel cylinder geom not found")

    def _fk(self, **q):
        """把给定关节角塞进 qpos 后做纯运动学，返回关心的锚点位置（base 系）。"""
        d = self._data
        for k, v in q.items():
            d.qpos[self.qadr[k]] = v
        mujoco.mj_kinematics(self.m, d)
        hip = d.xpos[int(self.m.jnt_bodyid[self.ids["rear"]])].copy()
        knee = d.xpos[int(self.m.jnt_bodyid[self.ids["shank"]])].copy()
        wheel = d.xpos[self.wheel_body_id].copy()
        return hip, knee, wheel

    # ---------------- 后链的等效串联参数 ----------------
    def _serial_constants(self):
        """平面坐标取 (x, z) = (前, 上)，推出与复旦同形式的 l1 / l2 / 角度零点。"""
        hip0, knee0, wheel0 = self._fk(rear=0.0, shank=0.0)

        def ang(u, v):
            return float(np.arctan2(v[2] - u[2], v[0] - u[0]))

        self.l1 = float(np.hypot(knee0[0] - hip0[0], knee0[2] - hip0[2]))
        self.l2 = float(np.hypot(wheel0[0] - knee0[0], wheel0[2] - knee0[2]))
        self.a1 = ang(hip0, knee0)
        self.a2 = ang(knee0, wheel0) - self.a1
        self.hip_base = hip0.copy()
        # 关节正方向 -> 平面角的正方向（注意 ±pi 环绕：装配位姿大腿几乎朝后）
        h, k, _ = self._fk(rear=0.1, shank=0.0)
        self.sgn_rear = float(np.sign(wrap_to_pi(ang(h, k) - self.a1) / 0.1))
        h, k, w = self._fk(rear=0.0, shank=0.1)
        self.sgn_shank = float(np.sign(wrap_to_pi(ang(k, w) - ang(knee0, wheel0)) / 0.1))

    def virtual_leg(self, q_rear: float, q_shank: float, offset: float = 0.0,
                    convention: str = "down"):
        """虚拟腿: L0（髋到轮心的距离）与 theta0（倾角）。

        等价于 theta1 = a1 + sgn_rear*q_rear, theta2 = a2 + sgn_shank*q_shank,
        end_x = offset + l1*cos(theta1) + l2*cos(theta1+theta2), 同理 end_y。

        convention:
          "down"  : theta0 = atan2(ex, -ey)，腿竖直向下时为 0、轮在前为正（我们采用，
                    v11 装配位姿正好是竖直向下，不会有 ±pi 环绕跳变）
          "fudan" : theta0 = atan2(ey, ex) - pi/2，复旦原版口径
        """
        t1 = self.a1 + self.sgn_rear * q_rear
        t2 = self.a2 + self.sgn_shank * q_shank
        ex = offset + self.l1 * np.cos(t1) + self.l2 * np.cos(t1 + t2)
        ey = self.l1 * np.sin(t1) + self.l2 * np.sin(t1 + t2)
        if convention == "fudan":
            th = float(np.arctan2(ey, ex) - np.pi / 2)
        else:
            th = float(np.arctan2(ex, -ey))
        return float(np.hypot(ex, ey)), th

    def ik_virtual(self, L0: float, theta0: float, hint=(0.0, 0.0)):
        """(L0, theta0) -> (q_rear, q_shank)。两个膝弯分支里取靠近 hint 的那个。"""
        ex, ey = L0 * np.sin(theta0), -L0 * np.cos(theta0)
        c2 = (L0 * L0 - self.l1**2 - self.l2**2) / (2 * self.l1 * self.l2)
        if abs(c2) > 1:
            raise ValueError(f"L0={L0} 超出可达范围 [{abs(self.l1 - self.l2):.3f}, {self.l1 + self.l2:.3f}]")
        best = None
        for t2 in (np.arccos(c2), -np.arccos(c2)):
            t1 = np.arctan2(ey, ex) - np.arctan2(self.l2 * np.sin(t2), self.l1 + self.l2 * np.cos(t2))
            q_rear = wrap_to_pi(t1 - self.a1) / self.sgn_rear
            q_shank = wrap_to_pi(t2 - self.a2) / self.sgn_shank
            d = abs(q_rear - hint[0]) + abs(q_shank - hint[1])
            if best is None or d < best[0]:
                best = (d, float(q_rear), float(q_shank))
        return best[1], best[2]

    # ---------------- 闭环 ----------------
    def _pin_residual(self, q: dict, joints=("front", "child1", "child2", "child3", "rear", "shank")) -> np.ndarray:
        d = self._data
        for k in joints:
            d.qpos[self.qadr[k]] = q[k]
        mujoco.mj_kinematics(self.m, d)
        p = self.site_ids
        p1a, p1b = d.site_xpos[p["pin1_a"]], d.site_xpos[p["pin1_b"]]
        p2a, p2b = d.site_xpos[p["pin2_a"]], d.site_xpos[p["pin2_b"]]
        return np.array(
            [p1a[0] - p1b[0], p1a[2] - p1b[2], p2a[0] - p2b[0], p2a[2] - p2b[2]]
        )

    def _solve(self, fixed: dict, unknowns, seed=None, tol=1e-13, iters=60, max_step=0.35):
        """通用闭环求解：固定 fixed 指定的关节，解 unknowns 指定的关节。"""
        q = dict.fromkeys(("front", "child1", "child2", "child3", "rear", "shank", "wheel"), 0.0)
        q.update(fixed)
        joints = tuple(fixed) + tuple(unknowns)
        x = np.zeros(len(unknowns)) if seed is None else np.asarray(seed, dtype=float).copy()
        res = float("inf")
        for _ in range(iters):
            for i, k in enumerate(unknowns):
                q[k] = x[i]
            r = self._pin_residual(q, joints)
            res = float(np.linalg.norm(r))
            if res < tol:
                break
            J = np.zeros((4, 4))
            eps = 1e-7
            for i in range(4):
                xp = x.copy()
                xp[i] += eps
                for j, k in enumerate(unknowns):
                    q[k] = xp[j]
                J[:, i] = (self._pin_residual(q, joints) - r) / eps
            try:
                dx = np.linalg.solve(J, -r)
            except np.linalg.LinAlgError:
                dx = -np.linalg.lstsq(J, r, rcond=None)[0]
            if not np.all(np.isfinite(dx)):
                break
            # 限步长，避免一步跳到大变形分支
            step = float(np.max(np.abs(dx)))
            if step > max_step:
                dx *= max_step / step
            x += dx
            if np.linalg.norm(dx) < 1e-15:
                break
        for i, k in enumerate(unknowns):
            q[k] = x[i]
        return q, float(np.linalg.norm(self._pin_residual(q, joints))), x

    def solve_loop(self, q_front: float, q_rear: float, seed=None, **kw):
        """真机驱动输入 (曲柄, 大腿) -> 全部关节角。返回 (q dict, 残差, 种子)。"""
        return self._solve({"front": float(q_front), "rear": float(q_rear)}, self.free, seed, **kw)

    def solve_virtual(self, q_rear: float, q_shank: float, seed=None, **kw):
        """虚拟腿姿态 (大腿, 膝) -> 全部关节角（含真机曲柄角）。"""
        return self._solve(
            {"rear": float(q_rear), "shank": float(q_shank)},
            ("front", "child1", "child2", "child3"),
            seed,
            **kw,
        )

    def shank_from_inputs(self, q_front: float, q_rear: float, seed=None):
        """真机 (曲柄, 大腿) -> 虚拟膝角（小腿相对大腿的关节角）。"""
        q, res, seed4 = self.solve_loop(q_front, q_rear, seed=seed)
        return q["shank"], q, res, seed4

    def solutions(self, q_front: float, q_rear: float, n_start: int = 12, seed=None):
        """枚举同一输入下的多个装配分支，返回 [(残差, x4, q dict)]，按残差排序。

        机构在平面上确实有多解（镜像装配），部署时必须靠连续性锁死物理分支，
        这里用于量化说明。
        """
        starts = [np.zeros(4), np.zeros(4)]
        if seed is not None:
            starts[0] = np.asarray(seed, dtype=float)
        rng = np.random.default_rng(0)
        starts += [rng.uniform(-1.0, 1.0, 4) for _ in range(n_start)]
        found = []
        for s in starts:
            q, res, x = self.solve_loop(q_front, q_rear, seed=s)
            if res > 1e-9:
                continue
            if not any(np.max(np.abs(x - f[1])) < 1e-6 for f in found):
                found.append((res, x.copy(), q))
        found.sort(key=lambda t: t[0])
        return found

    def jacobian(self, q_front: float, q_rear: float, seed=None, eps=1e-6):
        """返回 (dq_shank/dq_rear, dq_shank/dq_front)。"""
        def f(qf, qr):
            return self.shank_from_inputs(qf, qr, seed=seed)[0]

        d_rear = (f(q_front, q_rear + eps) - f(q_front, q_rear - eps)) / (2 * eps)
        d_front = (f(q_front + eps, q_rear) - f(q_front - eps, q_rear)) / (2 * eps)
        return d_rear, d_front

    def torque_map(self, tau_rear_virtual: float, tau_shank_virtual: float, q_front: float, q_rear: float, seed=None):
        """虚拟 (大腿, 膝) 力矩 -> 真机 (后髋, 前髋) 力矩（虚功原理）。"""
        d_rear, d_front = self.jacobian(q_front, q_rear, seed=seed)
        return tau_rear_virtual + tau_shank_virtual * d_rear, tau_shank_virtual * d_front


def load(xml_path: str, base_body: str = "base_link"):
    """便捷入口：返回 (model, left_leg, right_leg)。"""
    m = mujoco.MjModel.from_xml_path(xml_path)
    return m, V11Leg(m, "left", base_body), V11Leg(m, "right", base_body)


if __name__ == "__main__":
    import sys

    xml = sys.argv[1] if len(sys.argv) > 1 else "../../mjcf/rl_chassis_v12_weld.xml"
    m, L, R = load(xml)
    for leg in (L, R):
        print(
            f"[{leg.side:5s}] l1={leg.l1:.6f} l2={leg.l2:.6f} "
            f"a1={np.degrees(leg.a1):8.3f}deg a2={np.degrees(leg.a2):8.3f}deg "
            f"sgn=(rear {leg.sgn_rear:+.0f}, shank {leg.sgn_shank:+.0f}) R_wheel={leg.wheel_radius:.5f}"
        )
        L0, th0 = leg.virtual_leg(0.0, 0.0)
        print(f"          q=0 装配位姿: L0={L0:.4f} m, theta0={np.degrees(th0):+.2f} deg")
