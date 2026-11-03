#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
求对齐后 rl_chassis 模型的静态平衡站立姿态，并在 MuJoCo 里用 PD 顶住验证。

姿态定义（与 align_frame.py 输出的 _std.urdf 配套）：
  左腿 (=q1, q2)，右腿 = (-q1, -q2)，车轮关节 = 0
  约束 1: 车体水平、轮子着地、base_link 离地高度 = h
  约束 2: 整机质心水平投影落在轮子接地线上（平衡点）
两个未知量 (q1, q2)，两个方程，用牛顿法解。

用法:
    python3 tools/standing_pose.py                 # 默认扫一遍高度并验证
    python3 tools/standing_pose.py --height 0.12   # 只解某个高度并仿真验证
"""

import argparse
import math
import os
import re
import shutil
import tempfile

import numpy as np
import xml.etree.ElementTree as ET
import mujoco

WHEEL_R = 0.06          # 轮子半径 (meshes 里量出来是 0.06)
LEG_JOINTS = ["l_big_hit_joint", "l_knee_joint", "l_wheel_joint",
              "r_big_hit_joint", "r_knee_joint", "r_wheel_joint"]


def detect_leg_chains(urdf):
    """
    自动识别两条腿的 (髋, 膝, 轮) 关节名。
    从 base_link 的每个子关节往下走，能找到名字含 "wheel" 的 link 的那条链就是
    主腿链（v9 每腿还有一条从 base 直接出来的连杆分支，走不到 wheel，会被排除）。
    侧别用子件在 y 方向的正负判断。
    """
    root = ET.parse(urdf).getroot()
    kids = {}
    for j in root.findall("joint"):
        kids.setdefault(j.find("parent").get("link"), []).append(
            (j.get("name"), j.find("child").get("link")))

    def walk(link, path):
        if "wheel" in link.lower():
            return path, link
        for jn, ch in kids.get(link, []):
            got = walk(ch, path + [jn])
            if got:
                return got
        return None

    chains = {}
    for jn, ch in kids.get("base_link", []):
        got = walk(ch, [jn])
        if got and len(got[0]) == 3:
            hip, knee, wheel = got[0]
            side = "l" if "l" in hip[:2].lower() or "l" in ch[:2].lower() else "r"
            side = "l" if ch.lower().startswith("l") else "r"
            chains[side] = (hip, knee, wheel)
    assert set(chains) == {"l", "r"}, "腿链识别失败: %s" % chains
    return chains


# --------------------------------------------------------------------------
def prep(urdf, workdir):
    os.makedirs(workdir, exist_ok=True)
    txt = open(urdf, encoding="utf-8").read()
    txt = re.sub(r"<mujoco>.*?</mujoco>", "", txt, flags=re.S)
    txt = re.sub(r'filename="([^"]+)"',
                 lambda m: 'filename="%s"' % os.path.basename(m.group(1)), txt)
    meshdir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(urdf)), "..", "meshes"))
    txt = txt.replace("</robot>",
                      '  <mujoco><compiler meshdir="%s"/></mujoco>\n</robot>' % meshdir)
    out = os.path.join(workdir, "model.urdf")
    open(out, "w", encoding="utf-8").write(txt)
    return out


class Rig:
    def __init__(self, urdf, workdir, floating=True, floor=True, self_collision=False,
                 wheel_cylinder=False):
        p = prep(urdf, workdir)
        spec = mujoco.MjSpec.from_file(p)
        self.base_body = spec.body("base_link")
        if floating:
            self.base_body.add_freejoint()
        # 这个 URDF 的碰撞体直接用 STL 网格，MuJoCo 取凸包：底座（含侧面装甲板）
        # 的凸包会把腿一起包进去，于是腿和底盘产生伪碰撞。训练侧也是同样处理
        # （gym 配置里 self_collisions = 1 表示关闭）。
        # 注意：不能把 contype/conaffinity 同时置 0 —— MuJoCo 会把这种 geom
        # 直接从模型里删掉（连网格一起没了）。这里用 机器人: contype=1,
        # conaffinity=0 / 地面: contype=0, conaffinity=1 的写法，两个条件都满足：
        # 机器人内部两两不碰撞，但 轮子<->地面 正常碰撞。
        if not self_collision:
            for b in spec.bodies:
                if b.name == "world":
                    continue
                for g in b.geoms:
                    g.contype = 1
                    g.conaffinity = 0
        self.floor = None
        if floor:
            g = spec.worldbody.add_geom()
            g.name = "floor"
            g.type = mujoco.mjtGeom.mjGEOM_PLANE
            g.size = [5.0, 5.0, 0.05]
            g.pos = [0.0, 0.0, 0.0]
            g.contype = 0
            g.conaffinity = 1
            g.rgba = np.array([0.45, 0.47, 0.5, 1.0])
        # 轮子的碰撞体换成圆柱（轮子本来就是圆柱）。三角网格的最低顶点会偏离
        # 轮心 1.7 mm，在轮心正下方这个平衡点上是 0.13 N·m 的残余力矩，足以让
        # 倒立摆指数发散；换成解析圆柱后平衡点才是精确的。
        if wheel_cylinder:
            for b in spec.bodies:
                if "wheel" not in b.name.lower():
                    continue
                for g in b.geoms:
                    g.contype = 0
                    g.conaffinity = 0
                cg = b.add_geom()
                cg.name = b.name + "_cyl"
                cg.type = mujoco.mjtGeom.mjGEOM_CYLINDER
                cg.size = [WHEEL_R, 0.0292, 0.0]
                cg.pos = [0.0, 0.0, 0.0]
                cg.contype = 1
                cg.conaffinity = 0
        self.m = spec.compile()
        self.d = mujoco.MjData(self.m)
        self.chain = detect_leg_chains(urdf)
        self.leg_joints = [j for s in ("l", "r") for j in self.chain[s]]
        self.all_joints = [mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_JOINT, i)
                           for i in range(self.m.njnt)
                           if self.m.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE]
        self.jid = {n: mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, n)
                    for n in self.all_joints}
        self.qadr = {n: self.m.jnt_qposadr[i] for n, i in self.jid.items()}
        self.dadr = {n: self.m.jnt_dofadr[i] for n, i in self.jid.items()}
        self.bid = {n: mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, n)
                    for n in ("base_link",)}
        self.wheel_bid = {s: int(self.m.jnt_bodyid[self.jid[self.chain[s][2]]])
                          for s in ("l", "r")}
        self.has_free = floating

    def set_pose(self, q1, q2, base=(0.0, 0.0, 0.0)):
        d, m = self.d, self.m
        d.qpos[:] = 0.0
        d.qvel[:] = 0.0
        if self.has_free:
            d.qpos[0:3] = base
            d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        for side, q in (("l", (q1, q2)), ("r", (-q1, -q2))):
            hip, knee, wheel = self.chain[side]
            d.qpos[self.qadr[hip]] = q[0]
            d.qpos[self.qadr[knee]] = q[1]
            d.qpos[self.qadr[wheel]] = 0.0
        mujoco.mj_forward(m, d)

    def wheel_xz(self):
        p = self.d.xpos[self.wheel_bid["l"]]
        return p[0], p[2]

    def com(self):
        return self.d.subtree_com[0].copy()

    def tilt_deg(self):
        R = self.d.xmat[self.bid["base_link"]].reshape(3, 3)
        return math.degrees(math.acos(min(1.0, max(-1.0, R[2, 2]))))


# --------------------------------------------------------------------------
def solve(h, rig, seed=(0.0, 0.0)):
    """解 (q1,q2)：使轮底刚好接地于 base 高度 h，且质心在接地线上。"""
    q = np.array(seed, dtype=float)

    def resid(qq):
        rig.set_pose(qq[0], qq[1])
        xw, zw = rig.wheel_xz()
        c = rig.com()
        return np.array([zw - (WHEEL_R - h), c[0] - xw])

    for _ in range(200):
        r = resid(q)
        if np.abs(r).max() < 1e-11:
            break
        J = np.zeros((2, 2))
        eps = 1e-7
        for k in range(2):
            dq = np.zeros(2)
            dq[k] = eps
            J[:, k] = (resid(q + dq) - r) / eps
        try:
            step = np.linalg.solve(J + 1e-9 * np.eye(2), -r)
        except np.linalg.LinAlgError:
            return None, "雅可比奇异"
        # 限制步长，避免跳到别的分支
        n = np.abs(step).max()
        if n > 0.2:
            step = step * 0.2 / n
        q = q + step
        if np.abs(q).max() > 3.0:
            return None, "跑出范围"
    r = resid(q)
    if np.abs(r).max() > 1e-8:
        return None, "未收敛 (残差 %.2e)" % np.abs(r).max()
    return q, None


def solve_pose(h, rig, seed=(0.0, 0.0)):
    """解完再取回质心/轮心等实际量。"""
    q, err = solve(h, rig, seed)
    if q is None:
        return None, err
    rig.set_pose(q[0], q[1])
    xw, zw = rig.wheel_xz()
    c = rig.com()
    return dict(q1=q[0], q2=q[1], com=c, wheel_x=xw, wheel_z=zw), None


# --------------------------------------------------------------------------
def sim_check(rig_factory, pose, h, x0=0.0, omega=1500.0, zeta=0.7, dt=2e-4,
              lock_wheels=True, T=3.0, assist_k=0.0, assist_d=0.0):
    """
    把机器人放到解出来的姿态上演一遍，PD 顶住髋/膝，看会不会塌。
    增益按质量矩阵自动整定：kp = M_ii*omega^2, kd = 2*zeta*M_ii*omega
    （显式积分要求 kd*dt/M_ii = 2*zeta*omega*dt < 2，取 dt=2e-4 很宽松）。
    """
    r = rig_factory()
    m, d = r.m, r.d
    m.opt.timestep = dt
    q1, q2 = pose["q1"], pose["q2"]
    r.set_pose(q1, q2, base=(x0, 0.0, h))
    joints = r.all_joints                 # 所有关节都顶住（v9 多出来的连杆也保持原样）
    dof = {n: r.dadr[n] for n in joints}
    # 每个关节的等效惯量：qvel = e_i 时的动能 = 0.5*M_ii
    m.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_ENERGY
    gains = {}
    for j in joints:
        d.qvel[:] = 0.0
        d.qvel[dof[j]] = 1.0
        mujoco.mj_forward(m, d)
        I = float(2.0 * d.energy[1])
        gains[j] = (I * omega ** 2, 2.0 * zeta * I * omega, I)
    r.set_pose(q1, q2, base=(x0, 0.0, h))
    target = {n: float(d.qpos[r.qadr[n]]) for n in joints}   # 目标就是当前姿态
    # MJCF 里带 actuator 的关节改用 ctrl 驱动
    act = {}
    for i in range(m.nu):
        jid = int(m.actuator_trnid[i, 0])
        jn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if jn:
            act[jn] = i
    tilt0 = r.tilt_deg()
    # 基座姿态稳定器（可选）：倒立摆本身不稳定，用一个很弱的虚拟弹簧把
    # 车体俯仰/横滚拉住，就能把"这个姿态是不是平衡点"量化出来。
    free_dof = int(m.jnt_dofadr[0]) if m.njnt and m.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE else None
    log = []
    assist_hist = []
    tau_hist = {j: [] for j in joints}
    for _ in range(int(T / dt)):
        for j in joints:
            a = dof[j]
            wheel = j in (r.chain["l"][2], r.chain["r"][2])
            if wheel and not lock_wheels:
                tau = 0.0
            else:
                kp, kd, _ = gains[j]
                tau = kp * (target[j] - d.qpos[r.qadr[j]]) - kd * d.qvel[a]
            if j in act:
                d.ctrl[act[j]] = tau
            else:
                d.qfrc_applied[a] = tau
            tau_hist[j].append(tau)
        mujoco.mj_step(m, d)
        if free_dof is not None and assist_k > 0:
            q = d.qpos[3:7] / max(1e-12, np.linalg.norm(d.qpos[3:7]))
            w, v = q[0], q[1:]
            n = np.linalg.norm(v)
            rot = (2.0 * math.atan2(n, w) * v / n) if n > 1e-12 else np.zeros(3)
            for k in (0, 1):
                d.qfrc_applied[free_dof + 3 + k] = -assist_k * rot[k] - assist_d * d.qvel[free_dof + 3 + k]
            assist_hist.append(max(abs(-assist_k * rot[0] - assist_d * d.qvel[free_dof + 3]),
                                   abs(-assist_k * rot[1] - assist_d * d.qvel[free_dof + 4])))
        log.append((d.qpos[0], d.qpos[2], r.tilt_deg()))
    log = np.array(log)
    ntail = max(1, len(log) // 6)
    tau = {j: float(np.mean(v[-ntail:])) for j, v in tau_hist.items()}
    return dict(tilt0=tilt0, tilt_end=float(log[-1, 2]), tilt_max=float(log[:, 2].max()),
                h_end=float(log[-1, 1]), x_end=float(log[-1, 0]),
                tau=tau, gains={j: (g[0], g[2]) for j, g in gains.items()},
                dt=dt, n_steps=len(log), d=d, r=r,
                assist=float(np.mean(assist_hist[-ntail:])) if assist_hist else None,
                trace=[(i * dt, float(log[i, 2]), float(log[i, 1]))
                       for i in range(0, len(log), max(1, len(log) // 6))])


# --------------------------------------------------------------------------
def vmc_hint(rig):
    """给出与 V4 风格 VMC 对应的平面运动学关系。"""
    rig.set_pose(0.0, 0.0)
    hip, knee, wheel = rig.chain["l"]
    ph = rig.d.xpos[int(rig.m.jnt_bodyid[rig.jid[hip]])]
    pk = rig.d.xpos[int(rig.m.jnt_bodyid[rig.jid[knee]])]
    pw = rig.d.xpos[rig.wheel_bid["l"]]
    l1 = float(np.linalg.norm(pk - ph))
    l2 = float(np.linalg.norm(pw - pk))
    a_thigh0 = math.atan2(pk[2] - ph[2], pk[0] - ph[0])
    a_shin0 = math.atan2(pw[2] - pk[2], pw[0] - pk[0])
    return l1, l2, a_shin0, a_thigh0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                   "..", "urdf", "rl_chassis_v7.urdf"))
    ap.add_argument("--height", type=float, default=None, help="只解/验证这个高度")
    ap.add_argument("--lock-wheels", default="1")
    args = ap.parse_args()

    work = tempfile.mkdtemp(prefix="rlchassis_")
    rig = Rig(args.urdf, os.path.join(work, "solve"), floating=True, floor=False)

    print("== 质量 %.4f kg, 关节 %d, 自由度 %d" % (rig.m.body_mass.sum(), rig.m.njnt, rig.m.nv))
    l1, l2, a0, at0 = vmc_hint(rig)
    print("== 识别到的腿链: %s / %s" % (rig.chain["l"], rig.chain["r"]))
    print("== 平面运动学: l1=%.5f  l2=%.5f  thigh 零位角 %.3f rad  shin 零位角 %.3f rad"
          % (l1, l2, at0, a0))

    if args.height is None:
        print("\n== 扫描站立高度（左腿 q1=髋, q2=膝; 右腿取负）")
        print("   %-8s %-22s %-22s %-10s %-10s" % ("高度h(m)", "左腿髋 q1(rad/deg)", "左腿膝 q2(rad/deg)", "质心x(mm)", "轮心x(mm)"))
        seed = (0.0, 0.0)
        for h in [0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.18, 0.20]:
            pose, err = solve_pose(h, rig, seed)
            if pose is None:
                print("   %-8.3f 失败: %s" % (h, err))
                continue
            seed = (pose["q1"], pose["q2"])
            print("   %-8.3f %-22s %-22s %-10.2f %-10.2f" % (
                h,
                "%+.4f / %+.2f" % (pose["q1"], math.degrees(pose["q1"])),
                "%+.4f / %+.2f" % (pose["q2"], math.degrees(pose["q2"])),
                pose["com"][0] * 1000, pose["wheel_x"] * 1000))

    # 对指定高度做完整验证
    h = args.height if args.height is not None else 0.12
    pose, err = solve_pose(h, rig, seed=(0.0, 0.0))
    if pose is None:
        print("高度 %.3f 求解失败: %s" % (h, err))
        return
    print("\n== h=%.3f 的站立姿态（左腿）: 髋 %+.4f rad (%+.2f°), 膝 %+.4f rad (%+.2f°)"
          % (h, pose["q1"], math.degrees(pose["q1"]), pose["q2"], math.degrees(pose["q2"])))
    print("   右腿: 髋 %+.4f, 膝 %+.4f ; 车轮 = 0" % (-pose["q1"], -pose["q2"]))
    print("   质心 %.4f %.4f %.4f, 轮心 x=%.4f -> 质心投影偏差 %.2f mm"
          % (*pose["com"], pose["wheel_x"], (pose["com"][0] - pose["wheel_x"]) * 1000))

    # 平面角度
    print("\n== VMC 用到的平面角（左腿）:")
    print("   thigh 方向角  a1 = %.4f - q1          = %+.4f rad" % (at0, at0 - pose["q1"]))
    print("   shin  方向角  a2 = %.4f - q1 - q2     = %+.4f rad" % (a0, a0 - pose["q1"] - pose["q2"]))
    print("   => 髋->轮 向量 = l1*(cos a1, sin a1) + l2*(cos a2, sin a2)")
    print("   V4 风格 VMC 的等效置零: theta1 = %.4f - q1, theta2 = %.4f - q2"
          % (at0, a0 - at0))

    lock = args.lock_wheels in ("1", "true", "True")
    print("\n== MuJoCo PD 仿真验证 (轮子%s, 3 s)" % ("锁死" if lock else "自由"))
    # 静态残余力矩（绕轮子接地点）
    rig.set_pose(pose["q1"], pose["q2"])
    xc = rig.wheel_xz()[0]
    mom = sum(rig.m.body_mass[i] * 9.81 * (rig.d.xipos[i][0] - xc) for i in range(rig.m.nbody))
    print("   静态残余重力矩(绕轮心垂线) %+.3f N·m  (等效质心偏移 %+.2f mm)"
          % (mom, mom / (rig.m.body_mass.sum() * 9.81) * 1000))
    mk = lambda: Rig(args.urdf, os.path.join(work, "sim"),
                     floating=True, floor=True, wheel_cylinder=True)
    res = sim_check(mk, pose, h, lock_wheels=lock, assist_k=200.0, assist_d=20.0)
    print("   dt=%.0e, 步数 %d" % (res["dt"], res["n_steps"]))
    print("   倾角 %.3f° -> %.3f° (过程最大 %.3f°)" % (res["tilt0"], res["tilt_end"], res["tilt_max"]))
    print("   基座高度 %.4f -> %.4f m, 基座水平位移 %+.1f mm" % (h, res["h_end"], res["x_end"] * 1000))
    if res["assist"] is not None:
        print("   车体姿态稳定器(虚拟弹簧)稳态力矩 %.3f N·m  <- 越小说明越接近平衡点" % res["assist"])
    print("   时间轨迹 (t, 倾角°, 基座高度 m):")
    for t, tilt, hh in res["trace"]:
        print("      %5.2f s   %6.3f°   %.4f m" % (t, tilt, hh))
    print("   自动整定增益 (kp / 等效惯量 kg·m²):")
    for j in rig.leg_joints:
        print("      %-16s kp=%9.1f  I=%.2e" % (j, res["gains"][j][0], res["gains"][j][1]))
    for j in rig.leg_joints:
        print("   稳态关节力矩 %-16s %+.2f N·m" % (j, res["tau"][j]))
    # 对给定 Kp 解释成 PD 目标偏置（像 ur24 的 target_offset）
    kp_cfg = 20.0
    print("   Kp=%.0f 时对应的 PD 目标偏置 (tau/Kp):" % kp_cfg)
    for j in (rig.chain["l"][0], rig.chain["l"][1]):
        print("      %-16s %+.3f rad" % (j, res["tau"][j] / kp_cfg))


if __name__ == "__main__":
    main()
