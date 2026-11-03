#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把 urdf/rl_chassis_v7.urdf 转成可直接放进 MuJoCo 的原生 MJCF。

在 MuJoCo 自己转换结果之上做了这些事：
  1. 给 base_link 加 <freejoint/>。不加的话 MuJoCo 会把根 link 焊到世界系，
     并且**丢掉底盘的质量**（7.73 kg -> 2.12 kg）。
  2. 车轮关节去掉 range。URDF 里写的是 continuous，但导出器同时写了
     ±6.28 的 limit，MuJoCo 会当成真实行程 —— 轮子转一圈就被卡死。
  3. 接触过滤：机器人各 geom contype=1/conaffinity=0，地面 contype=0/
     conaffinity=1。这样机器人内部互不碰撞（STL 凸包会把底盘和腿包在一起，
     不关掉就是一堆伪碰撞），但都还能和地面/地形碰撞。
  4. 轮子碰撞体换成解析圆柱（半径 0.06、半长 0.0292），保留原网格做显示。
     网格化轮子的最低顶点偏离轮心 1.7 mm，在平衡点上就是 0.13 N·m 偏置。
  5. 6 个力矩电机 + 一个站立姿态 keyframe（默认 h=0.14 m）。

用法:
    python3 tools/make_mjcf.py                 # 生成 mjcf/rl_chassis_v7.xml 和 scene.xml
    python3 tools/make_mjcf.py --stand-height 0.12
"""

import argparse
import math
import os
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from standing_pose import Rig, solve_pose, detect_leg_chains   # noqa: E402

WHEEL_R, WHEEL_HALF_W = 0.06, 0.0292
MODEL_NAME = "rl_chassis_v9"
LEG_JOINTS = ["l_big_hit_joint", "l_knee_joint", "l_wheel_joint",
              "r_big_hit_joint", "r_knee_joint", "r_wheel_joint"]
WHEEL_JOINTS = {"l_wheel_joint", "r_wheel_joint"}


def build_spec(urdf, robot_name="base_link"):
    """只负责 URDF -> 结构 + 给 base_link 加 freejoint。

    接触过滤 / 轮子碰撞体 / 执行器 / keyframe 都在 XML 文本层做：
    MjSpec.to_xml() 会把"不参与碰撞"的 URDF 网格 geom 直接省略，所以在
    spec 里改 contype/conaffinity 反而会把轮子的显示网格弄丢。
    """
    spec = mujoco.MjSpec.from_file(urdf)
    spec.body(robot_name).add_freejoint()
    return spec


def fix_geoms(root):
    """接触过滤 + 轮子用圆柱碰撞（网格只做显示）。"""
    n_cyl = 0
    for body in root.iter("body"):
        name = (body.get("name") or "").lower()
        is_wheel = "wheel" in name
        for g in body.findall("geom"):
            if is_wheel:
                # 显示网格：不参与任何碰撞（原生 MJCF 里这种 geom 会保留）
                g.set("contype", "0")
                g.set("conaffinity", "0")
                g.set("group", "2")
                c = ET.SubElement(body, "geom")
                c.set("name", (body.get("name") or "wheel") + "_collision")
                c.set("type", "cylinder")
                c.set("size", "%.4g %.4g" % (WHEEL_R, WHEEL_HALF_W))
                c.set("pos", "0 0 0")
                c.set("group", "3")
                c.set("rgba", "0 0 0 0")
                c.set("contype", "1")
                c.set("conaffinity", "0")
                n_cyl += 1
            else:
                # 与地面碰撞，但机器人内部互不碰撞
                g.set("contype", "1")
                g.set("conaffinity", "0")
                # 底盘在 URDF 里是 alpha=0.25 的材质（CAD 导出的痕迹），
                # 在 viewer 里会糊成半透明，这里改成不透明。
                if g.get("mesh") == "base_link":
                    g.set("rgba", "0.72 0.72 0.78 1")
    assert n_cyl == 2, n_cyl


def detect_loop_pairs(rig, verbose=True):
    """
    自动找出需要闭链的「销钉对」。

    这类模型的平行四边形连杆，两端各有一个薄薄的销钉 link 表示共用的横向销轴；
    在零位（= CAD 装配位）下，这一对的位置**只相差横向一个量**（lx/lz 几乎相同、
    ly 差一个销轴长度）。据此成对匹配。

    返回值: [(body1, body2, anchor_on_body1)]，anchor 取 body2 原点在 body1 系下
    零位时的坐标——MuJoCo 的 <connect> 正是把 body2 的原点拉到 body1 的这个点上。
    """
    rig.set_pose(0.0, 0.0)
    m, d = rig.m, rig.d
    pins = []
    for i in range(m.nbody):
        n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i)
        if not n or n == "base_link":
            continue
        if m.body_mass[i] > 5e-3:          # 销钉都很轻
            continue
        if "dp" not in n.lower():
            continue
        pins.append((n, i, d.xpos[i].copy(), d.xmat[i].reshape(3, 3).copy()))
    pairs = []
    used = set()
    for a in range(len(pins)):
        for b in range(len(pins)):
            if a == b or pins[a][0] in used or pins[b][0] in used:
                continue
            na, ia, pa, Ra = pins[a]
            nb, ib, pb, Rb = pins[b]
            dy = abs(pa[1] - pb[1])                  # 横向（对齐后是 y）
            dxz = max(abs(pa[0] - pb[0]), abs(pa[2] - pb[2]))
            if dxz < 2e-3 and 5e-3 < dy < 0.06:      # 只差一个销轴长度
                anchor = Ra.T @ (pb - pa)            # body2 原点在 body1 系下的位置
                pairs.append((na, nb, anchor))
                used.add(na)
                used.add(nb)
    if verbose:
        print("== 识别到 %d 组销钉对（要闭的环）:" % len(pairs))
        for a, b, anc in pairs:
            print("     %-14s <-> %-14s  anchor=%s" % (a, b, np.round(anc, 5)))
    return pairs


def weld_pins_and_site_loops(root, rig, pairs, verbose=True):
    """
    按参考模型（training_scene_fix.xml）的做法闭链：

    1. **把销钉焊死**：销钉 link 自带一个转动关节，只要销钉自己能转，
       <connect> 就能靠"销钉自转"被满足，环其实没闭上。所以这里把销钉的关节
      从模型里去掉 —— body 没有关节就刚性焊在父件上（网格还留着）。
    2. **site 放在两个销钉中心的中点**（装配位上重合），左右各写一个 site，
       再用 <connect site1 site2> 连起来。
    3. 被动关节补 armature / damping（参考模型里所有关节都有），
       否则细连杆 + 硬约束在数值上很脆。
    """
    rig.set_pose(0.0, 0.0)
    m, d = rig.m, rig.d
    bodies = {b.get("name"): b for b in root.iter("body")}
    eq = ET.SubElement(root, "equality")
    n_site = 0
    for b1, b2, _anchor in pairs:
        i1 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b1)
        i2 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b2)
        p = 0.5 * (d.xpos[i1] + d.xpos[i2])            # 销轴中点（世界系，零位）
        for bname, i in ((b1, i1), (b2, i2)):
            R = d.xmat[i].reshape(3, 3)
            site = ET.SubElement(bodies[bname], "site")
            site.set("name", "loop_%s" % bname)
            site.set("pos", "%.6g %.6g %.6g" % tuple(R.T @ (p - d.xpos[i])))
            site.set("size", "0.005")
            # 焊死：删掉该 body 自己的关节
            for j in list(bodies[bname].findall("joint")):
                bodies[bname].remove(j)
            n_site += 1
        c = ET.SubElement(eq, "connect")
        c.set("name", "loop_%s_%s" % (b1, b2))
        c.set("site1", "loop_%s" % b1)
        c.set("site2", "loop_%s" % b2)
        c.set("solref", "0.006 1")
        c.set("solimp", "0.99 0.99 0.01 0.5 2")
    # 关节补 armature / damping（参考模型：0.01 / 0.2，轮子 0.005 / 0.02）
    n_j = 0
    for j in root.iter("joint"):
        if j.get("type") == "free" or j.get("name") is None:
            continue
        wheel = "wheel" in j.get("name").lower()
        if j.get("armature") is None:
            j.set("armature", "0.005" if wheel else "0.01")
        if j.get("damping") is None:
            j.set("damping", "0.02" if wheel else "0.2")
        n_j += 1
    if verbose:
        print("== 闭链：焊死 %d 个销钉、加了 %d 个 site、%d 个关节补 armature/damping"
              % (len(pairs) * 2, n_site, n_j))


def add_extras(root, pose, h, mujoco_file, chains, loop_pairs=()):
    """option / actuator / keyframe / 网格路径整理"""
    # option：齿传动的关节惯量小，implicitfast 更稳
    opt = root.find("option")
    if opt is None:
        opt = ET.SubElement(root, "option")
    opt.set("timestep", "0.002")
    opt.set("integrator", "implicitfast")
    opt.set("cone", "elliptic")
    opt.set("impratio", "10")
    opt.set("solver", "Newton")
    opt.set("iterations", "100")

    comp = root.find("compiler")
    if comp is None:
        comp = ET.Element("compiler")
        root.insert(0, comp)
    comp.set("angle", "radian")
    comp.set("autolimits", "true")

    # 车轮关节改成无行程
    for j in root.iter("joint"):
        if j.get("name") in WHEEL_JOINTS:
            if "range" in j.attrib:
                del j.attrib["range"]
            if "limited" in j.attrib:
                del j.attrib["limited"]

    # 执行器：只给两条腿的 6 个主动关节（额外的连杆/标记件是纯被动，不加驱动）
    act = ET.SubElement(root, "actuator")
    for side in ("l", "r"):
        hip, knee, wheel = chains[side]
        for jn, lim in ((hip, 40), (knee, 40), (wheel, 20)):
            short = jn.replace("_big_hit_joint", "_hip").replace("_Joint", "").replace("_joint", "")
            m = ET.SubElement(act, "motor")
            m.set("name", short)
            m.set("joint", jn)
            m.set("gear", "1")
            m.set("ctrlrange", "%.1f %.1f" % (-lim, lim))

    # 站立 keyframe：按 MJCF 里的关节顺序铺满（其余关节给 0）
    order, vals = [], {}
    for side in ("l", "r"):
        q1 = pose["q1"] if side == "l" else -pose["q1"]
        q2 = pose["q2"] if side == "l" else -pose["q2"]
        vals[chains[side][0]] = q1
        vals[chains[side][1]] = q2
        vals[chains[side][2]] = 0.0
    for body in root.iter("body"):
        for j in body.findall("joint"):
            if j.get("type") == "free" or j.get("name") is None:
                continue
            order.append(j.get("name"))
    qpos = [0.0, 0.0, h, 1.0, 0.0, 0.0, 0.0] + [vals.get(n, 0.0) for n in order]
    kf = ET.SubElement(root, "keyframe")
    k = ET.SubElement(kf, "key")
    k.set("name", "stand_h%.3f" % h)
    k.set("qpos", " ".join("%.6g" % v for v in qpos))

    root.set("model", MODEL_NAME)
    ET.indent(root, space="  ")
    with open(mujoco_file, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="utf-8"?>\n')
        f.write("<!-- 由 tools/make_mjcf.py 生成。坐标系 x 前 / y 左 / z 上；\n"
                "     base_link 带 freejoint；关节正方向已与 infantry_V4 对齐。\n"
                "     机器人内部关闭自碰撞；轮子用解析圆柱碰撞，网格仅做显示。\n"
                "     只给两条腿的髋/膝/轮加了力矩电机，其余连杆是纯被动（会自由摆动）。 -->\n")
        f.write(ET.tostring(root, encoding="unicode"))
        f.write("\n")


def write_scene(path, robot_file):
    scene = f'''<?xml version="1.0" encoding="utf-8"?>
<!-- 由 tools/make_mjcf.py 生成：带地面的可视化场景。
     训练时用 rl_chassis_v7.xml，自己准备地形即可。 -->
<mujoco model="rl_chassis_v7 scene">
  <include file="{robot_file}"/>
  <statistic center="0 0 0.15" extent="0.6"/>
  <visual>
    <headlight diffuse="0.7 0.7 0.7" ambient="0.35 0.35 0.35" specular="0.1 0.1 0.1"/>
    <global azimuth="140" elevation="-16" offwidth="1280" offheight="960"/>
    <map znear="0.02"/>
  </visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" width="256" height="256"
             rgb1="0.32 0.34 0.37" rgb2="0.26 0.28 0.31"/>
    <material name="grid" texture="grid" texrepeat="4 4" reflectance="0.05"/>
  </asset>
  <worldbody>
    <light pos="0 0 2.5" dir="0 0 -1" directional="true" castshadow="true"/>
    <geom name="floor" type="plane" size="4 4 0.1" material="grid"
          contype="0" conaffinity="1" condim="3" friction="1 0.005 0.0001"/>
  </worldbody>
</mujoco>
'''
    with open(path, "w", encoding="utf-8") as f:
        f.write(scene)


def settle_loops(robot_xml, h, chains, steps=6000, dt=5e-4):
    """
    闭链约束加上后，连杆关节必须转到"能把环合上"的角度，keyframe 才自洽。
    做法：先按站姿摆好，腿关节用硬 PD 顶住，其余关节加阻尼自由收敛，
    收敛后的关节角就写回 keyframe。
    """
    m = mujoco.MjModel.from_xml_path(robot_xml)
    d = mujoco.MjData(m)
    d.qpos[:] = np.array(m.key_qpos[0])          # 先用现成的 keyframe 摆好
    m.opt.timestep = dt
    names = [j for s in ("l", "r") for j in chains[s]]
    hold = {n: float(d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]])
            for n in names}
    act = {}
    for i in range(m.nu):
        jn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, int(m.actuator_trnid[i, 0]))
        if jn:
            act[jn] = i
    legs = set(names)
    free_dof = int(m.jnt_dofadr[0]) if m.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE else None
    for _ in range(steps):
        if free_dof is not None:                 # 收敛期间把基座钉住，别让它掉下去
            d.qpos[0:3] = [0.0, 0.0, h]
            d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
            d.qvel[free_dof:free_dof + 6] = 0.0
        for jn, v in hold.items():               # 腿关节硬锁：PD 顶不住硬约束会被拉偏
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)
            d.qpos[m.jnt_qposadr[jid]] = v
            d.qvel[m.jnt_dofadr[jid]] = 0.0
        for jn in hold:
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)
            a = m.jnt_dofadr[jid]
            tau = 200.0 * (hold[jn] - d.qpos[m.jnt_qposadr[jid]]) - 20.0 * d.qvel[a]
            if jn in act:
                d.ctrl[act[jn]] = tau
            else:
                d.qfrc_applied[a] = tau
        for jid in range(m.njnt):                # 其余（被动连杆）加阻尼帮助收敛
            jn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
            if jn in legs or m.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
                continue
            a = m.jnt_dofadr[jid]
            d.qfrc_applied[a] += -2.0 * d.qvel[a]
        mujoco.mj_step(m, d)
    q = np.array(d.qpos)
    q[0:3] = [0.0, 0.0, h]                       # 只把关节角带走，基座保持 h
    q[3:7] = [1.0, 0.0, 0.0, 0.0]
    return q


def verify(scene_path, h, omega=200.0, zeta=0.7, T=3.0, chains=None):
    """用交付的 MJCF 自己跑一遍：keyframe 起立、关节力矩 PD、看会不会塌。

    时间和增益都用 MJCF 自带设置，不用外部 magics：
      kp = M_ii*omega^2, kd = 2*zeta*M_ii*omega（显式积分要求 2*zeta*omega*dt < 2）
      另加一个很弱的车体姿态弹簧，用来量化"这个姿态离平衡点有多远"。
    """
    m = mujoco.MjModel.from_xml_path(scene_path)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    dt = m.opt.timestep
    names = [j for s_ in ("l", "r") for j in chains[s_]]
    all_j = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)
             if m.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE]
    dadr = {n: m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in names}
    qadr = {n: m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in all_j}
    act = {}
    for i in range(m.nu):
        jn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, int(m.actuator_trnid[i, 0]))
        if jn:
            act[jn] = i
    m.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_ENERGY
    adr = {n: m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in all_j}
    gains = {}
    for n in all_j:
        d.qvel[:] = 0.0
        d.qvel[adr[n]] = 1.0
        mujoco.mj_forward(m, d)
        I = float(2.0 * d.energy[1])
        gains[n] = (I * omega ** 2, 2.0 * zeta * I * omega)

    assert 2.0 * zeta * omega * dt < 2.0, "增益对时间步太大，显式阻尼会发散"
    mujoco.mj_resetDataKeyframe(m, d, 0)
    qdes = {n: float(d.qpos[qadr[n]]) for n in all_j}
    free = m.jnt_dofadr[0] if m.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE else None
    log, assist = [], []
    for _ in range(int(T / dt)):
        for n in all_j:
            kp, kd = gains[n]
            tau = kp * (qdes[n] - d.qpos[qadr[n]]) - kd * d.qvel[adr[n]]
            if n in act:
                d.ctrl[act[n]] = tau
            else:
                d.qfrc_applied[adr[n]] = tau        # 被动连杆也顶住，隔离出站姿本身
        if free is not None:
            q = d.qpos[3:7] / max(1e-12, np.linalg.norm(d.qpos[3:7]))
            w, v = q[0], q[1:]
            nv = np.linalg.norm(v)
            rot = (2.0 * math.atan2(nv, w) * v / nv) if nv > 1e-12 else np.zeros(3)
            for k in (0, 1):
                a = -30.0 * rot[k] - 3.0 * d.qvel[free + 3 + k]
                d.qfrc_applied[free + 3 + k] = a
                assist.append(abs(a))
        mujoco.mj_step(m, d)
        R = d.xmat[1].reshape(3, 3)
        log.append((d.qpos[2], math.degrees(math.acos(min(1.0, max(-1.0, R[2, 2]))))))
    log = np.array(log)
    nt = max(1, len(log) // 5)
    print("\n== 交付 MJCF 自检（dt=%.0e, %.0f s, 用文件自带 actuator）" % (dt, T))
    print("   倾角 0.000° -> %.3f°（过程最大 %.3f°）" % (log[-1, 1], log[:, 1].max()))
    print("   基座高度 %.4f -> %.4f m（变化 %+.1f mm）" % (h, log[-1, 0], (log[-1, 0] - h) * 1000))
    mm = np.mean(assist[-nt:]) / (m.body_mass.sum() * 9.81) * 1000
    print("   车体姿态弹簧稳态力矩 %.3f N·m（等效质心偏移 %.2f mm）-> %s"
          % (np.mean(assist[-nt:]), mm, "平衡点" if abs(mm) < 5 else "偏离平衡点"))
    tau = {n: float(d.ctrl[act[n]]) for n in names if n in act}
    print("   稳态关节力矩: " + "  ".join("%s=%+.2f" % (n.replace("_Joint", ""), v) for n, v in tau.items()))
    return log[-1, 1]


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    pkg = os.path.normpath(os.path.join(here, ".."))
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", default=os.path.join(pkg, "urdf", "rl_chassis_v7.urdf"))
    ap.add_argument("--outdir", default=os.path.join(pkg, "mjcf"))
    ap.add_argument("--stand-height", type=float, default=0.14)
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--name", default=MODEL_NAME, help="模型名（决定输出文件名）")
    ap.add_argument("--no-close-loops", action="store_true",
                    help="不生成 equality 闭链约束（默认生成）")
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # 站立姿态（用 URDF 解，和 MJCF 无关）
    import tempfile
    work = args.workdir or tempfile.mkdtemp(prefix="mjcf_")
    rig = Rig(args.urdf, os.path.join(work, "solve"), floating=True, floor=False)
    pose, err = solve_pose(args.stand_height, rig, (0.0, 0.0))
    assert pose is not None, err

    chains = detect_leg_chains(args.urdf)
    spec = build_spec(args.urdf)
    root = ET.fromstring(spec.to_xml())
    fix_geoms(root)
    mj = os.path.join(args.outdir, args.name + ".xml")
    loop_pairs = [] if args.no_close_loops else detect_loop_pairs(rig)
    if loop_pairs:
        weld_pins_and_site_loops(root, rig, loop_pairs)
    add_extras(root, pose, args.stand_height, mj, chains, loop_pairs)
    write_scene(os.path.join(args.outdir, "scene.xml"), args.name + ".xml")
    print("写出 %s\n写出 %s" % (mj, os.path.join(args.outdir, "scene.xml")))
    if loop_pairs:                     # 先把闭链收敛到自洽位置，再写回 keyframe
        q = settle_loops(mj, args.stand_height, chains)
        txt = open(mj, encoding="utf-8").read()
        import re as _re
        txt = _re.sub(r'(<key[^>]*qpos=")[^"]*(")', lambda mo: mo.group(1) + " ".join("%.8g" % v for v in q) + mo.group(2), txt)
        open(mj, "w", encoding="utf-8").write(txt)
        print("== 闭链已收敛，keyframe 已更新")
    print("腿链: %s / %s"%(chains["l"],chains["r"]))
    print("站立 keyframe: h=%.3f m, 左腿 髋=%+.4f 膝=%+.4f (右腿取负)"
          % (args.stand_height, pose["q1"], pose["q2"]))
    if not args.no_verify:
        verify(os.path.join(args.outdir, "scene.xml"), args.stand_height, chains=chains)


if __name__ == "__main__":
    main()
