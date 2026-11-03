#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Welded-base stance sweep for a closed-loop wheel-legged model.

基座焊死之后，驱动只剩每条腿两个髋关节（前 = 连杆根部，后 = 大腿根部，和 COD
官方 MJCF 的执行器一致）。这个脚本做两件事：

  1. 保位测试：用 PD 把四个髋关节压在 home 姿态上跑一段时间，看闭链残差、
     关节漂移、车轮是否还落在地面上；
  2. 站姿扫描：在 (前髋, 后髋) 平面上网格扫描，每个点短跑一下，记录车轮圆心
     的 (x, z) 和车轮最低点 z，画出可达区域与"刚好触地"的站姿流形。

用法：
    python3 tools/stance_sweep.py --xml mjcf/rl_chassis_v12_weld_scene.xml
    python3 tools/stance_sweep.py --xml ... --hold-seconds 3 --n 11 --plot /tmp/stance.png
"""

import argparse
import math
import os

import numpy as np
import mujoco


def leg_drives(m):
    """每条腿用来驱动闭链的两个关节（髋 + 连杆根，母体 = 基座），
    返回 {'Left': [j_front, j_rear], 'Right': [...]}。

    用**结构**判定而不是名字：基座 body 的子关节就是两个髋电机，
    这样右腿叫 Right_front_joint2/3_joint 的被动关节不会被误认成驱动。
    """
    root = 0
    for b in range(m.nbody):
        if m.body(b).name in ("base_link", "base"):
            root = b
            break
    out = {}
    for j in range(m.njnt):
        if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        if m.body_parentid[m.jnt_bodyid[j]] != root:
            continue
        link = m.body(m.jnt_bodyid[j]).name
        tag = "Left" if link.lower().startswith("left") else \
              "Right" if link.lower().startswith("right") else None
        if tag:
            out.setdefault(tag, []).append(j)
    for tag in out:
        out[tag].sort(key=lambda j: "rear" in m.joint(j).name.lower())
    return out


def wheel_geoms(m):
    return [g for g in range(m.ngeom)
            if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CYLINDER
            and "wheel" in m.geom(g).name.lower()]


def wheel_center_z(m, d, g):
    return float(d.geom_xpos[g][2])


def wheel_bottom(m, d, g):
    axis = d.geom_xmat[g].reshape(3, 3)[:, 2]
    return float(d.geom_xpos[g][2]
                 - m.geom_size[g][0] * math.sqrt(max(0.0, 1.0 - axis[2] ** 2)))


def contact_force_for(m, d, g):
    """该 geom 上的法向接触力之和。"""
    f = 0.0
    for c in range(d.ncon):
        con = d.contact[c]
        if g in (con.geom1, con.geom2):
            n = np.zeros(6)
            mujoco.mj_contactForce(m, d, c, n)
            f += abs(n[0])
    return float(f)


def pd_control(m, d, targets, kp, kd, wheel_ctrl=0.0):
    """targets: {关节名: 目标角}; 其余执行器给 wheel_ctrl。"""
    for a in range(m.nu):
        j = int(m.actuator_trnid[a][0])
        name = m.joint(j).name
        if name in targets:
            q, v = d.qpos[m.jnt_qposadr[j]], d.qvel[m.jnt_dofadr[j]]
            c = kp * (targets[name] - q) - kd * v
        else:
            c = wheel_ctrl
        d.ctrl[a] = float(np.clip(c, *m.actuator_ctrlrange[a]))


def run(m, d, targets, seconds, kp, kd, wheel_ctrl=0.0, record=None):
    n = int(seconds / m.opt.timestep)
    for k in range(n):
        pd_control(m, d, targets, kp, kd, wheel_ctrl)
        mujoco.mj_step(m, d)
        if record is not None:
            record(m, d, k)


def main():
    ap = argparse.ArgumentParser(description="焊死基座：保位测试 + 站姿扫描")
    ap.add_argument("--xml", required=True)
    ap.add_argument("--hold-seconds", type=float, default=3.0)
    ap.add_argument("--n", type=int, default=11, help="每个髋关节的网格点数")
    ap.add_argument("--span", type=float, default=1.0, help="扫描范围 ±span rad")
    ap.add_argument("--sweep-seconds", type=float, default=0.6)
    ap.add_argument("--kp", type=float, default=120.0)
    ap.add_argument("--kd", type=float, default=6.0)
    ap.add_argument("--plot", default=None)
    args = ap.parse_args()

    m = mujoco.MjModel.from_xml_path(os.path.abspath(args.xml))
    d = mujoco.MjData(m)
    drives = leg_drives(m)
    wg = wheel_geoms(m)
    pairs = [(int(m.eq_obj1id[e]), int(m.eq_obj2id[e])) for e in range(m.neq)]

    def gap():
        return max(float(np.linalg.norm(d.site_xpos[a] - d.site_xpos[b]))
                   for a, b in pairs)

    print("== 保位测试（PD 压在 home 姿态）")
    mujoco.mj_resetDataKeyframe(m, d, 0)
    tgt = {m.joint(j).name: 0.0 for js in drives.values() for j in js}
    print("  驱动关节: " + ", ".join(f"{k}" for k in tgt))
    maxgap, maxdrift = 0.0, 0.0

    def rec(mm, dd, k):
        nonlocal maxgap, maxdrift
        maxgap = max(maxgap, gap())
        for j in (j for js in drives.values() for j in js):
            maxdrift = max(maxdrift, abs(dd.qpos[mm.jnt_qposadr[j]]))

    run(m, d, tgt, args.hold_seconds, args.kp, args.kd, record=rec)
    print(f"  闭链最大残差 {maxgap*1e6:.1f} µm，驱动关节最大漂移 {maxdrift*180/math.pi:.3f}°")
    free = next((j for j in range(m.njnt)
                 if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE), None)
    if free is not None:
        a = int(m.jnt_qposadr[free])
        quat = d.qpos[a + 3:a + 7]
        tilt = math.degrees(2 * math.acos(min(1.0, abs(quat[0]))))
        print("  基座 pos={} 倾角 {:.3f}°（free 基座：倾角小就是站得住）".format(
            np.round(d.qpos[a:a + 3], 4), tilt))
    for g in wg:
        print(f"  {m.geom(g).name:22s} 轮心 z={wheel_center_z(m,d,g)*1000:+7.2f} mm  "
              f"最低点 z={wheel_bottom(m,d,g)*1000:+7.2f} mm  "
              f"法向接触力 {contact_force_for(m,d,g):6.2f} N")

    print(f"== 站姿扫描（每髋 ±{args.span} rad，{args.n}×{args.n} 点，"
          f"每点 {args.sweep_seconds}s）")
    span = np.linspace(-args.span, args.span, args.n)
    rows = []
    for qf in span:
        for qr in span:
            mujoco.mj_resetDataKeyframe(m, d, 0)
            tt = {}
            for tag, js in drives.items():
                # 左腿关节世界轴 = +y，右腿 = -y（URDF 里左右镜像体现在 origin 的
                # rpy 上），所以**同一个符号**在两侧得到镜像对称的姿态。
                tt[m.joint(js[0]).name] = qf
                tt[m.joint(js[1]).name] = qr
            run(m, d, tt, args.sweep_seconds, args.kp, args.kd)
            rows.append(dict(qf=qf * (1.0), qr=qr,
                             x=float(d.geom_xpos[wg[0]][0]),
                             z=float(d.geom_xpos[wg[0]][2]),
                             zb=float(wheel_bottom(m, d, wg[0])),
                             gap=gap()))
    zs = np.array([r["zb"] for r in rows])
    touch = [r for r in rows if abs(r["zb"]) < 0.002]
    print(f"  扫描 {len(rows)} 个目标姿态；车轮最低点 z ∈ [{zs.min()*1000:+.1f},"
          f" {zs.max()*1000:+.1f}] mm")
    print(f"  其中 {len(touch)} 个把车轮压在地面上（|z|<2 mm）→ 这些是可用站姿")
    if touch:
        xs = np.array([r["x"] for r in touch])
        print(f"  可用站姿下轮心 x ∈ [{xs.min():+.4f}, {xs.max():+.4f}] m，"
              f"轮心 z ∈ [{min(r['z'] for r in touch):+.4f}, "
              f"{max(r['z'] for r in touch):+.4f}] m")
        print(f"  最大闭链残差 {max(r['gap'] for r in touch)*1e6:.1f} µm")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(13, 6))
        zz = np.array([[r["zb"] * 1000 for r in rows if r["qf"] == qf] for qf in span])
        im = ax[0].pcolormesh(span, span, zz, shading="auto", cmap="viridis")
        ax[0].contour(span, span, zz, levels=[-2, 0, 2], colors="w")
        ax[0].set_xlabel("front hip target [rad]")
        ax[0].set_ylabel("rear hip target [rad]")
        ax[0].set_title("wheel lowest point z [mm]")
        fig.colorbar(im, ax=ax[0])
        for r in rows:
            ax[1].plot(r["x"], r["z"] * 1000, ".", ms=3,
                       color=("tab:red" if abs(r["zb"]) < 0.002 else "tab:gray"))
        ax[1].axhline(60.0, color="k", lw=1, ls="--")
        ax[1].set_xlabel("wheel centre x [m]")
        ax[1].set_ylabel("wheel centre z [mm]")
        ax[1].set_title("reachable wheel centre (red = touching ground)")
        ax[1].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.plot, dpi=130)
        print("  已画 " + args.plot)


if __name__ == "__main__":
    main()
