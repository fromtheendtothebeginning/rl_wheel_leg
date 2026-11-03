#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Closed-loop MJCF health check.

检查闭链 MJCF 到底有没有闭合、自由度对不对、跑起来稳不稳。做四件事：

  1. 结构：nq / nv / 总质量 / 执行器数量；
  2. 静态：home 位姿下每条闭链两端 site 的距离（应该 ~1e-9 m）；
  3. 自由度：数值求闭链约束雅可比 c(q) 的秩 → 每条腿独立约束数、真实自由度；
     同时打印奇异值，方便看"大/小"之间有没有明显断层；
  4. 动态：跑 T 秒（可以给控制指令），报告闭链最大残差、NaN 次数、
     车轮碰撞圆柱最低点 z（焊死基座应该 ~0 且不穿地）。

用法：
    python3 tools/check_mjcf_loop.py --xml mjcf/rl_chassis_v12_weld_scene.xml
    python3 tools/check_mjcf_loop.py --xml ... --seconds 3 --ctrl 0.6 -0.6
"""

import argparse
import math
import os

import numpy as np
import mujoco

COLLISION_GROUP = 3          # 与 make_mjcf_pins.py 里碰撞体的可视分组一致


def load(path):
    m = mujoco.MjModel.from_xml_path(os.path.abspath(path))
    return m, mujoco.MjData(m)


def pin_pairs(m):
    """[(site1_id, site2_id, name1, name2)]，只取 connect。site 未定义时退化到 body。"""
    pairs = []
    for e in range(m.neq):
        if m.eq_type[e] != mujoco.mjtEq.mjEQ_CONNECT:
            continue
        ids, names = [], []
        for k, oid in ((1, int(m.eq_obj1id[e])), (2, int(m.eq_obj2id[e]))):
            otype = int(m.eq_objtype[e])     # connect 的 obj1/obj2 同类型
            ids.append(oid)
            if otype == mujoco.mjtObj.mjOBJ_SITE:
                names.append(m.site(oid).name)
            elif otype == mujoco.mjtObj.mjOBJ_BODY:
                names.append("body:" + m.body(oid).name)
            else:
                names.append("obj{}".format(k))
        pairs.append((ids[0], ids[1], names[0], names[1]))
    return pairs


def site_gap(d, pairs):
    return np.concatenate([d.site_xpos[s1] - d.site_xpos[s2] for s1, s2, _, _ in pairs])


def constraint_jacobian(m, d, pairs):
    q0 = d.qpos.copy()
    J = np.zeros((3 * len(pairs), m.nq))
    h = 1e-7
    for i in range(m.nq):
        d.qpos[:] = q0
        d.qpos[i] += h
        mujoco.mj_forward(m, d)
        cp = site_gap(d, pairs)
        d.qpos[:] = q0
        d.qpos[i] -= h
        mujoco.mj_forward(m, d)
        J[:, i] = (cp - site_gap(d, pairs)) / (2 * h)
    d.qpos[:] = q0
    mujoco.mj_forward(m, d)
    return J


def joint_index(m):
    """[(qpos 下标, 关节名), ...]，跳过 freejoint。"""
    out = []
    for j in range(m.njnt):
        if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        out.append((int(m.jnt_qposadr[j]), m.joint(j).name))
    return out


def dof_report(m, d, pairs):
    J = constraint_jacobian(m, d, pairs)
    jidx = joint_index(m)
    Jj = J[:, [q for q, _ in jidx]]
    sv = np.linalg.svd(Jj, compute_uv=False)
    rank = int((sv > sv[0] * 1e-6).sum()) if sv.size else 0
    print("  约束雅可比 {}，奇异值: {}".format(
        Jj.shape, " ".join("{:.4g}".format(s) for s in sv[:16])))
    print("  关节自由度 {}，独立约束 {} → 模型自由度 {}".format(
        len(jidx), rank, len(jidx) - rank))
    for tag in ("Left", "Right"):
        idx = [i for i, (_, name) in enumerate(jidx) if name.startswith(tag)]
        if not idx:
            continue
        s = np.linalg.svd(Jj[:, idx], compute_uv=False)
        r = int((s > s[0] * 1e-6).sum()) if s.size else 0
        wheel = [i for i in idx if "wheel" in jidx[i][1].lower()]
        print("  {}: 关节 {} 个（含轮子 {}），约束秩 {} → 自由度 {}".format(
            tag, len(idx), len(wheel), r, len(idx) - r))
    return J


def wheel_lowest(m, d):
    zs = []
    for g in range(m.ngeom):
        if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_CYLINDER:
            continue
        if "wheel" not in m.geom(g).name.lower():
            continue
        axis = d.geom_xmat[g].reshape(3, 3)[:, 2]
        zs.append(float(d.geom_xpos[g][2]
                        - m.geom_size[g][0] * math.sqrt(max(0.0, 1 - axis[2] ** 2))))
    return zs


def render(m, d, path, width=1000, height=760, azimuth=135.0, elevation=-14.0,
           distance=1.5, site_size=6.0):
    """把当前状态渲染成 PNG（相机盯住机器人几何中心）。"""
    import mujoco
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    lo = d.geom_xpos.min(0)
    hi = d.geom_xpos.max(0)
    cam.lookat[:] = 0.5 * (lo + hi)
    cam.azimuth, cam.elevation, cam.distance = azimuth, elevation, distance
    opt = mujoco.MjvOption()
    mujoco.mjv_defaultOption(opt)
    opt.geomgroup[COLLISION_GROUP] = 0        # 凸分解出来的碰撞体不渲染
    opt.sitegroup[0] = 1          # 让 site 显示出来
    opt.sitegroup[1:5] = 0
    width = int(min(width, m.vis.global_.offwidth))
    height = int(min(height, m.vis.global_.offheight))
    r = mujoco.Renderer(m, height, width)
    r.update_scene(d, cam, opt)
    img = r.render()
    try:
        import PIL.Image
        PIL.Image.fromarray(img).save(path)
    except Exception:
        import numpy as np
        np.save(path + ".npy", img)
    print("  已渲染 " + path)


def main():
    ap = argparse.ArgumentParser(description="闭链 MJCF 体检")
    ap.add_argument("--xml", required=True)
    ap.add_argument("--ctrl", type=float, nargs="*", default=None)
    ap.add_argument("--seconds", type=float, default=2.0)
    ap.add_argument("--settle", type=float, default=0.0)
    ap.add_argument("--no-dof", action="store_true")
    ap.add_argument("--png", default=None, help="渲染前缀，如 /tmp/v11 会输出 _home/_end.png")
    args = ap.parse_args()

    m, d = load(args.xml)
    print("== " + args.xml)
    print("  nq={} nv={} nbody={} njnt={} neq={} ngeom={} nu={}".format(
        m.nq, m.nv, m.nbody, m.njnt, m.neq, m.ngeom, m.nu))
    print("  总质量 {:.4f} kg".format(m.body_subtreemass[0]))
    pairs = pin_pairs(m)
    print("  闭链 {} 条: {}".format(
        len(pairs), ", ".join("{}({})<->{}({})".format(n1, s1, n2, s2)
                              for s1, s2, n1, n2 in pairs)))

    if m.nkey:
        mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    print("== 静态（home / 装配位姿 qpos=0）")
    gap = site_gap(d, pairs)
    for i, (_, _, n1, n2) in enumerate(pairs):
        r = float(np.linalg.norm(gap[3 * i:3 * i + 3]))
        print("  {:28s} <-> {:28s} 间距 {:9.3f} µm".format(n1, n2, r * 1e6))
    for z in wheel_lowest(m, d):
        print("  车轮碰撞圆柱最低点 z = {:+.3f} mm".format(z * 1000))
    if not args.no_dof:
        print("== 自由度")
        dof_report(m, d, pairs)
    if args.png:
        render(m, d, args.png + "_home.png")

    print("== 动态（{} s，timestep {}）".format(args.seconds, m.opt.timestep))
    if m.nkey:
        mujoco.mj_resetDataKeyframe(m, d, 0)
    maxres, nan = 0.0, False
    nstep = int(args.seconds / m.opt.timestep)
    for k in range(nstep):
        if args.ctrl and k * m.opt.timestep >= args.settle:
            d.ctrl[:] = np.array(args.ctrl[:m.nu])
        mujoco.mj_step(m, d)
        maxres = max(maxres, float(np.abs(site_gap(d, pairs)).max()))
        if not np.all(np.isfinite(d.qpos)):
            nan = True
            break
    print("  闭链最大残差 {:.4f} mm，NaN: {}".format(maxres * 1000, nan))
    if m.nq > 1 and any(m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
                        for j in range(m.njnt)):
        print("  末态基座 pos = {}".format(np.round(d.qpos[:3], 4)))
    for z in wheel_lowest(m, d):
        print("  末态车轮最低点 z = {:+.3f} mm".format(z * 1000))
    if args.png:
        render(m, d, args.png + "_end.png")
    for j in range(min(m.njnt, 8)):
        if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        print("    {:28s} q={:+.4f}".format(m.joint(j).name, d.qpos[m.jnt_qposadr[j]]))


if __name__ == "__main__":
    main()
