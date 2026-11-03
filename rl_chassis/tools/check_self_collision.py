#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Self-collision check: does every link really have solid collision?

做两件事：
  1. 装配位姿下的自碰撞体检：列出机器人与自身、以及与地面的接触；
  2. 逐步把某个关节压到很大的角度，看它是不是被"撞住"了 ——
     没有自碰撞时关节能一路转到底（零件互相穿过去），有自碰撞时会被顶住。

用法：
    python3 tools/check_self_collision.py --xml mjcf/rl_chassis_v12_weld.xml
    python3 tools/check_self_collision.py --xml ... --joint Left_front_joint --to -1.2
"""

import argparse
import collections
import math
import os

import numpy as np
import mujoco


def contacts(m, d):
    """返回 {零件对: (接触点数, 最浅/最深距离)}，地面记作 floor。"""
    out = collections.defaultdict(lambda: [0, 0.0])
    for c in range(d.ncon):
        g1, g2 = int(d.contact[c].geom1), int(d.contact[c].geom2)
        b1 = m.body(m.geom_bodyid[g1]).name
        b2 = m.body(m.geom_bodyid[g2]).name
        if m.geom(g1).name == "floor" or m.geom(g2).name == "floor" or "world" in (b1, b2):
            b1 = "floor"
        key = tuple(sorted((b1, b2)))
        out[key][0] += 1
        out[key][1] = min(out[key][1], float(d.contact[c].dist))
    return out


def main():
    ap = argparse.ArgumentParser(description="自碰撞体检 + 关节阻挡测试")
    ap.add_argument("--xml", required=True)
    ap.add_argument("--joint", default="Left_front_joint")
    ap.add_argument("--to", type=float, default=-1.2, help="把该关节压到这个角度")
    ap.add_argument("--seconds", type=float, default=2.0)
    args = ap.parse_args()

    m = mujoco.MjModel.from_xml_path(os.path.abspath(args.xml))
    d = mujoco.MjData(m)
    if m.nkey:
        mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    print(f"== {args.xml}")
    print(f"  ngeom={m.ngeom} nbody={m.nbody} 关节数={m.njnt} 执行器={m.nu}")
    cs = contacts(m, d)
    self_c = {k: v for k, v in cs.items() if "floor" not in k}
    ground = {k: v for k, v in cs.items() if "floor" in k}
    print(f"  装配位姿：自碰撞 {len(self_c)} 对 / 地面接触 {len(ground)} 对")
    for k, v in sorted(self_c.items(), key=lambda kv: kv[1][1]):
        print(f"    {k[0]:26s} <-> {k[1]:26s} 点数 {v[0]:3d} 最深 {v[1]*1000:+.3f} mm")
    for k, v in sorted(ground.items()):
        print(f"    {k[0]:26s} <-> {k[1]:26s} 点数 {v[0]:3d} 最深 {v[1]*1000:+.3f} mm")

    jid = next(j for j in range(m.njnt) if m.joint(j).name == args.joint)
    print(f"== 把 {args.joint} 压到 {args.to:+.2f} rad（PD kp=200）")
    mujoco.mj_resetDataKeyframe(m, d, 0)
    steps = int(args.seconds / m.opt.timestep)
    hit = set()
    maxpen = 0.0
    for k in range(steps):
        for a in range(m.nu):
            j = int(m.actuator_trnid[a][0])
            q, v = d.qpos[m.jnt_qposadr[j]], d.qvel[m.jnt_dofadr[j]]
            goal = args.to if j == jid else 0.0
            d.ctrl[a] = float(np.clip(200 * (goal - q) - 10 * v, *m.actuator_ctrlrange[a]))
        mujoco.mj_step(m, d)
        cs = contacts(m, d)
        for key, v in cs.items():
            if "floor" in key:
                continue
            hit.add(key)
            maxpen = min(maxpen, v[1])
    got = float(d.qpos[m.jnt_qposadr[jid]])
    print(f"  目标 {args.to:+.3f} rad，实际停在 {got:+.3f} rad"
          f"（误差 {math.degrees(got - args.to):+.1f}°）")
    if hit:
        print(f"  过程中发生的自碰撞（最深 {maxpen*1000:+.2f} mm）：")
        for k in sorted(hit):
            print(f"    {k[0]} <-> {k[1]}")
    else:
        print("  没有自碰撞 —— 这个关节方向上没东西挡着")


if __name__ == "__main__":
    main()
