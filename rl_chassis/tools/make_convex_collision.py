#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate per-link convex collision geometry (convex decomposition with coacd).

为什么需要
----------
MuJoCo 的网格碰撞体是**凸包**。这个模型的零件都不凸（凸包体积是真实体积的
1.8~22.7 倍，底盘更是 22.7 倍），所以直接拿网格做自碰撞会到处是伪碰撞 —— 之前
几版干脆把 conaffinity 设成 0（机器人内部不互相碰撞），代价就是腿会穿过底盘。

这里对每个 link 的网格做凸分解，把每一块写成独立 STL（零件自身坐标系，不用
再加偏移），再让 make_mjcf_pins.py 用这些小块当碰撞体、原始网格只做显示，
并给"本来就铰接在一起"的零件对加 <exclude>，这样：
  * 每个 link 都有实体碰撞，腿不会再穿底盘；
  * 关节处（父子 link、销轴连接的两个 link）不会自己跟自己打架。

车轮不做分解：它的碰撞体本来就是解析圆柱（半径/半宽精确）。

用法：
    python3 tools/make_convex_collision.py --urdf urdf/rl_chassis_v12_std.urdf
    # 输出到 <pkg>/meshes/convex/*.stl 和 meshes/convex/collision.json
"""

import argparse
import json
import os
import struct
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pin_loops import Urdf, rpy_to_mat, read_stl   # noqa: E402


def write_binary_stl(path, tris):
    """tris: (n,3,3) - 写成二进制 STL。"""
    n = len(tris)
    with open(path, "wb") as f:
        f.write(b" " * 80)
        f.write(struct.pack("<I", n))
        nrm = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
        ln = np.linalg.norm(nrm, axis=1, keepdims=True)
        nrm = np.divide(nrm, np.where(ln == 0, 1, ln))
        # 每条记录 50 字节 = 法向 12 + 三个顶点 36 + 属性 2（属性必须逐条内嵌）
        rec = np.zeros((n, 50), dtype=np.uint8)
        rec[:, :48] = (np.concatenate([nrm, tris.reshape(n, 9)], axis=1)
                       .astype("<f4").view(np.uint8).reshape(n, 48))
        f.write(rec.tobytes())


def mesh_in_body_frame(urt, link):
    """网格顶点转到该 link 的 body 坐标系。"""
    tri, _ = urt.mesh_world(link, urt.fk())
    R, p = urt.fk()[link]
    v = tri.reshape(-1, 3)
    return ((v - p) @ R).reshape(-1, 3, 3)


def decompose(verts, faces, threshold, max_hulls):
    import coacd
    m = coacd.Mesh(verts.astype(np.float64), faces.astype(np.int32))
    return coacd.run_coacd(m, threshold=threshold, max_convex_hull=max_hulls,
                           preprocess_mode="auto", preprocess_resolution=50,
                           max_ch_vertex=128)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    pkg = os.path.dirname(here)
    ap = argparse.ArgumentParser(description="每个 link 的凸分解碰撞体")
    ap.add_argument("--urdf", default=os.path.join(pkg, "urdf", "rl_chassis_v12_std.urdf"))
    ap.add_argument("--outdir", default=None, help="默认 <urdf 同级>/meshes/convex")
    ap.add_argument("--threshold", type=float, default=0.05)
    ap.add_argument("--max-hulls", type=int, default=32)
    ap.add_argument("--skip-wheel", action="store_true", default=True)
    args = ap.parse_args()

    urt = Urdf(args.urdf)
    outdir = args.outdir or os.path.join(os.path.dirname(urt.dir), "meshes", "convex")
    os.makedirs(outdir, exist_ok=True)

    info = {"threshold": args.threshold, "max_hulls": args.max_hulls, "links": {}}
    print(f"{'link':26s} {'faces':>7s} {'pieces':>7s} {'秒':>7s}  输出")
    for link in urt.links:
        if not urt.links[link].get("mesh"):
            continue
        if args.skip_wheel and "wheel" in link.lower():
            info["links"][link] = {"mode": "cylinder"}
            print(f"{link:26s} {'-':>7s} {'-':>7s} {'-':>7s}  解析圆柱")
            continue
        tri = mesh_in_body_frame(urt, link)
        verts = tri.reshape(-1, 3)
        # 顶点去重 + 面索引（coacd 需要索引面）
        uniq, inv = np.unique(np.round(verts, 9), axis=0, return_inverse=True)
        faces = inv.reshape(-1, 3)
        t0 = time.time()
        parts = decompose(uniq, faces, args.threshold, args.max_hulls)
        dt = time.time() - t0
        files = []
        for k, (pv, pf) in enumerate(parts):
            pv = np.asarray(pv, dtype=np.float64)
            pf = np.asarray(pf, dtype=np.int64)
            name = f"{link}_c{k:02d}.STL"
            write_binary_stl(os.path.join(outdir, name), pv[pf])
            files.append(name)
        info["links"][link] = {"mode": "convex", "files": files}
        print(f"{link:26s} {tri.shape[0]:7d} {len(files):7d} {dt:7.1f}  {len(files)} 个 STL")

    with open(os.path.join(outdir, "collision.json"), "w") as f:
        json.dump(info, f, indent=1, ensure_ascii=False)
    print("\n写入 " + os.path.join(outdir, "collision.json"))


if __name__ == "__main__":
    main()
