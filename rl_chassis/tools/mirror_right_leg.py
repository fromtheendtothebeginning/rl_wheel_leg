#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把 URDF 里以 "r" 开头的 link 的关节参数，改成其 "l" 开头对应件的精确镜像。

做法不是猜 xyz/rpy 的符号，而是：
  1. 顺序遍历整棵树（保证父级已修正）；
  2. 对每个 r*_Link，取 l*_Link 的位姿 T_L（base 系），求它的镜像
     mirror(T) = (M @ t, M @ R @ M)，M = diag(-1, 1, 1)（绕 x=0 平面反射）；
  3. 用 T_origin = (T_parent)^{-1} @ mirror(T_L) 反解该关节的 xyz/rpy；
  4. 只在与原值差异超过 tol 时才改写，避免无意义的大 diff。

用法:
    python3 tools/mirror_right_leg.py model.urdf -o model_mirrored.urdf
"""

import argparse
import os
import sys

import numpy as np
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from align_frame import (rpy_to_R, R_to_rpy, T_of, pretty_xyz,  # noqa: E402
                           read_stl_tri)

M = np.diag([-1.0, 1.0, 1.0])


def mesh_center(path):
    """网格包围盒中心。比顶点平均稳，不受两个网格三角化密度不同的影响。"""
    tri = read_stl_tri(path).reshape(-1, 3)
    return (tri.min(0) + tri.max(0)) / 2.0


def placement(link_el, urdf_dir, pose):
    """该 link 的网格在世界系里的实际摆放中心 = pose @ visual_origin @ 网格质心。"""
    vis = link_el.find("visual")
    if vis is None:
        return None
    mesh = vis.find("geometry/mesh")
    if mesh is None:
        return None
    o = vis.find("origin")
    V = T_of((o.get("xyz") if o is not None else "0 0 0").split(),
             (o.get("rpy") if o is not None else "0 0 0").split()) if o is not None else np.eye(4)
    path = os.path.join(urdf_dir, "..", "meshes", os.path.basename(mesh.get("filename")))
    if not os.path.exists(path):
        return None
    T = pose @ V
    return T[:3, :3] @ mesh_center(path) + T[:3, 3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("urdf")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--tol", type=float, default=1e-6, help="小于该差异就不改写")
    args = ap.parse_args()
    out = args.out or args.urdf.replace(".urdf", "_mirrored.urdf")

    root = ET.parse(args.urdf).getroot()
    joints = {}
    order = []
    for j in root.findall("joint"):
        o = j.find("origin")
        xyz = [float(v) for v in (o.get("xyz") if o is not None else "0 0 0").split()]
        rpy = [float(v) for v in (o.get("rpy") if o is not None else "0 0 0").split()]
        joints[j.get("name")] = dict(el=j, parent=j.find("parent").get("link"),
                                     child=j.find("child").get("link"), xyz=xyz, rpy=rpy)
        order.append(j.get("name"))

    # 拓扑序
    poses = {"base_link": np.eye(4)}
    pending = list(order)
    seq = []
    while pending:
        progressed = False
        for n in list(pending):
            rec = joints[n]
            if rec["parent"] in poses:
                poses[rec["child"]] = poses[rec["parent"]] @ T_of(rec["xyz"], rec["rpy"])
                seq.append(n)
                pending.remove(n)
                progressed = True
        if not progressed:
            raise RuntimeError("joint 树断开: %s" % pending)

    links = {l.get("name"): l for l in root.findall("link")}
    urdf_dir = os.path.dirname(os.path.abspath(args.urdf))
    changed = []
    for n in seq:                                   # 已按父->子顺序
        rec = joints[n]
        child = rec["child"]
        left = ("l" + child[1:]) if child.startswith("r") else None
        cL = placement(links[left], urdf_dir, poses[left]) if left in poses else None
        cR = placement(links[child], urdf_dir, poses[child])
        if cL is None or cR is None:
            continue
        # 判据只能是**网格的实际摆放**：右件的摆放中心应当是左件的 x 镜像。
        # link 坐标系的原点/朝向在 CAD 里是任意的（右轮的原点就摆在轴向另一处），
        # 所以既不能按原点比，也不能改朝向——只做平移补偿。
        delta_world = M @ cL - cR
        d = float(np.abs(delta_world).max())
        if d <= args.tol:
            continue
        delta_local = poses[rec["parent"]][:3, :3].T @ delta_world
        n_xyz = np.array(rec["xyz"]) + delta_local
        o = rec["el"].find("origin")
        if o is None:
            o = ET.SubElement(rec["el"], "origin")
        o.set("xyz", pretty_xyz(n_xyz))
        changed.append((n, child, np.array(rec["xyz"]).tolist(), n_xyz.tolist(), d))
        poses[child] = poses[rec["parent"]] @ T_of(n_xyz, rec["rpy"])
        for k in range(seq.index(n) + 1, len(seq)):     # 更新下游位姿
            p = joints[seq[k]]["parent"]
            if p in poses:
                poses[joints[seq[k]]["child"]] = poses[p] @ T_of(joints[seq[k]]["xyz"],
                                                                 joints[seq[k]]["rpy"])

    print("== 修正 %d 个摆放不对称的 link（只平移，不动朝向）:" % len(changed))
    for n, child, ox, nx, d in changed:
        print("   %-14s (→%s)  摆放偏差 %.2f mm" % (n, child, d * 1000))
        print("        原 xyz=(%s)  ->  新 xyz=(%s)" % (pretty_xyz(ox), pretty_xyz(nx)))
    ET.indent(root, space="  ")
    with open(out, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="utf-8"?>\n')
        f.write(ET.tostring(root, encoding="unicode"))
        f.write("\n")
    print("== 写出 %s" % out)


if __name__ == "__main__":
    main()
