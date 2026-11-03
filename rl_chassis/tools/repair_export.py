#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修正 sw2urdf 导出中两类常见问题，输出一份可供 align_frame.py 继续处理的 URDF：

1. ``<axis xyz="0 0 0">``：从参考 URDF 的同名关节复制轴向量；
2. 车轮网格按 CAD 全局坐标导出（visual origin = 0，但网格远离 link 原点）：
   把关节原点/child link 原点平移到车轮几何中心，并同步补偿 visual / inertial /
   collision origin，保证 q=0 时所有网格和惯性坐标的世界位姿完全不变。

第 2 步只对名字匹配 ``--recenter-pattern`` 的 link 生效，避免误动正常导出的
腿部零件。平移后关节轴自然穿过轮轴中心，后续 align_frame.py 才能正常做
坐标规范化和惯量重算。

用法：
    python3 tools/repair_export.py \
        --urdf ../rl_chassis_v12/urdf/rl_chassis_v12.urdf \
        --axis-reference ../rl_chassis_v11/urdf/rl_chassis_v11.urdf \
        -o /tmp/rl_chassis_v12_repaired.urdf
"""

import argparse
import copy
import os
import re
import sys
import xml.etree.ElementTree as ET

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pin_loops import Urdf, rpy_to_mat, read_stl  # noqa: E402


def _vec(s, default=(0.0, 0.0, 0.0)):
    if s is None:
        return np.array(default, dtype=float)
    return np.array([float(x) for x in s.split()], dtype=float)


def _fmt(v):
    return " ".join("%.10g" % float(x) for x in v)


def _shift_origin(node, delta):
    if node is None:
        return
    o = node.find("origin")
    if o is None:
        return
    cur = _vec(o.get("xyz"))
    o.set("xyz", _fmt(cur + delta))


def _mesh_center(urt, link):
    """link 坐标系里的网格包围盒中心。"""
    d = urt.links[link]
    path = os.path.normpath(os.path.join(urt.dir, "..", "meshes",
                                         os.path.basename(d["mesh"])))
    tri, _ = read_stl(path)
    bmin, bmax = tri.reshape(-1, 3).min(0), tri.reshape(-1, 3).max(0)
    Rv = rpy_to_mat(*d["vis_rpy"])
    return d["vis_xyz"] + Rv @ (0.5 * (bmin + bmax))


def repair(urdf_in, urdf_out, axis_reference=None, recenter_pattern=r"pom_wheel.*_Link",
           base_mesh=None, copy_inertial=()):
    urt = Urdf(urdf_in)
    pose = urt.fk()
    tree = ET.parse(urdf_in)
    root = tree.getroot()
    links = {l.get("name"): l for l in root.findall("link")}
    joints = {j.get("name"): j for j in root.findall("joint")}

    ref = ET.parse(axis_reference).getroot() if axis_reference else None
    if ref is not None:
        ref_axes = {}
        for j in ref.findall("joint"):
            a = j.find("axis")
            if a is not None and a.get("xyz"):
                ref_axes[j.get("name")] = a.get("xyz")
        fixed_axes = []
        for name, j in joints.items():
            a = j.find("axis")
            if a is None:
                continue
            xyz = _vec(a.get("xyz"))
            if np.linalg.norm(xyz) > 1e-9:
                continue
            if name not in ref_axes:
                raise RuntimeError("joint %s 轴为零，参考 URDF 里也没有同名关节" % name)
            a.set("xyz", ref_axes[name])
            fixed_axes.append((name, ref_axes[name]))
        for name, xyz in fixed_axes:
            print("  补回零轴: %-28s axis=%s" % (name, xyz))

    if copy_inertial:
        if ref is None:
            raise RuntimeError("--copy-inertial 需要同时给 --axis-reference")
        ref_links = {l.get("name"): l for l in ref.findall("link")}
        for name in copy_inertial:
            if name not in links or name not in ref_links:
                raise RuntimeError("--copy-inertial: 找不到 link %s" % name)
            ref_ine = ref_links[name].find("inertial")
            if ref_ine is None:
                raise RuntimeError("参考 URDF 的 %s 没有 inertial" % name)
            old = links[name].find("inertial")
            if old is not None:
                links[name].remove(old)
            links[name].insert(0, copy.deepcopy(ref_ine))
            print("  恢复 link 惯性: %s <- %s" % (name, axis_reference))

    pat = re.compile(recenter_pattern)
    moved = []
    for link, node in links.items():
        if not pat.search(link) or link not in urt.links or not urt.links[link].get("mesh"):
            continue
        parent_joint = next((j for j in urt.joints.values() if j["child"] == link), None)
        if parent_joint is None:
            raise RuntimeError("link %s 没有父关节，无法平移" % link)
        parent = parent_joint["parent"]
        Rc, pc = pose[link]
        Rp, pp = pose[parent]
        center_link = _mesh_center(urt, link)
        center_world = Rc @ center_link + pc
        delta_local = Rc.T @ (pc - center_world)
        joint_el = joints[parent_joint["name"]]
        o = joint_el.find("origin")
        if o is None:
            o = ET.SubElement(joint_el, "origin")
        o.set("xyz", _fmt(Rp.T @ (center_world - pp)))
        for tag in ("visual", "inertial", "collision"):
            for elem in node.findall(tag):
                _shift_origin(elem, delta_local)
        jname = joint_el.get("name")
        if jname.endswith("_Joint"):
            new_name = jname[:-6] + "_joint"
            if new_name not in joints:
                joint_el.set("name", new_name)
                print("  关节名规范化: %s -> %s" % (jname, new_name))
        moved.append((link, parent_joint["name"], center_world, np.linalg.norm(center_world - pc)))

    for link, joint, center, dist in moved:
        print("  车轮坐标移到轮轴: %-20s joint=%-22s centre=%s 平移 %.2f mm"
              % (link, joint, np.round(center, 6), dist * 1000))

    if base_mesh:
        root_link = urt.root_link
        node = links[root_link]
        mesh_name = "package://%s/meshes/%s" % (root.get("name"), os.path.basename(base_mesh))
        n = 0
        for m in node.iter("mesh"):
            m.set("filename", mesh_name)
            n += 1
        if not n:
            raise RuntimeError("base_mesh 指定了，但 %s 没有 mesh" % root_link)
        print("  替换底盘网格: %-20s -> %s" % (root_link, mesh_name))

    ET.indent(tree, space="  ")
    tree.write(urdf_out, encoding="utf-8", xml_declaration=True)
    print("写入 " + urdf_out)
    return fixed_axes, moved


def main():
    ap = argparse.ArgumentParser(description="修复 sw2urdf 零轴与全局坐标车轮导出")
    ap.add_argument("--urdf", required=True)
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--axis-reference", default=None,
                    help="从该 URDF 的同名关节复制零轴（通常是上一版原始导出）")
    ap.add_argument("--recenter-pattern", default=r"pom_wheel.*_Link",
                    help="需要把 link 原点移到网格中心的 link 名正则")
    ap.add_argument("--base-mesh", default=None,
                    help="用这份 STL 替换 base_link 的 visual/collision 网格（"
                         "sw2urdf 有时只导出部分底盘零件）")
    ap.add_argument("--copy-inertial", nargs="*", default=(),
                    help="从参考 URDF 恢复这些 link 的整段 <inertial>")
    args = ap.parse_args()
    repair(args.urdf, args.out, args.axis_reference, args.recenter_pattern,
           args.base_mesh, args.copy_inertial)


if __name__ == "__main__":
    main()
