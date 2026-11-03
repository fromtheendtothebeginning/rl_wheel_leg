#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Snap revolute joint axes onto the physical pin bores.

问题
----
sw2urdf 有时会把某个 link 的坐标系放在它的**远端**销孔上，于是该 link 的
joint origin 也跟着落到了远端 —— 关节轴线和真正的销孔差了十几厘米。装配位姿下
看不出来（网格摆位是对的），但一转动关节，两个零件就会在销孔处撕开。
v11 上有 4 处这样：Left/Right_front_child1_joint、Left_front_child3_joint、
Right_front_joint3_joint（错位 115~118 mm，转 0.3 rad 就分开 34~35 mm）。

做法
----
把该关节的 origin 平移 Δ（父系里），同时把**挂在 child 上的一切**反向平移 Δ
（child 自己的 visual / inertial origin，以及 child 的所有子关节 origin），
这样：
  * 关节轴落到销孔轴线上 → 零件再也撕不开；
  * 所有网格 / 惯量 / 其它关节的**世界位姿一点没变**（脚本会校验）。

注意：左右销轴都平行于 base_link 的 y 轴，所以脚本假设 URDF 已经用
tools/align_frame.py 对齐过（所有 joint axis = 0 0 1）—— 这正是
rl_chassis_v12_std.urdf 的状态。

用法：
    python3 tools/snap_pin_axes.py --urdf urdf/rl_chassis_v12_std.urdf --check
    python3 tools/snap_pin_axes.py --urdf urdf/rl_chassis_v12_std.urdf
"""

import argparse
import itertools
import math
import os
import shutil
import sys
import xml.etree.ElementTree as ET

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pin_loops import Urdf, find_bores, rpy_to_mat   # noqa: E402


def _rot(ax, th):
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * (K @ K)


def fk_angles(urt, q):
    """带关节角的 FK：q = {关节名: 弧度}，返回 {link: (R, p)}。"""
    parent_of = {j["child"]: j for j in urt.joints.values()}

    def chain(link):
        out = []
        while link in parent_of:
            j = parent_of[link]
            out.append(j)
            link = j["parent"]
        return out[::-1]

    out = {urt.root_link: (np.eye(3), np.zeros(3))}
    for link in urt.links:
        ch = chain(link)
        if not ch:
            continue
        R, p = np.eye(3), np.zeros(3)
        for j in ch:
            p = p + R @ j["xyz"]                 # xyz 在父系里
            R = R @ rpy_to_mat(*j["rpy"])
            ax = j["axis"] / np.linalg.norm(j["axis"])
            R = R @ _rot(ax, q.get(j["name"], 0.0))
        out[link] = (R, p)
    return out


def to_local(pose, link, world):
    R, p = pose[link]
    return R.T @ (world - p)


def world_of(pose, link, local):
    R, p = pose[link]
    return R @ local + p


def find_joint_pins(urt, verbose=False):
    """返回 [(joint, parent_link, child_link, pin_world, separation_at_0.3rad)]。"""
    pose = urt.fk()
    links = [l for l in urt.links if urt.links[l].get("mesh")]
    bores = {l: find_bores(urt, l, pose) for l in links}
    jmap = {(j["parent"], j["child"]): j for j in urt.joints.values()}
    out = []
    for a, b in itertools.combinations(links, 2):
        if a[0] != b[0]:
            continue
        j = jmap.get((a, b)) or jmap.get((b, a))
        if j is None:
            continue                       # 只看父子关节；闭链销轴交给 connect
        par, chi = j["parent"], j["child"]
        for da in bores[a]:
            for db in bores[b]:
                if math.hypot(da.x - db.x, da.z - db.z) > 0.002:
                    continue
                y = 0.5 * (0.5 * (da.y0 + da.y1) + 0.5 * (db.y0 + db.y1))
                P = np.array([0.5 * (da.x + db.x), y, 0.5 * (da.z + db.z)])
                ua, ub = to_local(pose, par, P), to_local(pose, chi, P)
                sep = max(np.linalg.norm(world_of(fk_angles(urt, {j["name"]: th}), par, ua)
                                         - world_of(fk_angles(urt, {j["name"]: th}), chi, ub))
                          for th in (0.3, -0.3))
                out.append((j, par, chi, P, float(sep)))
    if verbose:
        groups = {}
        for e in out:
            groups.setdefault(e[0]["name"], []).append(e)
        for name, es in groups.items():
            _, par, chi, P, sep = max(es, key=lambda e: e[4])
            extra = "（另有 {} 个同轴孔对，像螺栓孔圈）".format(len(es) - 1) if len(es) > 1 else ""
            print("   {:26s} {:24s} -> {:24s} 销孔({:+.5f},{:+.5f}) "
                  "转 0.3 rad 撕开 {:7.3f} mm{}".format(
                      name, par, chi, P[0], P[2], sep * 1000, extra))
    return out


def plan_edits(urt, joint_pins, tol=0.0005, bolt_circle_min=3):
    """算出需要改的 joint / visual / inertial origin。返回 (edits, skipped)。"""
    origin_pose = urt.fk()
    edits, skipped = [], []
    groups = {}
    for j, par, chi, P, sep in joint_pins:
        groups.setdefault(j["name"], []).append((j, par, chi, P, sep))
    for name, es in groups.items():
        j, par, chi, P, sep = max(es, key=lambda e: e[4])
        if len(es) >= bolt_circle_min:      # 同轴孔对 ≥3 = 螺栓孔圈（轮毂），不是销轴
            skipped.append((name, -1.0))
            continue
        Rp, pp = origin_pose[par]
        Rc, pc = origin_pose[chi]
        target = np.array([P[0], pc[1], P[2]])      # 只挪 (x,z)，y 保持不动
        d_world = target - pc
        d = float(np.linalg.norm(d_world))
        if d < tol:
            skipped.append((j["name"], d))
            continue
        edits.append(dict(joint=j["name"], parent=par, child=chi,
                          d_local_parent=Rp.T @ d_world,      # 父系里的位移
                          d_local_child=Rc.T @ d_world,       # 子系里的反向补偿
                          offset=d, sep=sep))
    return edits, skipped


def apply_edits(path, edits, out_path=None):
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    tree = ET.parse(path, parser=parser)
    root = tree.getroot()
    links = {l.get("name"): l for l in root.findall("link")}
    joints = {j.get("name"): j for j in root.findall("joint")}
    kids = {}
    for j in root.findall("joint"):
        kids.setdefault(j.find("parent").get("link"), []).append(j)

    def shift_origin(elem, delta):
        if elem is None:
            return
        o = elem.find("origin")
        if o is None:
            return
        cur = np.array([float(x) for x in (o.get("xyz") or "0 0 0").split()])
        o.set("xyz", " ".join("%.9g" % v for v in cur + delta))

    for e in edits:
        j = joints[e["joint"]]
        shift_origin(j, e["d_local_parent"])              # 关节轴挪到销孔
        link = links[e["child"]]
        for tag in ("visual", "inertial"):
            shift_origin(link.find(tag), -e["d_local_child"])
        for k in kids.get(e["child"], []):                # 子关节 origin 也跟着补
            if k is not j:
                shift_origin(k, -e["d_local_child"])

    ET.indent(tree, space="  ")
    out = out_path or path
    tree.write(out, encoding="utf-8", xml_declaration=True)
    return out


def main():
    ap = argparse.ArgumentParser(description="把关节轴对齐到销孔（修正 sw2urdf 的 origin 错位）")
    ap.add_argument("--urdf", required=True)
    ap.add_argument("--check", action="store_true", help="只检查，不改文件")
    ap.add_argument("--out", default=None, help="输出到另一个文件（默认原地改，先备份 .bak）")
    args = ap.parse_args()

    urt = Urdf(args.urdf)
    print("== 每个父子关节：轴线 vs 物理销孔")
    jp = find_joint_pins(urt, verbose=True)
    edits, skipped = plan_edits(urt, jp)
    print("\n== 需要修正的关节")
    if not edits:
        print("   没有需要修正的（都已对齐）")
        return
    for e in edits:
        print("   {:26s} 平移 {:7.2f} mm → 销孔 (父系 Δ={})".format(
            e["joint"], e["offset"] * 1000,
            np.round(e["d_local_parent"], 6)))
    if skipped:
        bolt = sorted({n for n, d in skipped if d < 0})
        ok = sorted({n for n, d in skipped if d >= 0})
        if bolt:
            print("   跳过（同轴孔对 ≥3 个，判定为螺栓孔圈不是销轴）: " + ", ".join(bolt))
        if ok:
            print("   已经对齐、无需改动: " + ", ".join(ok))
    if args.check:
        print("\n（--check：没有写文件）")
        return

    if args.out is None:
        shutil.copy2(args.urdf, args.urdf + ".bak")
        print("\n已备份 " + args.urdf + ".bak")
    out = apply_edits(args.urdf, edits, args.out)
    print("已写入 " + out)

    # 校验：几何没被改动 + 4 处销孔传动了
    new = Urdf(out)
    print("\n== 校验")
    jp2 = find_joint_pins(new, verbose=True)
    bolt = set()
    groups = {}
    for j, par, chi, P, sep in jp2:
        groups.setdefault(j["name"], []).append(sep)
    bolt = {n for n, s in groups.items() if len(s) >= 3}
    real = [max(s) for n, s in groups.items() if n not in bolt]
    # 几何必须一点没动
    pose2 = new.fk()
    dgeo = 0.0
    for l in urt.links:
        if not urt.links[l].get("mesh") or l not in pose2:
            continue
        R1, p1 = urt.fk()[l]
        R2, p2 = pose2[l]
        v = np.array([0.0, 0.0, 0.0])
        dgeo = max(dgeo, float(np.linalg.norm((R1 @ v + p1 + R1 @ urt.links[l]["vis_xyz"])
                                             - (R2 @ v + p2 + R2 @ new.links[l]["vis_xyz"]))))
    print("   关节轴对孔后：剩余真实销孔最大撕开 %.4f mm（螺栓孔圈不计：%s）"
          % (max(real) * 1000 if real else 0.0, ", ".join(sorted(bolt)) or "无"))
    print("   网格世界位姿最大变化 %.4f µm" % (dgeo * 1e6))


if __name__ == "__main__":
    main()
