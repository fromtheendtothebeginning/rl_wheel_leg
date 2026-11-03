#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把修好坐标系的 URDF 转成闭链正确的 MuJoCo MJCF。

和 tools/make_mjcf.py 的区别：闭链的 site 不再靠最近点/暴力搜索，而是用
tools/pin_loops.py 从 STL 里几何识别销轴轴线（识别出的 (x,z) 与两零件的
圆孔轴线重合到 1e-5 m）。这样：

  * qpos=0（URDF 的 SW 装配位姿）就是精确闭合位姿，connect 残差 ~0；
  * 机构自由度正确（每腿 6 个闭链关节 − 4 个独立约束 = 2 个真实驱动自由度，
    再加轮子 1 个 = 3）；
  * 不会出现多出来的自由度把腿拽死的情况。

生成两种基座：
  --base weld   基座焊死在世界系（和 COD 官方 MJCF 一样），整机抬高到车轮刚好
                触地，用来单独看/调机构；
  --base free   基座带 freejoint，可以拿去训练。

碰撞体两种模式：
  --collision-mode convex（默认）  用 tools/make_convex_collision.py 的 coacd 凸分解块；
  --collision-mode solid          每个 link 用原始网格的凸包当实心碰撞体
                                  （车轮仍解析圆柱）。凸包会填满镂空，装配位姿
                                  互相顶住的零件对由 interfering_pairs 自动排除。

用法：
    python3 tools/make_mjcf_pins.py --urdf ../rl_chassis_v11/urdf/rl_chassis_v11_std.urdf \
        --outdir ../rl_chassis_v11/mjcf --base weld
"""

import argparse
import json
import math
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pin_loops import Urdf, find_pins, rpy_to_mat   # noqa: E402

CONNECT_SOLREF = "0.002 1"
CONNECT_SOLIMP = "0.99 0.99 0.01 0.5 2"

WHEEL_RGBA = "0.79 0.82 0.93 1"
LINK_RGBA = "0.90 0.92 0.93 1"
BASE_RGBA = "0.85 0.85 0.88 1"
COLLISION_RGBA = "0.2 0.85 0.35 0.15"
COLLISION_GROUP = 3
# 碰撞体比 MuJoCo 默认（solref 0.02）硬一些：既压不住得深，又比步长留足余量
COLLISION_SOLREF = "0.008 1"
COLLISION_SOLIMP = "0.95 0.99 0.001"


def quat_from_rpy(rpy):
    """URDF rpy (ZYX 外旋) -> MuJoCo wxyz 四元数。"""
    R = rpy_to_mat(*rpy)
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s,
             (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s,
             (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s,
             (R[1, 2] + R[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s,
             0.25 * s]
    q = np.array(q, dtype=float)
    return -q if q[0] < 0 else q


def fmt(v, n=8):
    return " ".join(f"{float(x):.{n}g}" for x in np.atleast_1d(v))


def is_wheel(link):
    return "wheel" in link.lower()


def inertia_body_frame(d):
    """URDF 的惯性张量是在 inertial 自己的坐标系里给的；MuJoCo 的 fullinertia
    要求写在 body 系（而且不能同时给 quat），所以这里旋转过去。
    返回 [ixx iyy izz ixy ixz iyz]（= MuJoCo fullinertia 的顺序）。"""
    ixx, ixy, ixz, iyy, iyz, izz = d["inertia"]
    I = np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])
    R = rpy_to_mat(*d["inr_rpy"])
    Ib = R @ I @ R.T
    return [Ib[0, 0], Ib[1, 1], Ib[2, 2], Ib[0, 1], Ib[0, 2], Ib[1, 2]]


class Builder:
    def __init__(self, urdf, outdir, name=None, base="weld",
                 solref=CONNECT_SOLREF, solimp=CONNECT_SOLIMP, collision=None,
                 extra_excludes=(), passive=(), collision_mode="convex",
                 solid_skip=()):
        self.urt = Urdf(urdf)
        self.urdf = os.path.abspath(urdf)
        self.outdir = os.path.abspath(outdir)
        self.name = name or os.path.basename(self.urt.robot)
        self.base = base
        self.pins, self.bores, self.pose = find_pins(self.urt)
        self.solref, self.solimp = solref, solimp
        self.timestep = "0.001"
        self.collision_mode = collision_mode
        self.solid_skip = set(solid_skip or ())
        self.collision = self._load_collision(collision)
        self.extra = set(extra_excludes or ())
        self.passive = set(passive or ())
        self.wheel_cyl = {}
        self.nact = 0
        self._measure_wheels()

    def _load_collision(self, collision):
        """读取 tools/make_convex_collision.py 生成的凸分解结果（可选）。

        solid 模式默认不用凸分解：每个 link 的原始网格直接当碰撞体（MuJoCo 取凸包）。
        但 --solid-skip 里的 link 仍用凸分解块（例如底盘必须保留腿坑）。"""
        if self.collision_mode == "solid" and not self.solid_skip:
            self.collision_dir = None
            return {}
        cand = [collision] if collision else []
        cand.append(os.path.join(os.path.dirname(self.urt.dir), "meshes", "convex",
                                 "collision.json"))
        for p in cand:
            if p and os.path.exists(p):
                with open(p) as f:
                    data = json.load(f)
                self.collision_dir = os.path.dirname(os.path.abspath(p))
                return data.get("links", {})
        self.collision_dir = None
        return {}

    def has_convex(self, link):
        d = self.collision.get(link) or {}
        if self.collision_mode == "solid" and link not in self.solid_skip:
            return False
        return d.get("mode") == "convex" and d.get("files")

    def has_solid(self, link):
        """solid 模式：非车轮、且不在 solid_skip 里的 link 用原始网格凸包。"""
        return (self.collision_mode == "solid" and not is_wheel(link)
                and link not in self.solid_skip
                and bool(self.urt.links[link].get("mesh")))

    @staticmethod
    def interfering_pairs(path, depth=0.0002, verbose=True):
        """凸分解是近似的（每块都是凸包，会稍微"发胖"），紧配合的零件在装配位姿
        下会互相顶住。这个函数找出"装配位姿就已经压进去 > depth"的零件对，
        调用方把它们加进 <exclude>（其余自碰撞照常生效）。"""
        import mujoco
        m = mujoco.MjModel.from_xml_path(os.path.abspath(path))
        d = mujoco.MjData(m)
        if m.nkey:
            mujoco.mj_resetDataKeyframe(m, d, 0)
        mujoco.mj_forward(m, d)
        worst = {}
        for c in range(d.ncon):
            if d.contact[c].dist > -depth:
                continue
            b1 = m.body(m.geom_bodyid[d.contact[c].geom1]).name
            b2 = m.body(m.geom_bodyid[d.contact[c].geom2]).name
            k = tuple(sorted((b1, b2)))
            worst[k] = min(worst.get(k, 0.0), d.contact[c].dist)
        if verbose and worst:
            print("  装配位姿下互相顶住的零件对（凸分解近似导致，自动排除）：")
            for k, v in sorted(worst.items(), key=lambda kv: kv[1]):
                print(f"    {k[0]:26s} <-> {k[1]:26s} 最深 {v*1000:+7.2f} mm")
        return set(worst)

    def _measure_wheels(self):
        """车轮碰撞体用解析圆柱（网格轮最低点比轮心低 1.7 mm，会让平衡点偏）。

        圆柱按**关节轴**定义：轴 = 车轮 body 系的 z，圆心只取轴向分量，
        半径取绕轴的最大距离。这样左右轮严格一致（用网格 bbox 会差 0.17 mm）。
        """
        for link, d in self.urt.links.items():
            if not is_wheel(link) or not d.get("mesh"):
                continue
            tri, _ = self.urt.mesh_world(link, self.pose)
            v = tri.reshape(-1, 3)
            R, p = self.pose[link]
            v_link = (v - p) @ R                     # 顶点转回车轮 body 系
            lo, hi = v_link.min(0), v_link.max(0)
            radius = float(np.hypot(v_link[:, 0], v_link[:, 1]).max())
            cz = 0.5 * (lo[2] + hi[2])
            self.wheel_cyl[link] = dict(
                center=np.array([0.0, 0.0, cz]),
                radius=radius,
                half=0.5 * (hi[2] - lo[2]),
            )

    def wheel_raise(self):
        """整机抬高量：让最低车轮的解析圆柱最低点刚好落在 z=0。"""
        lows = []
        for link, w in self.wheel_cyl.items():
            R, p = self.pose[link]
            c = p + R @ w["center"]
            lows.append(c[2] - w["radius"])
        return -min(lows)

    def mesh_path(self, link):
        f = self.urt.links[link]["mesh"]
        abs_p = os.path.normpath(os.path.join(self.urt.dir, f))
        return os.path.relpath(abs_p, self.outdir).replace(os.sep, "/")

    # ------------------------------------------------------------------ XML
    def build(self):
        mj = ET.Element("mujoco", {"model": self.name})
        base_note = ("基座焊死在世界系（body 上没有 joint），整机抬高 "
                     f"{self.wheel_raise():+.6f} m 让车轮最低点正好落在 z=0"
                     if self.base == "weld" else
                     "基座带 freejoint，home keyframe 里基座比地面高 2 mm 生成")
        mj.append(ET.Comment(
            " 由 tools/make_mjcf_pins.py 生成；闭链 site 由 tools/pin_loops.py "
            "从 STL 圆孔几何识别（装配位姿下闭合误差 0）。\n       "
            "坐标系 x 前 / y 左 / z 上，关节轴 = 0 0 1（与 URDF 一致）；"
            f"{base_note}。"))
        ET.SubElement(mj, "compiler", {"angle": "radian", "autolimits": "true"})
        dflt = ET.SubElement(mj, "default")
        ET.SubElement(dflt, "geom", {"contype": "1", "conaffinity": "0"})
        ET.SubElement(dflt, "joint", {"damping": "0.02", "armature": "0.01"})

        asset = ET.SubElement(mj, "asset")
        for link in self.urt.links:
            if self.urt.links[link].get("mesh"):
                ET.SubElement(asset, "mesh", {"name": link, "file": self.mesh_path(link)})
            if self.has_convex(link):
                for fn in self.collision[link]["files"]:
                    name = os.path.splitext(fn)[0]
                    ET.SubElement(asset, "mesh", {
                        "name": name,
                        "file": os.path.relpath(os.path.join(self.collision_dir, fn),
                                                self.outdir).replace(os.sep, "/")})

        h = self.wheel_raise()
        wb = ET.SubElement(mj, "worldbody")
        bpos = [0.0, 0.0, h] if self.base == "weld" else [0.0, 0.0, h + 0.002]
        root = ET.SubElement(wb, "body", {"name": self.urt.root_link, "pos": fmt(bpos)})
        if self.base == "free":
            ET.SubElement(root, "freejoint", {"name": "floating_base"})
        self._link_children(root, self.urt.root_link, is_root=True)

        eq = ET.SubElement(mj, "equality")
        for i, pin in enumerate(self.pins, 1):
            side = "L" if pin.a.lower().startswith("left") else "R"
            ET.SubElement(eq, "connect", {
                "name": f"{side}_loop{i}", "site1": f"pin{i}_A", "site2": f"pin{i}_B",
                "solref": self.solref, "solimp": self.solimp})

        self._actuators(mj)
        # 本来就铰接/销连在一起的零件对：不让它们自碰撞（否则关节处自己跟自己打架）
        if self.collision or self.collision_mode == "solid":
            skip = {(j["parent"], j["child"]) for j in self.urt.joints.values()}
            skip |= {(p.a, p.b) for p in self.pins}
            skip |= self.extra
            contact = ET.SubElement(mj, "contact")
            for a, b in sorted(skip):
                if a in self.urt.links and b in self.urt.links:
                    ET.SubElement(contact, "exclude", {"body1": a, "body2": b})
        # keyframe：焊死基座 qpos = 全 0；free 基座必须把基座位姿也写进 keyframe，
        # 否则 mj_resetDataKeyframe 会把基座拉回原点（车轮就掉到地面下了）。
        njnt = len(self.urt.joints)
        key_qpos = list(np.zeros(njnt))
        if self.base == "free":
            key_qpos = [0.0, 0.0, bpos[2], 1.0, 0.0, 0.0, 0.0] + key_qpos
        ET.SubElement(mj, "keyframe").append(
            ET.Element("key", {"name": "home", "qpos": fmt(key_qpos),
                               "ctrl": fmt(np.zeros(self.nact))}))

        self.tree = ET.ElementTree(mj)
        self.height = bpos[2]
        return self.tree

    def _link_children(self, parent, link, is_root=False):
        d = self.urt.links[link]
        if d.get("mass") is not None:
            I = inertia_body_frame(d)
            ET.SubElement(parent, "inertial", {
                "pos": fmt(d["inr_xyz"]),
                "mass": f"{d['mass']:.9g}",
                "fullinertia": fmt(I, 8)})
        convex = self.has_convex(link)
        solid = self.has_solid(link)
        # 显示网格：有凸分解 / 实心碰撞 / 解析圆柱时只做显示（contype=0）
        ET.SubElement(parent, "geom", {
            "name": f"{link}_vis", "type": "mesh", "mesh": link,
            "pos": fmt(d["vis_xyz"]), "quat": fmt(quat_from_rpy(d["vis_rpy"])),
            "contype": "0" if (convex or solid or is_wheel(link)) else "1",
            "conaffinity": "0",
            "rgba": WHEEL_RGBA if is_wheel(link) else (BASE_RGBA if is_root else LINK_RGBA)})
        if solid:
            # 实心碰撞体 = link 的原始网格（MuJoCo 网格碰撞体取凸包）。
            # 注意：凸包会把镂空/内凹处填满，腿在底盘腹腔里会有伪碰撞；
            # 装配位姿压入的零件对由 interfering_pairs 自动加 <exclude>。
            ET.SubElement(parent, "geom", {
                "name": f"{link}_col", "type": "mesh", "mesh": link,
                "pos": fmt(d["vis_xyz"]), "quat": fmt(quat_from_rpy(d["vis_rpy"])),
                "contype": "1", "conaffinity": "1",
                "group": str(COLLISION_GROUP), "rgba": COLLISION_RGBA,
                "solref": COLLISION_SOLREF, "solimp": COLLISION_SOLIMP,
                "friction": "1 0.005 0.0001"})
        elif convex:
            # 凸分解出来的每一块都参与碰撞（机器人内部自碰撞 + 与地面）
            for fn in self.collision[link]["files"]:
                ET.SubElement(parent, "geom", {
                    "name": os.path.splitext(fn)[0] + "_col", "type": "mesh",
                    "mesh": os.path.splitext(fn)[0], "contype": "1", "conaffinity": "1",
                    "group": str(COLLISION_GROUP), "rgba": COLLISION_RGBA,
                    "solref": COLLISION_SOLREF, "solimp": COLLISION_SOLIMP,
                    "friction": "1 0.005 0.0001"})
        elif is_wheel(link):
            w = self.wheel_cyl[link]
            # 圆柱默认轴 = body 系 z = 车轮关节轴，所以不需要 quat
            ET.SubElement(parent, "geom", {
                "name": f"{link}_col", "type": "cylinder",
                "size": f"{w['radius']:.6g} {w['half']:.6g}",
                "pos": fmt(w["center"]),
                "contype": "1", "conaffinity": "1",
                "group": str(COLLISION_GROUP),
                "solref": COLLISION_SOLREF, "solimp": COLLISION_SOLIMP,
                "friction": "1 0.005 0.0001", "rgba": "0 0 0 0"})
        for i, pin in enumerate(self.pins, 1):
            if pin.a == link:
                ET.SubElement(parent, "site", {"name": f"pin{i}_A",
                                               "pos": fmt(pin.local_a, 11), "size": "0.005",
                                               "rgba": "1 0 0 1"})
                parent.append(ET.Comment(f" 销轴{i}: {pin.a} <-> {pin.b}"))
            if pin.b == link:
                ET.SubElement(parent, "site", {"name": f"pin{i}_B",
                                               "pos": fmt(pin.local_b, 11), "size": "0.005",
                                               "rgba": "1 0 0 1"})
        for j in self.urt.children(link):
            b = ET.SubElement(parent, "body", {
                "name": j["child"], "pos": fmt(j["xyz"]),
                "quat": fmt(quat_from_rpy(j["rpy"]))})
            ax = j["axis"] / np.linalg.norm(j["axis"])
            attrs = {"name": j["name"], "axis": fmt(ax, 8)}
            if is_wheel(j["child"]):
                attrs["damping"] = "0.01"
            else:
                attrs["range"] = fmt([j["lower"], j["upper"]])
                attrs["actuatorfrcrange"] = f"-{j['effort']:.6g} {j['effort']:.6g}"
            ET.SubElement(b, "joint", attrs)
            self._link_children(b, j["child"])

    def _actuators(self, mj):
        act = ET.SubElement(mj, "actuator")
        n = 0
        for j in self.urt.joints.values():
            if j["name"] in self.passive:
                continue
            root_child = j["parent"] == self.urt.root_link
            wheel = is_wheel(j["child"])
            if not (root_child or wheel):
                continue
            ET.SubElement(act, "motor", {
                "name": j["name"] + "_motor", "joint": j["name"], "gear": "1",
                "ctrlrange": "-20 20" if wheel else "-40 40"})
            n += 1
        self.nact = n

    def write(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        ET.indent(self.tree, space="  ")
        self.tree.write(path, encoding="utf-8", xml_declaration=True)
        return path

    def scene(self, robot_file, floor=True):
        mj = ET.Element("mujoco", {"model": self.name + "_scene_" + self.base})
        ET.SubElement(mj, "include", {"file": os.path.basename(robot_file)})
        ET.SubElement(mj, "option", {"timestep": self.timestep})
        vis = ET.SubElement(mj, "visual")
        ET.SubElement(vis, "headlight", {"diffuse": "0.7 0.7 0.7",
                                         "ambient": "0.35 0.35 0.35"})
        ET.SubElement(vis, "global", {"azimuth": "140", "elevation": "-16",
                                      "offwidth": "1280", "offheight": "960"})
        ET.SubElement(vis, "map", {"znear": "0.02"})
        asset = ET.SubElement(mj, "asset")
        ET.SubElement(asset, "texture", {"name": "grid", "type": "2d",
                                         "builtin": "checker", "width": "256",
                                         "height": "256", "rgb1": "0.32 0.34 0.37",
                                         "rgb2": "0.26 0.28 0.31"})
        ET.SubElement(asset, "material", {"name": "grid", "texture": "grid",
                                          "texrepeat": "4 4", "reflectance": "0.05"})
        wb = ET.SubElement(mj, "worldbody")
        ET.SubElement(wb, "light", {"pos": "0 0 2.5", "dir": "0 0 -1",
                                    "directional": "true", "castshadow": "true"})
        if floor:
            ET.SubElement(wb, "geom", {"name": "floor", "type": "plane", "size": "4 4 0.1",
                                       "material": "grid", "contype": "0",
                                       "conaffinity": "1", "condim": "3",
                                       "friction": "1 0.005 0.0001"})
        ET.indent(mj, space="  ")
        return ET.ElementTree(mj)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    pkg = os.path.dirname(here)
    ap = argparse.ArgumentParser(description="URDF -> 闭链正确的 MJCF")
    ap.add_argument("--urdf", default=os.path.join(pkg, "urdf", "rl_chassis_v11_std.urdf"))
    ap.add_argument("--outdir", default=os.path.join(pkg, "mjcf"))
    ap.add_argument("--name", default=None)
    ap.add_argument("--base", choices=["weld", "free"], default="weld")
    ap.add_argument("--solref", default=CONNECT_SOLREF, help="闭链 connect 的 solref")
    ap.add_argument("--solimp", default=CONNECT_SOLIMP, help="闭链 connect 的 solimp")
    ap.add_argument("--timestep", default="0.001", help="scene 里的仿真步长")
    ap.add_argument("--passive", nargs="*", default=(),
                    help="不加电机的关节名（导轮等被动自由度）")
    ap.add_argument("--collision-mode", choices=["convex", "solid"], default="convex",
                    help="convex=每 link 用 coacd 凸分解块（默认）；"
                         "solid=每 link 用原始网格凸包当实心碰撞体（车轮仍解析圆柱）")
    ap.add_argument("--solid-skip", nargs="*", default=(),
                    help="solid 模式下仍用凸分解块的 link（例如 base_link 必须保留"
                         "腿坑，否则单个凸包会把腿吞进去）")
    args = ap.parse_args()

    b = Builder(args.urdf, args.outdir, name=args.name, base=args.base,
                solref=args.solref, solimp=args.solimp, passive=args.passive,
                collision_mode=args.collision_mode, solid_skip=args.solid_skip)
    b.timestep = args.timestep
    b.build()
    # 顺手体检：父子关节的轴线是不是落在物理销孔上（sw2urdf 会把 origin 放错位置）
    try:
        from snap_pin_axes import find_joint_pins, plan_edits
        edits, _ = plan_edits(b.urt, find_joint_pins(b.urt))
    except Exception:
        edits = []
    if edits:
        print("⚠ 有 {} 个关节的轴线不在销孔上（转 0.3 rad 会撕开 {:0.1f}~{:0.1f} mm）："
              .format(len(edits), min(e["sep"] for e in edits) * 1000,
                      max(e["sep"] for e in edits) * 1000))
        print("   " + ", ".join(e["joint"] for e in edits))
        print("   建议先跑：python3 tools/snap_pin_axes.py --urdf " + args.urdf)
    # 两遍：先写一份不带"额外排除"的，量一下哪些零件在装配位姿就互相顶住，再重写
    if b.collision or args.collision_mode == "solid":
        os.makedirs(args.outdir, exist_ok=True)
        tmp = os.path.join(os.path.abspath(args.outdir), "." + b.name + "_probe.xml")
        b.write(tmp)
        try:
            extra = Builder.interfering_pairs(tmp)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
        if extra:
            b = Builder(args.urdf, args.outdir, name=args.name, base=args.base,
                        solref=args.solref, solimp=args.solimp,
                        extra_excludes=extra, passive=args.passive,
                        collision_mode=args.collision_mode,
                        solid_skip=args.solid_skip)
            b.timestep = args.timestep
            b.build()
    suffix = "_weld" if args.base == "weld" else ""
    out = os.path.join(args.outdir, f"{b.name}{suffix}.xml")
    b.write(out)
    b.scene(out).write(os.path.join(args.outdir, f"{b.name}{suffix}_scene.xml"))
    print("写入 " + out)
    print(f"抬升高度 h = {b.height:+.6f} m（车轮最低点落在 z=0）")
    print(f"识别到 {len(b.pins)} 条闭链：")
    for p in b.pins:
        print("   " + p.describe())


if __name__ == "__main__":
    main()
