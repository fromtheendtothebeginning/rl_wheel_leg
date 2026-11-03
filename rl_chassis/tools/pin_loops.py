#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 URDF + STL 里**几何识别**闭链销轴（pin）坐标 —— 不再靠猜最近点。

为什么这么做
------------
闭链的 site 必须落在**销轴轴线**上。之前几版是把 site 放在两个网格的最近点上，
最近点很容易落在平面或倒角上；站点一偏，MuJoCo 的 <connect> 就变成"硬把两个
零件拽到一起"，机构多出来的自由度会把腿锁死或者乱飞。

本模块的做法（对这类 SolidWorks 导出的棱柱零件非常稳）：
  1. 这类零件的法向只有两个方向：平行于销轴（上下大平面）或垂直于销轴（侧壁）。
     丢掉上下大平面，把侧壁投影到垂直于销轴的平面上；
  2. 一个圆孔在投影里就是**一圈彼此相连的侧壁**（它与零件其它部分只在上下平面
     处相连，而上下平面已被丢掉），按共享边做并查集就能把每个孔单独切出来；
  3. 对每个切出来的多边形环做最小二乘圆拟合，圆度（rms/r）达标的才是圆孔；
  4. 两个零件上**位置重合**（同一个销轴）的圆孔配对 → 就是一条闭链。

销轴都平行于 base_link 的 y 轴（这类轮腿的销轴与髋轴平行），所以一个孔用
(x, z) + 半径 + 沿 y 的区间就能完整描述。
"""

import math
import os
import struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np


# ------------------------------------------------------------------ URDF/STL
def rpy_to_mat(r, p, y):
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def read_stl(path):
    """返回 (三角形 (n,3,3), 面法向 (n,3))，单位 m。"""
    with open(path, "rb") as f:
        f.read(80)
        n = struct.unpack("<I", f.read(4))[0]
        blob = f.read(n * 50)
    if len(blob) != n * 50:
        raise ValueError("STL 长度不符: " + path)
    data = np.frombuffer(blob, dtype=np.uint8).reshape(n, 50)
    nrm = data[:, 0:12].copy().view("<f4").reshape(n, 3).astype(np.float64)
    tri = data[:, 12:48].copy().view("<f4").reshape(n, 3, 3).astype(np.float64)
    return tri, nrm


def _vec(s, default=(0.0, 0.0, 0.0)):
    if s is None:
        return np.array(default, dtype=float)
    return np.array([float(x) for x in s.split()], dtype=float)


class Urdf:
    """只取本工具需要的部分：树结构、visual、inertial。"""

    def __init__(self, path):
        self.path = os.path.abspath(path)
        self.dir = os.path.dirname(self.path)
        root = ET.parse(self.path).getroot()
        self.robot = root.get("name") or "robot"
        self.joints = {}
        for j in root.findall("joint"):
            o = j.find("origin")
            ax = j.find("axis")
            lim = j.find("limit")
            self.joints[j.get("name")] = dict(
                name=j.get("name"),
                parent=j.find("parent").get("link"),
                child=j.find("child").get("link"),
                xyz=_vec(o.get("xyz") if o is not None else None),
                rpy=_vec(o.get("rpy") if o is not None else None),
                axis=_vec(ax.get("xyz") if ax is not None else None, default=(0, 0, 1)),
                lower=float(lim.get("lower", -math.pi)) if lim is not None else -math.pi,
                upper=float(lim.get("upper", math.pi)) if lim is not None else math.pi,
                effort=float(lim.get("effort", 40.0)) if lim is not None else 40.0,
            )
        self.links = {}
        for l in root.findall("link"):
            vis = l.find("visual")
            if vis is None:
                continue
            vo = vis.find("origin")
            mesh = vis.find("geometry/mesh")
            d = dict(
                name=l.get("name"),
                mesh=mesh.get("filename") if mesh is not None else None,
                vis_xyz=_vec(vo.get("xyz") if vo is not None else None),
                vis_rpy=_vec(vo.get("rpy") if vo is not None else None),
            )
            inr = l.find("inertial")
            if inr is not None:
                io = inr.find("origin")
                I = inr.find("inertia")
                d.update(
                    mass=float(inr.find("mass").get("value")),
                    inr_xyz=_vec(io.get("xyz") if io is not None else None),
                    inr_rpy=_vec(io.get("rpy") if io is not None else None),
                    inertia=[float(I.get(k)) for k in
                             ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")],
                )
            self.links[l.get("name")] = d
        self.root_link = self._find_root()

    def _find_root(self):
        children = {j["child"] for j in self.joints.values()}
        for name in self.links:
            if name not in children:
                return name
        raise ValueError("URDF 里找不到根 link")

    def children(self, link):
        return [j for j in self.joints.values() if j["parent"] == link]

    def fk(self):
        """qpos=0（= SW 装配位姿）时每个 link 在根 link 系下的位姿 (R, p)。"""
        pose = {self.root_link: (np.eye(3), np.zeros(3))}
        todo = list(self.joints.values())
        while todo:
            rest = []
            for j in todo:
                if j["parent"] in pose:
                    Rp, pp = pose[j["parent"]]
                    pose[j["child"]] = (Rp @ rpy_to_mat(*j["rpy"]), pp + Rp @ j["xyz"])
                else:
                    rest.append(j)
            if len(rest) == len(todo):
                break
            todo = rest
        return pose

    def mesh_world(self, link, pose):
        """把该 link 的网格顶点/法向变换到根 link 系（qpos=0）。"""
        v = self.links[link]
        R, p = pose[link]
        path = os.path.normpath(os.path.join(self.dir, v["mesh"]))
        tri, nrm = read_stl(path)
        Rm = R @ rpy_to_mat(*v["vis_rpy"])
        pm = p + R @ v["vis_xyz"]
        v3 = tri.reshape(-1, 3) @ Rm.T + pm
        n3 = nrm @ Rm.T
        return v3.reshape(-1, 3, 3), n3


# ------------------------------------------------------------------- 圆孔识别
@dataclass
class Bore:
    link: str
    x: float
    z: float
    r: float
    rms: float
    n: int
    y0: float
    y1: float


def _profile_loops(urt, link, pose, max_ny=0.25):
    tri, nrm = urt.mesh_world(link, pose)
    wall = np.where(np.abs(nrm[:, 1]) < max_ny)[0]
    parent = list(range(len(tri)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    owner = {}
    for i in wall:
        for a, b in ((0, 1), (1, 2), (2, 0)):
            ka, kb = tuple(np.round(tri[i, a], 6)), tuple(np.round(tri[i, b], 6))
            key = (min(ka, kb), max(ka, kb))
            if key in owner:
                union(owner[key], i)
            else:
                owner[key] = i
    groups = {}
    for i in wall:
        groups.setdefault(find(i), []).append(i)
    for g in groups.values():
        if len(g) < 4:
            continue
        v3 = tri[g].reshape(-1, 3)
        p2 = np.unique(np.round(v3[:, [0, 2]], 7), axis=0)
        if len(p2) >= 5:
            yield p2, v3


def _fit_circle(p2):
    A = np.concatenate([p2, np.ones((len(p2), 1))], 1)
    b = (p2 ** 2).sum(1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    c = sol[:2] / 2
    r = math.sqrt(max(sol[2] + float((c ** 2).sum()), 0.0))
    res = np.abs(np.linalg.norm(p2 - c, axis=1) - r)
    return c, r, float(np.sqrt((res ** 2).mean()))


def find_bores(urt, link, pose, rmin=0.0015, rmax=0.02, rms_max=0.06,
               cover_min=300.0):
    tri, nrm = urt.mesh_world(link, pose)
    wall = tri[np.abs(nrm[:, 1]) < 0.25]
    wall_y = wall.reshape(-1, 3)[:, 1]
    wall_xz = wall.reshape(-1, 3)[:, [0, 2]]
    out = []
    for p2, v3 in _profile_loops(urt, link, pose):
        c, r, rms = _fit_circle(p2)
        if not (rmin <= r <= rmax) or rms / max(r, 1e-9) > rms_max:
            continue
        ang = np.sort(np.arctan2(p2[:, 1] - c[1], p2[:, 0] - c[0]))
        gaps = np.diff(np.concatenate([ang, ang[:1] + 2 * math.pi]))
        if 360 - math.degrees(gaps.max()) < cover_min:
            continue
        # 这一圈孔壁自己的 y 区间（不用"轴心附近所有侧壁"，那会把邻近的孔也算进来）
        y0, y1 = float(v3[:, 1].min()), float(v3[:, 1].max())
        out.append(Bore(link, float(c[0]), float(c[1]), r, rms, len(p2),
                        y0, y1))
    out.sort(key=lambda d: -d.n)
    uniq = []
    for o in out:
        if all(math.hypot(o.x - u.x, o.z - u.z) > 0.0015 for u in uniq):
            uniq.append(o)
    return uniq


@dataclass
class Pin:
    """一条闭链销轴：两个零件 + 世界（根 link 系）坐标 + 各自局部坐标。"""
    a: str
    b: str
    world: np.ndarray
    axis: np.ndarray
    local_a: np.ndarray
    local_b: np.ndarray
    r_a: float
    r_b: float

    def describe(self):
        return (f"{self.a} r={self.r_a*1000:5.2f}mm <-> {self.b} r={self.r_b*1000:5.2f}mm "
                f"轴线@(x={self.world[0]:+.5f}, z={self.world[2]:+.5f}) axis={self.axis}")


def _to_local(pose, world):
    R, p = pose
    return R.T @ (world - p)


def find_pins(urt, tol_xz=0.002, max_y_span=0.04, same_side_only=True, links=None,
              exclude_parent_child=True):
    """扫描所有 link 对，找 (x,z) 重合的圆孔 → 闭链销轴。"""
    pose = urt.fk()
    links = links or [l for l in urt.links if urt.links[l]["mesh"]]
    bores = {l: find_bores(urt, l, pose) for l in links}
    pc = {(j["parent"], j["child"]) for j in urt.joints.values()}
    pc |= {(b, a) for a, b in pc}

    def side(name):
        return "L" if name.lower().startswith(("left", "l_")) else \
               "R" if name.lower().startswith(("right", "r_")) else "-"

    pins = []
    for i, a in enumerate(links):
        for b in links[i + 1:]:
            if exclude_parent_child and (a, b) in pc:
                continue
            if same_side_only and side(a) != side(b):
                continue
            for da in bores[a]:
                for db in bores[b]:
                    if math.hypot(da.x - db.x, da.z - db.z) > tol_xz:
                        continue
                    y_lo, y_hi = min(da.y0, db.y0), max(da.y1, db.y1)
                    if y_hi - y_lo > max_y_span:
                        continue
                    # 销轴取在两零件孔中心的中间：左右腿严格镜像，且装配位姿下
                    # connect 的 y 分量残差恒等于 0（这个机构所有关节轴都平行于 y）
                    y = 0.5 * (0.5 * (da.y0 + da.y1) + 0.5 * (db.y0 + db.y1))
                    world = np.array([0.5 * (da.x + db.x), y, 0.5 * (da.z + db.z)])
                    pins.append(Pin(a, b, world, np.array([0.0, 1.0, 0.0]),
                                    _to_local(pose[a], world), _to_local(pose[b], world),
                                    da.r, db.r))
    pins.sort(key=lambda p: (p.a, p.b))
    return pins, bores, pose


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="几何识别 URDF 闭链销轴（site 坐标）")
    ap.add_argument("--urdf", required=True)
    ap.add_argument("--show-bores", action="store_true", help="顺便打印每个 link 的所有圆孔")
    ap.add_argument("--verify", action="store_true",
                    help="校验 site 是否落在两个零件各自的孔里（到最近网格顶点的距离）")
    args = ap.parse_args()

    urt = Urdf(args.urdf)
    pins, bores, pose = find_pins(urt)
    if args.show_bores:
        for l, bs in bores.items():
            print("--- " + l)
            for d in bs:
                print(f"    (x={d.x:+.5f}, z={d.z:+.5f}) r={d.r*1000:6.2f}mm "
                      f"rms={d.rms*1000:4.2f} y=[{d.y0:+.4f},{d.y1:+.4f}]")
    print(f"\n共识别到 {len(pins)} 条闭链销轴：")
    for p in pins:
        print("  " + p.describe())
        print(f"      site@{p.a:26s} local={np.round(p.local_a, 6)}")
        print(f"      site@{p.b:26s} local={np.round(p.local_b, 6)}")
    if args.verify:
        print("\n== site 到两侧零件网格最近顶点的距离（≈孔半径说明 site 就在孔心）")
        for p in pins:
            for link, r_expected in ((p.a, p.r_a), (p.b, p.r_b)):
                tri, _ = urt.mesh_world(link, pose)
                v = tri.reshape(-1, 3)
                dmin = float(np.min(np.linalg.norm(v - p.world, axis=1)))
                print(f"  {link:26s} 最近顶点 {dmin*1000:6.2f} mm   "
                      f"（该孔半径 {r_expected*1000:5.2f} mm）")
