#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把 rl_chassis 系列 URDF 从 SolidWorks 原始导出坐标系，整体旋转到
infantry_V4 使用的标准坐标系，并把每个 joint 的参数按 V4 的写法重写。

原始坐标系（rl_chassis 导出的 CAD 全局系 Origin_global）:
    X = 左右（l_big_leg_Link 在 +X 一侧）
    Y = -上下（即 +Y 朝地，机器人"上方"是 -Y）
    Z = 前后

目标坐标系（infantry_V4 / ROS 约定）:
    X = 前, Y = 左, Z = 上

整体旋转矩阵 M（把 old 坐标映射到 new 坐标, v_new = M @ v_old）:
    默认（--forward=-z，机器人前方 = CAD 的 -Z，即 l_big_leg_Link 是左脚）:
        M = [[0, 0,-1],
             [1, 0, 0],
             [0,-1, 0]]      即 (x,y,z) -> (-z, x, -y)
    可选（--forward=+z，机器人前方 = CAD 的 +Z）:
        M2 = Rz(180°) @ M，两者只差一个 180° 偏航。

除此之外，每个 link 的坐标系还做一次"规范"重投影 G_L（纯旋转），使得：
  * 左腿三个 link 的坐标系 = Rx(-90°)（相对 base_link），右腿 = Rx(+90°)
    —— 与 infantry_V4 的 lf0/rf0 完全一致；
  * 于是 6 个 joint 的 axis 全部为 (0,0,1)，左右镜像体现在 joint origin 的
    rpy 上（和 V4 一模一样）；
  * joint 正方向的物理含义（在机体坐标系里的转轴）与转换前完全一致，
    不改变任何关节的符号约定。

link 坐标系是任意的"规范"，重投影不会改变任何物理量；脚本最后会用
网格顶点做一次端到端校验。

用法:
    python3 tools/align_frame.py urdf/rl_chassis.urdf
    python3 tools/align_frame.py urdf/rl_chassis_mujoco.urdf
    python3 tools/align_frame.py urdf/rl_chassis.urdf --forward=+z
"""

import argparse
import math
import os
import struct
import xml.etree.ElementTree as ET

import numpy as np


# --------------------------------------------------------------------------
# 旋转基础
# --------------------------------------------------------------------------
def rpy_to_R(rpy):
    r, p, y = [float(v) for v in rpy]
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def R_to_rpy(R):
    """R = Rz(yaw) @ Ry(pitch) @ Rx(roll)  ->  (roll, pitch, yaw)"""
    sp = max(-1.0, min(1.0, -R[2, 0]))
    pitch = math.asin(sp)
    if abs(sp) < 1.0 - 1e-9:
        roll = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:  # gimbal lock
        roll = 0.0
        yaw = math.atan2(-R[0, 1], R[1, 1])
    return np.array([roll, pitch, yaw])


def T_of(xyz, rpy):
    T = np.eye(4)
    T[:3, :3] = rpy_to_R(rpy)
    T[:3, 3] = [float(v) for v in xyz]
    return T


def pretty_xyz(v):
    out = []
    for x in v:
        x = float(x)
        if abs(x) < 1e-7:  # 抹掉浮点噪声（0.1 µm 以下）
            x = 0.0
        out.append("%.8g" % x if x != 0.0 else "0")
    return " ".join(out)


# --------------------------------------------------------------------------
# URDF 读取
# --------------------------------------------------------------------------
def read_urdf(path):
    root = ET.parse(path).getroot()
    seen = {}
    dup = []
    for j in root.findall("joint"):
        n = j.get("name")
        if n in seen:
            dup.append((n, j.find("child").get("link")))
        seen[n] = j
    if dup:
        print("!! 警告：URDF 里有重名 joint（非法，解析器会出错）：")
        for n, child in dup:
            print("     %s  (child=%s)" % (n, child))
    links = {}
    for link in root.findall("link"):
        entry = {"el": link, "joint": None}
        links[link.get("name")] = entry
    joints = []
    for j in root.findall("joint"):
        o = j.find("origin")
        xyz = [0.0, 0.0, 0.0]
        rpy = [0.0, 0.0, 0.0]
        if o is not None:
            if o.get("xyz"):
                xyz = [float(v) for v in o.get("xyz").split()]
            if o.get("rpy"):
                rpy = [float(v) for v in o.get("rpy").split()]
        ax = j.find("axis")
        axis = [1.0, 0.0, 0.0]
        if ax is not None and ax.get("xyz"):
            axis = [float(v) for v in ax.get("xyz").split()]
        rec = {
            "el": j,
            "name": j.get("name"),
            "type": j.get("type"),
            "parent": j.find("parent").get("link"),
            "child": j.find("child").get("link"),
            "xyz": np.array(xyz),
            "rpy": np.array(rpy),
            "axis": np.array(axis),
        }
        joints.append(rec)
        links[rec["child"]]["joint"] = rec
    return root, links, joints


def fk_all(links, joints):
    """每个 link 相对 base_link 的位姿（q=0）"""
    poses = {}
    roots = [n for n, v in links.items() if v["joint"] is None]
    assert len(roots) == 1, "期望唯一的根 link，实际: %s" % roots
    poses[roots[0]] = np.eye(4)
    pending = list(joints)
    while pending:
        progressed = False
        for rec in list(pending):
            if rec["parent"] in poses:
                poses[rec["child"]] = poses[rec["parent"]] @ T_of(rec["xyz"], rec["rpy"])
                pending.remove(rec)
                progressed = True
        if not progressed:
            raise RuntimeError("joint 树存在环或断开: %s" % [r["name"] for r in pending])
    return poses, roots[0]


# --------------------------------------------------------------------------
# 主变换
# --------------------------------------------------------------------------
def fix_duplicate_joint_names(root):
    """重名 joint 改成跟 child link 同名（xxx_Link -> xxx_Joint），这是最常见的笔误。"""
    joints = root.findall("joint")
    groups = {}
    for j in joints:
        groups.setdefault(j.get("name"), []).append(j)

    def stem(j):
        c = (j.find("child").get("link") or "")
        return c.replace("_Link", "_Joint").replace("_link", "_joint")

    used = {j.get("name") for j in joints if len(groups[j.get("name")]) == 1}
    renamed = []
    for n, group in groups.items():
        if len(group) == 1:
            continue
        # 名字与 child link 对得上的那个保留原名，其余改名
        group = sorted(group, key=lambda j: stem(j) != n)
        for j in group[1:]:
            cand = stem(j)
            if not cand or cand in used:
                i = 2
                while "%s_%d" % (n, i) in used:
                    i += 1
                cand = "%s_%d" % (n, i)
            renamed.append((j.get("name"), cand, j.find("child").get("link")))
            j.set("name", cand)
            used.add(cand)
    return renamed


def align(root, links, joints, forward, out_path, keep_signs=False, urdf_path=None,
          fix_inertia=False, robot_name=None, relative_meshes=False,
          fix_dup_joints=False):
    poses, base = fk_all(links, joints)

    if fix_dup_joints:
        for old, new, child in fix_duplicate_joint_names(root):
            print("== 重名 joint 改名: %s -> %s  (child=%s)" % (old, new, child))
    if robot_name:
        root.set("name", robot_name)
    if relative_meshes:
        for m in root.iter("mesh"):
            m.set("filename", "../meshes/" + os.path.basename(m.get("filename")))

    if fix_inertia:
        print("== 用网格几何重算惯量（修正 SolidWorks 漏掉的平行轴项）")
        for name, entry in links.items():
            recompute_inertia(entry["el"], urdf_path or out_path)

    # 整体旋转：old 坐标 -> new 坐标   v_new = M @ v_old
    M_minus_z = np.array([[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    if forward == "-z":
        # 机器人前方 = CAD 的 -Z  ->  l_big_leg_Link（+X 侧）落在 +Y（左脚）
        M = M_minus_z
    else:
        # 机器人前方 = CAD 的 +Z，与上面只差一个 180° 偏航（l_big 变成右脚）
        M = np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]) @ M_minus_z

    Rx = lambda a: np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])

    # 每个 link 的坐标系重投影 G（纯旋转）
    G = {}
    for name, entry in links.items():
        if name == base:
            # 根 link：坐标系整体旋转到新约定
            G[name] = M
            continue
        R_old = poses[name][:3, :3]
        # 该 link 在新坐标系里的横向位置决定它是左腿还是右腿
        y_new = (M @ poses[name][:3, 3])[1]
        R_target = Rx(-math.pi / 2) if y_new > 0 else Rx(+math.pi / 2)
        G[name] = R_target.T @ M @ R_old
        # 数值清理
        G[name] = np.round(G[name], 12)

    # ---- 改写每个 link 的 visual / collision / inertial ----
    for name, entry in links.items():
        g = G[name]
        link_el = entry["el"]
        for tag in ("visual", "collision", "inertial"):
            for node in link_el.findall(tag):
                o = node.find("origin")
                if o is None:
                    o = ET.SubElement(node, "origin")
                    o.set("xyz", "0 0 0")
                    o.set("rpy", "0 0 0")
                xyz = [float(v) for v in (o.get("xyz") or "0 0 0").split()]
                rpy = [float(v) for v in (o.get("rpy") or "0 0 0").split()]
                n_xyz = g @ np.array(xyz)
                n_rpy = R_to_rpy(g @ rpy_to_R(rpy))
                o.set("xyz", pretty_xyz(n_xyz))
                o.set("rpy", pretty_xyz(n_rpy))
        # 注意: <inertia> 的数值表达在 inertial frame 里，inertial frame 随
        # link frame 一起旋转，所以张量分量不用改。

    # ---- 改写每个 joint ----
    def G4(R3):
        T = np.eye(4)
        T[:3, :3] = R3
        return T

    for rec in joints:
        gp = G4(G[rec["parent"]])
        gc = G4(G[rec["child"]])
        T_origin_new = T_of(rec["xyz"], rec["rpy"])
        T_origin_new = gp @ T_origin_new @ gc.T
        o = rec["el"].find("origin")
        if o is None:
            o = ET.SubElement(rec["el"], "origin")
        o.set("xyz", pretty_xyz(T_origin_new[:3, 3]))
        o.set("rpy", pretty_xyz(R_to_rpy(T_origin_new[:3, :3])))
        ax_new = G[rec["child"]] @ rec["axis"]
        ax_new = np.round(ax_new, 10)
        # 源文件里的 rpy 只给了 5~6 位小数，会让轴向量带 ~3e-6 的偏差；
        # 这里吸附到精确的单位轴（角度误差 0.0002° 量级，无物理影响）。
        if abs(abs(ax_new[2]) - 1.0) < 1e-4:
            ax_new = np.array([0.0, 0.0, math.copysign(1.0, ax_new[2])])
        ax_new = np.round(ax_new, 9)
        if not keep_signs:
            # 让 joint 正方向的物理转轴与 infantry_V4 对齐：
            # V4 左腿三个关节的机体轴都是 +Y、右腿都是 -Y（axis 一律写作 (0,0,1)）。
            # rl 原始导出的轴向刚好整体反号，这里翻回来。
            body_axis_old = poses[rec["child"]][:3, :3] @ rec["axis"]
            side = (M @ poses[rec["child"]][:3, 3])[1]
            want = np.array([0.0, 1.0, 0.0]) if side > 0 else np.array([0.0, -1.0, 0.0])
            if float(np.dot(M @ body_axis_old, want)) < 0:
                ax_new = -ax_new
        ax = rec["el"].find("axis")
        if ax is not None:
            ax.set("xyz", pretty_xyz(ax_new))
        rec["new_xyz"] = T_origin_new[:3, 3]
        rec["new_rpy"] = R_to_rpy(T_origin_new[:3, :3])
        rec["new_axis"] = ax_new

    ET.indent(root, space="  ")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="utf-8"?>\n')
        f.write(
            "<!--\n"
            "  由 tools/align_frame.py 自动生成，请勿手改；\n"
            "  原始文件: %s\n"
            "  整体坐标系旋转对标 infantry_V4（x 前 / y 左 / z 上），\n"
            "  坐标映射 (x,y,z)_old -> %s\n"
            "  每个 link 还做了规范重投影：所有 joint axis = (0,0,1)，\n"
            "  左右镜像体现在 joint origin 的 rpy 上（与 infantry_V4 一致）。\n"
            "-->\n"
            % (
                os.path.basename(out_path).replace("_std", ""),
                "(-z, x, -y)" if forward == "-z" else "(z, -x, -y)",
            )
        )
        f.write(ET.tostring(root, encoding="unicode"))
        f.write("\n")

    return poses, M, G


# --------------------------------------------------------------------------
# 校验：用网格顶点比较新旧模型（新模型 = M @ 旧模型）
# --------------------------------------------------------------------------
def read_stl_tri(path, stride=1):
    with open(path, "rb") as f:
        data = f.read()
    n = struct.unpack("<I", data[80:84])[0]
    if 84 + n * 50 != len(data):
        return None
    arr = np.frombuffer(data, dtype=np.uint8, count=n * 50, offset=84).reshape(n, 50)
    tri = arr[:, 12:48].copy().view("<f4").reshape(n, 3, 3).astype(np.float64)
    return tri[::stride]


def mesh_volume_inertia(tri):
    """按四面体积分算网格的 体积 / 质心 / 绕质心的惯量（均匀密度，网格坐标系）。"""
    v0, v1, v2 = tri[:, 0], tri[:, 1], tri[:, 2]
    det = np.einsum("ij,ij->i", v0, np.cross(v1, v2))
    V = det.sum() / 6.0
    if abs(V) < 1e-12:
        return None
    S1 = v0 + v1 + v2
    c = ((det / 24.0)[:, None] * S1).sum(0) / V
    M2 = (np.einsum("ni,nj->nij", v0, v0) + np.einsum("ni,nj->nij", v1, v1)
          + np.einsum("ni,nj->nij", v2, v2) + np.einsum("ni,nj->nij", S1, S1))
    C = ((det / 120.0)[:, None, None] * M2).sum(0) - V * np.outer(c, c)
    return V, c, np.trace(C) * np.eye(3) - C


def recompute_inertia(link_el, urdf_path, verbose=True, rho_max=5000.0, rho_min=200.0):
    """
    用网格几何+URDF 质量重算惯量（绕 URDF 给的质心，均匀密度）。
    SolidWorks 导出装配体时常常漏掉各零件的平行轴项，导致惯量偏小很多；
    这里只改惯量张量，质心位置保持 URDF 原值。
    """
    ine = link_el.find("inertial")
    if ine is None:
        return None
    vis = link_el.find("visual")
    mesh = vis.find("geometry/mesh") if vis is not None else None
    if mesh is None:
        return None
    o = mesh.get("filename")
    path = os.path.join(os.path.dirname(os.path.abspath(urdf_path)), "..", "meshes", os.path.basename(o))
    if not os.path.exists(path):
        return None
    tri = read_stl_tri(path)
    if tri is None:
        return None
    r = mesh_volume_inertia(tri)
    if r is None:
        return None
    V, c_m, I_c = r
    mass = float(ine.find("mass").get("value"))
    # 先修「惯量数据坐标系与 link 坐标系不一致」：SolidWorks 导出时偶尔会把某个
    # link 的质量属性写在另一个取向上（典型症状：CoM 落在网格包围盒之外、
    # 惯量主轴与网格的不一致）。这里只在 CoM 出盒时才动手，遍历 24 个
    # 轴对齐旋转，挑「CoM 回到盒内 且 惯量最接近网格」的那个。
    tri_pts = tri.reshape(-1, 3)
    bmin, bmax = tri_pts.min(0), tri_pts.max(0)
    ine0 = link_el.find("inertial")
    oi0 = ine0.find("origin")
    xyz0 = np.array([float(v) for v in ((oi0.get("xyz") if oi0 is not None else None) or "0 0 0").split()])
    rpy0 = [float(v) for v in ((oi0.get("rpy") if oi0 is not None else None) or "0 0 0").split()]
    pad = 1e-3          # 1 mm：CoM 只要明显跑出网格包围盒就认为坐标系不对
    if np.any(xyz0 < bmin - pad) or np.any(xyz0 > bmax + pad):
        best = None
        for perm in [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]:
            for s in [(1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1),
                      (-1, 1, 1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1)]:
                R = np.zeros((3, 3))
                for i in range(3):
                    R[i, perm[i]] = s[i]
                if np.linalg.det(R) < 0.5:
                    continue
                p = R @ xyz0
                if np.any(p < bmin - pad) or np.any(p > bmax + pad):
                    continue
                it0 = ine0.find("inertia")
                Iold = np.array([[float(it0.get("ixx")), float(it0.get("ixy")), float(it0.get("ixz"))],
                                 [float(it0.get("ixy")), float(it0.get("iyy")), float(it0.get("iyz"))],
                                 [float(it0.get("ixz")), float(it0.get("iyz")), float(it0.get("izz"))]])
                err = float(np.abs(R @ Iold @ R.T - I_c * (mass / abs(V))).max())
                if best is None or err < best[0]:
                    best = (err, R, p)
        if best is not None:
            _, R, p = best
            if verbose:
                print("    %-20s 惯量坐标系被旋转了，自动纠正 (%s) CoM %s -> %s"
                      % (link_el.get("name"),
                         " ".join("%d" % int(round(v)) for v in np.diag(R)),
                         pretty_xyz(xyz0), pretty_xyz(p)))
            oi0.set("xyz", pretty_xyz(p))
            oi0.set("rpy", pretty_xyz(R_to_rpy(R @ rpy_to_R(rpy0))))
            it0 = ine0.find("inertia")
            Iold = np.array([[float(it0.get("ixx")), float(it0.get("ixy")), float(it0.get("ixz"))],
                             [float(it0.get("ixy")), float(it0.get("iyy")), float(it0.get("iyz"))],
                             [float(it0.get("ixz")), float(it0.get("iyz")), float(it0.get("izz"))]])
            Inew = R @ Iold @ R.T
            for k, v in zip(("ixx", "ixy", "ixz", "iyy", "iyz", "izz"),
                            (Inew[0, 0], Inew[0, 1], Inew[0, 2], Inew[1, 1], Inew[1, 2], Inew[2, 2])):
                it0.set(k, "%.10g" % v)
    else:
        it0 = ine0.find("inertia")
    oi = ine.find("origin")
    ci = np.array([float(v) for v in ((oi.get("xyz") if oi is not None else None) or "0 0 0").split()])
    rpy = [float(v) for v in ((oi.get("rpy") if oi is not None else None) or "0 0 0").split()]
    ov = vis.find("origin")
    t_v = np.array([float(v) for v in ((ov.get("xyz") if ov is not None else None) or "0 0 0").split()])
    R_v = rpy_to_R([float(v) for v in ((ov.get("rpy") if ov is not None else None) or "0 0 0").split()])
    # URDF 质心点在网格系里的位置
    rho = mass / abs(V)          # 均匀密度：把体积惯量折算成质量惯量
    if not (rho_min < rho < rho_max):
        # 网格体积与质量对不上（多半是导出时零件被隐藏/网格不封闭），
        # 这种情况按网格重算反而会给出错误惯量，宁可保留原值并报警。
        if verbose:
            print("    %-20s 质量 %6.3f 体积 %.5f -> 密度 %5.0f kg/m^3 不合理，"
                  "跳过惯量重算（请检查网格是否完整）" % (link_el.get("name"), mass, abs(V), abs(rho)))
        return None
    d = R_v.T @ (ci - t_v) - c_m
    I_p = rho * I_c + mass * ((d @ d) * np.eye(3) - np.outer(d, d))
    I_link = R_v @ I_p @ R_v.T
    I_inertial = rpy_to_R(rpy).T @ I_link @ rpy_to_R(rpy)
    it = ine.find("inertia")
    old = [it.get(k) for k in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")]
    vals = [I_inertial[0, 0], I_inertial[0, 1], I_inertial[0, 2],
            I_inertial[1, 1], I_inertial[1, 2], I_inertial[2, 2]]
    if verbose:
        print("    %-20s 质量 %6.3f  体积 %.5f m^3 (密度 %5.0f kg/m^3)" % (
            link_el.get("name"), mass, abs(V), abs(rho)))
        print("       原惯量 (ixx,iyy,izz) = (%s, %s, %s)" % (old[0], old[3], old[5]))
        print("       新惯量 (ixx,iyy,izz) = (%s, %s, %s)   倍数 %s" % (
            vals[0], vals[3], vals[5],
            pretty_xyz([abs(vals[0] / float(old[0])), abs(vals[3] / float(old[3])),
                        abs(vals[5] / float(old[5]))])))
    for k, v in zip(("ixx", "ixy", "ixz", "iyy", "iyz", "izz"), vals):
        it.set(k, "%.10g" % v)
    return vals


def mesh_path_of(link_el, urdf_path):
    v = link_el.find("visual")
    if v is None:
        return None
    m = v.find("geometry/mesh")
    if m is None:
        return None
    fn = m.get("filename")
    base = os.path.basename(fn)
    return os.path.join(os.path.dirname(os.path.abspath(urdf_path)), "..", "meshes", base)


def visual_origin(link_el):
    v = link_el.find("visual")
    o = v.find("origin") if v is not None else None
    if o is None:
        return np.eye(4)
    return T_of((o.get("xyz") or "0 0 0").split(), (o.get("rpy") or "0 0 0").split())


def verify(old_urdf, new_urdf, M):
    """两个文件都重新从磁盘读，避免拿到被改写过的元素对象。"""
    _, o_links, o_joints = read_urdf(old_urdf)
    _, n_links, n_joints = read_urdf(new_urdf)
    o_poses, _ = fk_all(o_links, o_joints)
    n_poses, _ = fk_all(n_links, n_joints)
    worst = 0.0
    for name, entry in o_links.items():
        path = mesh_path_of(entry["el"], old_urdf)
        if path is None or not os.path.exists(path):
            continue
        tri = read_stl_tri(path)
        if tri is None:
            continue
        # 取少量顶点即可
        pts = tri[:: max(1, len(tri) // 400)].reshape(-1, 3)
        T_old = o_poses[name] @ visual_origin(entry["el"])
        T_new = n_poses[name] @ visual_origin(n_links[name]["el"])
        w_old = (T_old[:3, :3] @ pts.T).T + T_old[:3, 3]
        w_new = (T_new[:3, :3] @ pts.T).T + T_new[:3, 3]
        expect = (M @ w_old.T).T
        err = float(np.abs(w_new - expect).max())
        worst = max(worst, err)
        print("    %-22s 顶点最大误差 %.3e m" % (name, err))
    print("  >> 全部 link 最大误差: %.3e m %s" % (worst, "(OK)" if worst < 1e-5 else "(!!! 失败)"))
    return worst


def report(out_path, links, joints):
    poses, base = fk_all(links, joints)
    print("\n  新模型 joint 参数:")
    print("    %-16s %-12s %-28s %-24s %-10s" % ("joint", "type", "origin xyz", "origin rpy", "axis"))
    for rec in joints:
        print(
            "    %-16s %-12s %-28s %-24s %-10s"
            % (
                rec["name"],
                rec["type"],
                pretty_xyz(rec["xyz"]),
                pretty_xyz(rec["rpy"]),
                pretty_xyz(rec["axis"]),
            )
        )
    print("\n  新模型关键点（base_link 系, m）:")
    for name in sorted(links):
        p = poses[name][:3, 3]
        print("    %-22s %s" % (name, pretty_xyz(p)))


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("urdf")
    ap.add_argument("--forward", choices=["-z", "+z"], default="-z",
                    help="机器人前方对应 CAD 的哪个方向（默认 -z，即 l_big_leg 是左脚）")
    ap.add_argument("--keep-signs", action="store_true",
                    help="保留原始 URDF 的关节正方向（axis 会写成 (0,0,-1)），"
                         "默认把关节正方向翻成与 infantry_V4 一致")
    ap.add_argument("--fix-inertia", action="store_true",
                    help="用网格几何+URDF 质量重算各 link 惯量（原导出漏掉了零件的平行轴项）")
    ap.add_argument("--robot-name", default=None, help="改写 <robot name=...>")
    ap.add_argument("--relative-meshes", action="store_true",
                    help="网格路径改成 ../meshes/xxx.STL（Isaac Gym / MuJoCo 可直接加载）")
    ap.add_argument("--fix-dup-joints", action="store_true",
                    help="重名 joint 按 child link 名改名（xxx_Link -> xxx_Joint）")
    ap.add_argument("-o", "--out", default=None)
    args = ap.parse_args()

    out = args.out
    if out is None:
        d, f = os.path.split(args.urdf)
        out = os.path.join(d, f.replace(".urdf", "_std.urdf"))

    print("== 读取 %s" % args.urdf)
    root, links, joints = read_urdf(args.urdf)
    poses, M, G = align(root, links, joints, args.forward, out, args.keep_signs,
                        urdf_path=args.urdf, fix_inertia=args.fix_inertia,
                        robot_name=args.robot_name,
                        relative_meshes=args.relative_meshes,
                        fix_dup_joints=args.fix_dup_joints)
    print("== 写出 %s" % out)

    nroot, nlinks, njoints = read_urdf(out)
    nposes, _ = fk_all(nlinks, njoints)

    print("== 校验：新模型 == M @ 旧模型（用 STL 顶点端到端比对）")
    verify(args.urdf, out, M)
    report(out, nlinks, njoints)


if __name__ == "__main__":
    main()
