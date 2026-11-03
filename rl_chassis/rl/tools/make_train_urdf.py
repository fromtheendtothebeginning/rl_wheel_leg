#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 rl_chassis_v11_std.urdf 生成 Isaac Gym 训练用的「串联腿」URDF。

思路（对标复旦 fudan_rl_wheel_leg）:
  * Isaac Gym 不支持闭链，所以训练模型只保留后链
        base_link -> {Left,Right}_rear_link(大腿) -> _rear_child1_link(小腿) -> _Wheel_link
    关节 = 髋(rear) / 膝(rear_child1) / 轮，共 6 个，几何与惯量**原样照抄** v11，
    因此训练模型的运动学就是真机后链的运动学（轮心位置逐点相同）。
  * 前链（front_Link 那条曲柄连杆）整体删掉：真机上它是驱动膝的输入，部署时用
    leg_model.py 的闭环逆解 + 雅可比把「虚拟髋/膝」映射到「后髋/前髋」。

用法:
  python3 make_train_urdf.py                      # 生成到 ../../urdf/rl_chassis_train.urdf
  python3 make_train_urdf.py --hip-effort 40 --knee-effort 40 --wheel-effort 20
"""

from __future__ import annotations

import argparse
import os
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))  # rl_chassis_v11/
SRC = os.path.join(ROOT, "urdf", "rl_chassis_v12_std.urdf")
DEFAULT_OUT = os.path.join(ROOT, "urdf", "rl_chassis_train.urdf")

KEEP_LINKS = {
    "base_link",
    "Left_rear_link", "Left_rear_child1_link", "Left_Wheel_link",
    "Right_rear_link", "Right_rear_child1_link", "Right_Wheel_link",
}
KEEP_JOINTS = {
    "Left_rear_joint", "Left_rear_child1_joint", "Left_Wheel_joint",
    "Right_rear_joint", "Right_rear_child1_joint", "Right_Wheel_joint",
}
EFFORT = {
    "Left_rear_joint": "hip", "Right_rear_joint": "hip",
    "Left_rear_child1_joint": "knee", "Right_rear_child1_joint": "knee",
    "Left_Wheel_joint": "wheel", "Right_Wheel_joint": "wheel",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--hip-effort", type=float, default=40.0)
    ap.add_argument("--knee-effort", type=float, default=40.0)
    ap.add_argument("--wheel-effort", type=float, default=20.0)
    ap.add_argument("--name", default="rl_chassis")
    args = ap.parse_args()

    effort = {"hip": args.hip_effort, "knee": args.knee_effort, "wheel": args.wheel_effort}
    tree = ET.parse(args.src)
    root = tree.getroot()

    dropped_links = [l.get("name") for l in root.findall("link") if l.get("name") not in KEEP_LINKS]
    # 删掉前链 link / joint / gazebo 等非标标签里的引用
    for tag in ("link", "joint"):
        for el in list(root.findall(tag)):
            nm = el.get("name")
            keep = KEEP_LINKS if tag == "link" else KEEP_JOINTS
            if nm not in keep:
                root.remove(el)
    for el in list(root):
        if el.tag not in ("link", "joint"):
            root.remove(el)
    # 关节限位：sw2urdf 导出的是 continuous ±6.28（等于没限位），这里给 effort 上限
    for j in root.findall("joint"):
        nm = j.get("name")
        j.set("type", "revolute")
        lim = j.find("limit")
        if lim is None:
            lim = ET.SubElement(j, "limit")
        lim.set("lower", "-6.283185307179586")
        lim.set("upper", "6.283185307179586")
        lim.set("effort", f"{effort[EFFORT[nm]]:g}")
        lim.set("velocity", "1000")
    root.set("name", args.name)

    body = ET.tostring(root, encoding="unicode")
    header = (
        "<!-- rl_chassis 训练用串联模型（自动生成，勿手改：改 make_train_urdf.py）\n"
        "     只保留后链 base -> rear_link(大腿) -> rear_child1(小腿) -> Wheel；\n"
        "     前链（曲柄 + child1/2/3 + 两个销）已删除，真机是闭链、Isaac Gym 不支持，\n"
        "     部署时用 rl/tools/leg_model.py 做虚拟腿映射。几何/惯量原样来自 rl_chassis_v12_std.urdf。 -->\n"
    )
    xml = '<?xml version="1.0" encoding="utf-8"?>\n' + header + body + "\n"
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(xml)

    n_links = len(root.findall("link"))
    n_joints = len(root.findall("joint"))
    mass = 0.0
    for l in root.findall("link"):
        m = l.find("inertial/mass")
        if m is not None:
            mass += float(m.get("value"))
    print(f"源文件   : {args.src}")
    print(f"输出     : {args.out}")
    print(f"删除 link: {', '.join(dropped_links)}")
    print(f"保留     : {n_links} link / {n_joints} joint，总质量 {mass:.4f} kg "
          f"(髋/膝/轮力矩上限 {args.hip_effort:g}/{args.knee_effort:g}/{args.wheel_effort:g} N·m)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
