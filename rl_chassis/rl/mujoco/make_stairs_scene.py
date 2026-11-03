#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成一个带台阶的 MuJoCo 场景（用于在闭链模型上试爬台阶）。

台阶参数对齐训练地形：训练里 `pyramid_stairs_terrain(step_width=0.7, step_height=0.05+0.18*d)`
—— 台阶宽（进深）0.7 m，高度随难度 5~21 cm。所以自测时挑 0.7 m 进深 + 7~14 cm 高更贴近分布。

用法:
  python make_stairs_scene.py --step-height 0.07 --step-depth 0.7 --n 6 --out /tmp/stairs.xml
  python onnx_mj_v11.py --xml /tmp/stairs.xml --onnx v11_terrain.onnx --cmd-lin-vel 0.6 --duration 12
"""

from __future__ import annotations

import argparse
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROBOT_XML = os.path.abspath(os.path.join(HERE, "..", "..", "mjcf", "rl_chassis_v12.xml"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step-height", type=float, default=0.07)
    ap.add_argument("--step-depth", type=float, default=0.7)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--start-x", type=float, default=2.0)
    ap.add_argument("--width", type=float, default=2.0)
    ap.add_argument("--out", default="/tmp/v11_stairs.xml")
    ap.add_argument("--friction", default="1 0.005 0.0001",
                    help="MuJoCo 地面/台阶摩擦 (slide torsional rolling)；训练用 PhysX 0.5")
    ap.add_argument("--contact", default='solref="0.02 1"',
                    help='接触参数，硬接触可用 solref="0.002 1" solimp="0.99 0.99 0.001"')
    args = ap.parse_args()

    ground = f'contype="0" conaffinity="1" condim="3" friction="{args.friction}" {args.contact}'
    lines = [
        '<mujoco model="rl_chassis_v12_stairs_gen">',
        f'  <include file="{ROBOT_XML}" />',
        '  <option timestep="0.005" />',
        '  <visual><global azimuth="140" elevation="-16" offwidth="1280" offheight="960" />'
        '<map znear="0.02" /></visual>',
        '  <asset>',
        '    <texture name="grid" type="2d" builtin="checker" width="256" height="256" '
        'rgb1="0.32 0.34 0.37" rgb2="0.26 0.28 0.31" />',
        '    <material name="grid" texture="grid" texrepeat="4 4" reflectance="0.05" />',
        '    <material name="step_mat" rgba="0.85 0.55 0.2 1" />',
        '  </asset>',
        '  <worldbody>',
        '    <light pos="0 0 2.5" dir="0 0 -1" directional="true" castshadow="true" />',
        f'    <geom name="floor" type="plane" size="14 4 0.1" material="grid" {ground} />',
    ]
    # 每级台阶做成从地面长上来的实心块，顶面 = (i+1)*step_height
    for i in range(args.n):
        top = (i + 1) * args.step_height
        cx = args.start_x + (i + 0.5) * args.step_depth
        lines.append(
            f'    <geom name="step{i+1}" type="box" pos="{cx:.4f} 0 {top/2:.4f}" '
            f'size="{args.step_depth/2:.4f} {args.width:.2f} {top/2:.4f}" material="step_mat" {ground} />'
        )
    top = args.n * args.step_height
    lines += [
        f'    <geom name="platform" type="box" pos="{args.start_x + args.n*args.step_depth + 2.5:.4f} 0 {top/2:.4f}" '
        f'size="2.5 {args.width:.2f} {top/2:.4f}" material="step_mat" {ground} />',
        '  </worldbody>',
        '</mujoco>',
    ]
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"写出 {args.out}: {args.n} 级台阶，每级 {args.step_height*100:.0f} cm 高 / "
          f"{args.step_depth:.2f} m 深，总爬升 {top*100:.0f} cm（从 x={args.start_x} 开始）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
