#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v12 sim2sim 启动入口：复用 v11 的 ONNX 策略控制栈，默认加载 v12 闭链 MJCF。

v12 相对 v11 只多了 4 个被动 POM 导轮，主动关节名称、虚拟腿逆解、力矩映射都与
v11 完全一致，所以直接复用 rl_chassis_v11/rl/mujoco/onnx_mj_v11.py。

用法：
    python onnx_mj_v12.py
    python onnx_mj_v12.py --cmd-lin-vel 1.5 --duration 8 --no-viewer
    python onnx_mj_v12.py --onnx ../rl_chassis_v11/rl/mujoco/v11_terrain_tall.onnx
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# 并入 rl_chassis/ 之后：v11 的 sim2sim 就在同目录，v12 的 MJCF 在 ../../mjcf/
V11_MUJOCO = HERE
V12_SCENE = os.path.abspath(os.path.join(HERE, "..", "..", "mjcf", "rl_chassis_v12_scene.xml"))
DEFAULT_ONNX = os.path.join(V11_MUJOCO, "v11_flat_slow.onnx")

sys.path.insert(0, V11_MUJOCO)
from onnx_mj_v11 import V11Sim2Sim, build_parser  # noqa: E402


def main():
    parser = build_parser()
    parser.set_defaults(xml=V12_SCENE, onnx=DEFAULT_ONNX)
    args = parser.parse_args()
    if args.duration is None:
        args.duration = 6.0 if args.no_viewer else 0.0
    return V11Sim2Sim(args).run()


if __name__ == "__main__":
    raise SystemExit(main())
