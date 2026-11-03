#!/usr/bin/env python3
"""把 rl_chassis 的 MJCF / URDF 转成 Isaac Sim 能吃的 USD。

两个转换器随 Isaac Sim 一起装（注意可执行文件名是下划线）：
``mujoco_usd_converter`` / ``urdf_usd_converter``。路径与输出目录来自
``config/project.ini``；``--isaac-venv`` 可覆盖 ``[paths] isaac_venv``。

两条路的区别：MJCF 保留 4 条闭链（``PhysicsSphericalJoint``）和 6 个电机；
URDF 只有运动学树（URDF 表达不了闭链），腿会自由摆动。

用法：
    $ISAAC_VENV/bin/python scripts/convert_robot_usd.py
    ... --out /tmp/usd
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import Config  # noqa: E402


def main() -> int:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None, help="配置文件（默认 config/project.ini）")
    known, _ = pre.parse_known_args()
    cfg = Config(known.config)

    ap = argparse.ArgumentParser(
        description=__doc__, parents=[pre], formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", default=str(cfg.out_dir()), help="输出目录")
    ap.add_argument("--isaac-venv", default=None, help="装了 Isaac Sim 的 venv（默认取 [paths] isaac_venv）")
    args = ap.parse_args()

    robot = cfg.robot_dir()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    venv = Path(
        os.path.expandvars(
            os.path.expanduser(args.isaac_venv or cfg.get("paths", "isaac_venv"))
        )
    )
    converters = {
        "mjcf": venv / "bin" / "mujoco_usd_converter",
        "urdf": venv / "bin" / "urdf_usd_converter",
    }
    missing = [str(p) for p in converters.values() if not p.exists()]
    if missing:
        print("找不到转换器：" + ", ".join(missing), file=sys.stderr)
        print("用 --isaac-venv 指向装了 Isaac Sim 的环境（例如 ~/IsaacLab-3.0/.venv）", file=sys.stderr)
        return 1

    jobs = [
        ("MJCF weld（基座焊死，保留 4 条闭链）", converters["mjcf"], robot / "mjcf" / "rl_chassis_v12_weld.xml", out / "rl_chassis_v12_mjcf_weld"),
        ("MJCF free（基座带 freejoint）", converters["mjcf"], robot / "mjcf" / "rl_chassis_v12.xml", out / "rl_chassis_v12_mjcf_free"),
        ("URDF（_std，只有运动学树）", converters["urdf"], robot / "urdf" / "rl_chassis_v12_std.urdf", out / "rl_chassis_v12_urdf"),
    ]

    for label, exe, src, dst in jobs:
        if not src.exists():
            print(f"跳过 {label}：找不到 {src}", file=sys.stderr)
            continue
        print(f"== {label} → {dst}")
        subprocess.run([str(exe), str(src), str(dst)], check=True)

    print("\n产出（文件名取的是 model name，不是输入文件名）：")
    for usda in sorted(out.glob("*/*.usda")):
        print(f"  {usda}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
