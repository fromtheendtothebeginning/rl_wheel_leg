#!/usr/bin/env python3
"""检查本机的 Isaac Sim / Isaac Lab 环境是否可用。

查四件事：驱动、torch 的 CUDA 架构里有没有 sm_120、Isaac Sim 版本、RTX 渲染能不能出帧。
Isaac 环境路径取 ``config/project.ini`` 的 ``[paths] isaac_venv``（可用 --isaac-venv 覆盖）。

用法：
    python3 scripts/check_env.py
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import Config  # noqa: E402

SM121_HINT = "没有 ← 训练会报 no kernel image"

PROBE = r'''
import importlib.metadata as md
import torch

print(f"torch   {torch.__version__}")
arch = torch.cuda.get_arch_list()
print(f"arch    {arch}")
print(f"sm_120  {'有' if any('120' in a for a in arch) else '%s'}")
for pkg in ("isaacsim", "isaaclab"):
    try:
        print(f"{pkg:8s} {md.version(pkg)}")
    except Exception:
        print(f"{pkg:8s} 未安装")
''' % SM121_HINT


def main() -> int:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None, help="配置文件（默认 config/project.ini）")
    known, _ = pre.parse_known_args()
    cfg = Config(known.config)

    ap = argparse.ArgumentParser(
        description=__doc__, parents=[pre], formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--isaac-venv", default=None, help="覆盖 [paths] isaac_venv")
    args = ap.parse_args()

    venv = Path(os.path.expandvars(os.path.expanduser(args.isaac_venv or cfg.get("paths", "isaac_venv"))))
    python = venv / "bin" / "python"

    print("===== 1. 驱动 =====")
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv"],
            capture_output=True, text=True, check=True,
        )
        print(out.stdout.strip())
    except Exception as exc:  # noqa: BLE001
        print(f"nvidia-smi 不可用: {exc}")
    print()
    print("驱动分支要和 Isaac Sim 版本配套，否则 RTX 渲染器可能直接崩或静默出空帧：")
    print("  Isaac Sim 6.2  → 驱动 >= 595.58.03")
    print("  Isaac Sim 5.1  → 驱动 580.x")
    print()

    print("===== 2. Isaac Sim / torch =====")
    if not python.exists():
        print(f"找不到 {python}", file=sys.stderr)
        print("设置 config/project.ini 的 [paths] isaac_venv，或用 --isaac-venv", file=sys.stderr)
        return 1
    subprocess.run([str(python), "-c", PROBE], check=False)
    print()

    print("===== 3. RTX 渲染冒烟 =====")
    rc = subprocess.run(
        [str(python), str(Path(__file__).resolve().parent / "smoke_isaac_sim.py")],
        env={**os.environ, "OMNI_KIT_ACCEPT_EULA": "YES"},
        check=False,
    ).returncode
    print()
    if rc == 0:
        print(">>> 环境可用。")
    else:
        print(">>> 渲染没出帧，看上面的日志。常见原因：驱动版本和 Isaac Sim 版本不配套。")
    return rc


if __name__ == "__main__":
    sys.exit(main())
