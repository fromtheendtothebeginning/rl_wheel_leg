#!/usr/bin/env python3
"""Isaac Sim 最小冒烟测试：启动 Kit -> 一个立方体自由落体 -> RTX 渲染一帧。

用来判断"渲染器到底工不工作"。画面全黑、``rgb shape=(0,)``、或显存只占 0.4 GB
都说明 RTX 没真正出图 —— 最常见的原因是驱动版本和 Isaac Sim 版本不配套
（见 docs/isaac_sim_setup.md）。

参数在 ``config/project.ini`` 的 ``[isaac.smoke]``。

用法：
    OMNI_KIT_ACCEPT_EULA=YES $ISAAC_VENV/bin/python scripts/smoke_isaac_sim.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

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
    ap.add_argument("--png", default=str(cfg.path_of("isaac.smoke", "png")))
    ap.add_argument("--steps", type=int, default=cfg.int("isaac.smoke", "steps"))
    args = ap.parse_args()

    # 别把我们自己的参数继续喂给 Kit
    sys.argv = sys.argv[:1]

    from isaacsim import SimulationApp

    print(">>> 启动 SimulationApp（headless）", flush=True)
    app = SimulationApp({"headless": True})

    import numpy as np
    import omni.replicator.core as rep
    import omni.usd
    from isaacsim.core.api import World
    from isaacsim.core.api.objects import DynamicCuboid
    from pxr import UsdGeom

    from scene_utils import add_auto_camera, add_ground, add_lighting  # noqa: PLC0415

    cube_size = cfg.float("isaac.smoke", "cube_size")
    stage = omni.usd.get_context().get_stage()
    add_ground(stage, size_m=cfg.float("isaac", "ground_size"))
    add_lighting(
        stage,
        dome_intensity=cfg.float("isaac", "dome_intensity"),
        key_intensity=cfg.float("isaac", "key_intensity"),
    )

    world = World()
    world.scene.add(
        DynamicCuboid(
            prim_path="/World/cube",
            name="cube",
            position=np.array([0.0, 0.0, cfg.float("isaac.smoke", "cube_z")]),
            size=cube_size,
        )
    )
    world.reset()

    cube_prim = stage.GetPrimAtPath("/World/cube")
    cam_path = add_auto_camera(stage, cube_prim, margin=cfg.float("isaac.smoke", "camera_margin"))
    print(f">>> [1/4] 场景就绪：地面 + 立方体（z={cfg.float('isaac.smoke', 'cube_z')} 自由落体）+ 相机 {cam_path}", flush=True)

    zs: list[float] = []
    for i in range(args.steps):
        world.step(render=True)
        zs.append(float(UsdGeom.Xformable(cube_prim).ComputeLocalToWorldTransform(0).ExtractTranslation()[2]))
        if i % 50 == 0:
            print(f"    t={i * world.get_physics_dt():.2f}s  z={zs[-1]:.3f}", flush=True)

    print(f">>> [2/4] 物理步进完成，z: {zs[0]:.3f} -> {zs[-1]:.3f}（落地后应≈{cube_size / 2:.2f}）", flush=True)

    render_product = rep.create.render_product(
        cam_path, (cfg.int("isaac.smoke", "width"), cfg.int("isaac.smoke", "height"))
    )
    annotator = rep.AnnotatorRegistry.get_annotator("rgb")
    annotator.attach(render_product)
    for _ in range(5):
        rep.orchestrator.step()

    frame = np.asarray(annotator.get_data())
    print(f">>> [3/4] RTX 渲染帧 shape={frame.shape}（(0,) 或全黑 = 渲染没工作）", flush=True)

    out = Path(args.png)
    out.parent.mkdir(parents=True, exist_ok=True)
    if frame.size:
        from PIL import Image

        Image.fromarray(frame).save(out)
        print(f"    已存 {out}", flush=True)

    import torch

    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        print(f">>> [4/4] 显存已用 {(total - free) / 1e9:.2f} GB / 共 {total / 1e9:.2f} GB", flush=True)

    app.close()
    print(">>> SMOKE OK" if frame.size else ">>> SMOKE FAILED: 渲染没有输出")
    return 0 if frame.size else 1


if __name__ == "__main__":
    sys.exit(main())
