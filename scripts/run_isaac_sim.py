#!/usr/bin/env python3
"""把转换好的机器人 USD 放进 Isaac Sim：步进物理、渲染一张图。

输入是 ``scripts/convert_robot_usd.py`` 转出来的 USD。MJCF 那条路会把 4 条
``<connect>`` 闭链保留成 ``PhysicsSphericalJoint``、6 个电机保留成 ``MjcActuator``；
URDF 那条路只有运动学树（表达不了闭链），腿会自由摆动。

场景参数（分辨率、灯光、地面尺寸、取景余量）在 ``config/project.ini`` 的 ``[isaac]``。

用法：
    OMNI_KIT_ACCEPT_EULA=YES $ISAAC_VENV/bin/python scripts/run_isaac_sim.py
    ... --steps 0 --png out/装配位姿.png
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
    ap.add_argument("--usd", default=str(cfg.path_of("isaac", "usd")), help="机器人 USD")
    ap.add_argument("--steps", type=int, default=cfg.int("isaac", "steps"), help="物理步进次数；0 = 只渲染装配位姿")
    ap.add_argument("--png", default=str(cfg.path_of("isaac", "png")))
    ap.add_argument("--width", type=int, default=cfg.int("isaac", "width"))
    ap.add_argument("--height", type=int, default=cfg.int("isaac", "height"))
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

    from scene_utils import (  # noqa: PLC0415 - 需要 SimulationApp 先起来
        add_auto_camera,
        add_ground,
        add_lighting,
        closed_loop_residual,
        drop_to_ground,
        find_prim,
        repair_robot_meshes,
        world_bbox,
    )

    omni.usd.get_context().open_stage(args.usd)
    stage = omni.usd.get_context().get_stage()
    root = stage.GetDefaultPrim()
    print(f">>> [1/5] 已加载 {args.usd}")
    print(f"    默认 prim: {root.GetPath()}  类型: {root.GetTypeName()}")

    lift = drop_to_ground(stage, root)
    if lift:
        print(f"    整体抬高 {lift * 1000:.2f} mm（让最低点贴到 z=0）")

    _, size, _ = world_bbox(stage, root)
    print(f"    包围盒尺寸: {size[0] * 1000:.0f} x {size[1] * 1000:.0f} x {size[2] * 1000:.0f} mm")

    add_ground(stage, size_m=cfg.float("isaac", "ground_size"))
    add_lighting(
        stage,
        dome_intensity=cfg.float("isaac", "dome_intensity"),
        key_intensity=cfg.float("isaac", "key_intensity"),
    )
    cam_path = add_auto_camera(
        stage,
        root,
        focal_length=cfg.float("isaac", "focal_length"),
        margin=cfg.float("isaac", "camera_margin"),
    )
    stats = repair_robot_meshes(stage)
    print(
        f">>> [2/5] 本地场景就绪：地面 + 穹顶光 + 平行光 + 相机；"
        f"{stats['visual']} 个视觉 mesh（{stats['normals_fixed']} 个补了法线，"
        f"{stats['failed']} 个失败），{stats['guides_hidden']} 个碰撞体已隐藏"
    )

    world = World()
    world.reset()

    base = find_prim(stage, "base_link")
    if base is None:
        print("    ⚠ 没找到 base_link，跳过位姿统计", flush=True)

    def pose_z() -> float:
        return world_bbox(stage, base)[0][2] if base is not None else float("nan")

    residual, n_loops = closed_loop_residual(stage)
    print(f">>> [3/5] base_link z={pose_z():.4f} m；闭链 {n_loops} 条，残差 {residual * 1e3:.3f} mm")

    for i in range(args.steps):
        world.step(render=True)
        if i % 80 == 0:
            print(f"    step {i:4d}  base z={pose_z():.4f}  闭链残差={closed_loop_residual(stage)[0] * 1e3:.3f} mm")

    print(f">>> [4/5] 物理步进 {args.steps} 步完成，base_link z={pose_z():.4f} m")

    render_product = rep.create.render_product(cam_path, (args.width, args.height))
    annotator = rep.AnnotatorRegistry.get_annotator("rgb")
    annotator.attach(render_product)
    for _ in range(5):
        rep.orchestrator.step()

    frame = np.asarray(annotator.get_data())
    out = Path(args.png)
    out.parent.mkdir(parents=True, exist_ok=True)
    from PIL import Image

    Image.fromarray(frame).save(out)
    print(f">>> [5/5] 渲染成功 rgb{frame.shape} -> {out}")

    import torch

    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        print(f"    显存已用 {(total - free) / 1e9:.2f} GB / 共 {total / 1e9:.2f} GB")

    app.close()
    print(">>> DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
