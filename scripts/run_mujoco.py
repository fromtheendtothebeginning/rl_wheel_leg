#!/usr/bin/env python3
"""在 MuJoCo 里跑 rl_chassis 的 MJCF，看机构 / 量闭环残差。

模型有 6 个力矩电机（每腿 髋前 / 髋后 / 轮 各一个），其余关节是闭环平行四边形的被动关节，
由 4 个 ``<connect>`` 约束决定，所以这里用 PD 把 6 个电机稳在 home 位姿。

所有默认值（场景路径、时长、PD 增益、渲染分辨率与机位）都在 ``config/project.ini``。

用法（仓库根的 .venv 里有 mujoco）：
    .venv/bin/python scripts/run_mujoco.py --view
    .venv/bin/python scripts/run_mujoco.py --headless --png out/mujoco.png
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import Config  # noqa: E402


def loop_residual(model, data) -> float:
    """4 个 ``<connect>`` 的当前残差（米）。量的是 site 之间的距离，即闭链有没有被拉开。"""
    worst = 0.0
    for e in range(model.neq):
        a, b = model.eq_obj1id[e], model.eq_obj2id[e]  # site id
        gap = float(((data.site_xpos[a] - data.site_xpos[b]) ** 2).sum() ** 0.5)
        worst = max(worst, gap)
    return worst


def pd_hold(model, data, q_des, kp: float, kd: float) -> list[float]:
    """6 个力矩电机的 PD 力矩，裁到 ctrlrange。"""
    ctrl = []
    for a in range(model.nu):
        joint = model.actuator_trnid[a][0]
        q = data.qpos[model.jnt_qposadr[joint]]
        qd = data.qvel[model.jnt_dofadr[joint]]
        tau = kp * (q_des[joint] - q) - kd * qd
        lo, hi = model.actuator_ctrlrange[a]
        ctrl.append(min(hi, max(lo, tau)))
    return ctrl


def main() -> int:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None, help="配置文件（默认 config/project.ini）")
    known, _ = pre.parse_known_args()
    cfg = Config(known.config)

    ap = argparse.ArgumentParser(
        description=__doc__, parents=[pre], formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--scene", default=str(cfg.path_of("mujoco", "scene")), help="场景 XML")
    ap.add_argument("--view", action="store_true", help="开 MuJoCo viewer 窗口")
    ap.add_argument("--headless", action="store_true", help="不开窗口，跑完打印状态")
    ap.add_argument("--seconds", type=float, default=cfg.float("mujoco", "seconds"))
    ap.add_argument("--kp", type=float, default=cfg.float("mujoco", "kp"))
    ap.add_argument("--kd", type=float, default=cfg.float("mujoco", "kd"))
    ap.add_argument("--wheel", type=float, default=cfg.float("mujoco", "wheel_torque"), help="轮子常值力矩")
    ap.add_argument("--png", default=None, help="渲染一张 PNG 到该路径")
    ap.add_argument("--width", type=int, default=cfg.int("mujoco.render", "width"))
    ap.add_argument("--height", type=int, default=cfg.int("mujoco.render", "height"))
    args = ap.parse_args()

    import mujoco
    import numpy as np

    if not args.view and not args.headless and not args.png:
        args.headless = True

    model = mujoco.MjModel.from_xml_path(args.scene)
    data = mujoco.MjData(model)
    print(f"模型: {args.scene}")
    print(f"  nq={model.nq} nv={model.nv} nu={model.nu} neq={model.neq} 质量={model.body_mass.sum():.4f} kg")

    mujoco.mj_resetDataKeyframe(model, data, 0)  # keyframe "home"
    mujoco.mj_forward(model, data)  # xpos/xmat 要 forward 后才是 keyframe 的位姿
    z0 = float(data.qpos[2])
    torso = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    start = data.xpos[torso].copy() if torso >= 0 else None

    # 目标角度 = home keyframe 里 6 个被驱动关节的角度
    q_des = np.zeros(model.nq)
    for a in range(model.nu):
        joint = model.actuator_trnid[a][0]
        q_des[joint] = data.qpos[model.jnt_qposadr[joint]]
    print(f"  home 基座高度 z={z0:.5f} m，初始闭链残差={loop_residual(model, data) * 1e6:.1f} µm")

    wheels = [
        a
        for a in range(model.nu)
        if "Wheel" in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or "")
    ]

    def step_once() -> None:
        data.ctrl[:] = pd_hold(model, data, q_des, args.kp, args.kd)
        for a in wheels:
            lo, hi = model.actuator_ctrlrange[a]
            data.ctrl[a] = min(hi, max(lo, args.wheel))
        mujoco.mj_step(model, data)

    if args.view:
        import mujoco.viewer

        print("\n打开 viewer（Ctrl+C 或关闭窗口退出）…")
        frame_dt = 1.0 / 60.0
        steps_per_frame = max(1, round(frame_dt / model.opt.timestep))
        next_t = time.perf_counter()
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                for _ in range(steps_per_frame):
                    step_once()
                viewer.sync()
                # 60 Hz 刷新、按仿真时间 1x 节流；落后时重新对齐，不追帧
                next_t += frame_dt
                wait = next_t - time.perf_counter()
                if wait > 0:
                    time.sleep(wait)
                else:
                    next_t = time.perf_counter()
        return 0

    steps = int(args.seconds / model.opt.timestep)
    for _ in range(steps):
        step_once()

    residual = loop_residual(model, data)
    upright = float(data.xmat[torso].reshape(3, 3)[2, 2]) if torso >= 0 else float("nan")
    print(f"\n仿真 {args.seconds:.1f} s（{steps} 步, dt={model.opt.timestep}）后:")
    print(f"  基座高度 z: {z0:.5f} -> {float(data.qpos[2]):.5f} m（落地差 {abs(float(data.qpos[2]) - z0) * 1000:.2f} mm）")
    print(f"  机体 z 轴竖直度(1.0=完全竖直): {upright:.4f}")
    print(f"  闭链残差: {residual * 1e6:.1f} µm")
    if start is not None:
        print(f"  基座位移: {float(np.linalg.norm(data.xpos[torso] - start)) * 1000:.2f} mm")
    if residual > 1e-4:
        print("  ⚠ 闭链残差偏大，机制可能被拉开了")

    if args.png:
        os.environ.setdefault("MUJOCO_GL", "egl")
        out = Path(args.png)
        out.parent.mkdir(parents=True, exist_ok=True)
        renderer = mujoco.Renderer(model, args.height, args.width)
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(model, camera)
        camera.azimuth = cfg.float("mujoco.render", "azimuth")
        camera.elevation = cfg.float("mujoco.render", "elevation")
        camera.distance = cfg.float("mujoco.render", "distance")
        camera.lookat[:] = data.qpos[:3]
        renderer.update_scene(data, camera)
        from PIL import Image

        Image.fromarray(renderer.render()).save(out)
        print(f"  已渲染 -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
