#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在 Isaac Gym 里试载训练 URDF，打印关节顺序 / 质量 / 力矩上限 / 几何数量。

关键目的：确认 Isaac 导入后的 dof 顺序（obs 里 dof_pos[[0,1,3,4]] 的下标依赖它）
以及 meshes 是否能解析。

用法（注意 LD_LIBRARY_PATH）:
  export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
  python check_isaac_asset.py --urdf ../../urdf/rl_chassis_train.urdf
"""

from __future__ import annotations

import argparse
import os

from isaacgym import gymapi  # noqa: E402  (必须先于 torch)
import numpy as np  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--urdf", default=os.path.join(here, "..", "..", "urdf", "rl_chassis_train.urdf"))
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    urdf = os.path.abspath(args.urdf)
    root, fname = os.path.split(urdf)
    print(f"加载 {urdf}")

    gym = gymapi.acquire_gym()
    params = gymapi.SimParams()
    params.dt = 0.005
    params.substeps = 1
    params.up_axis = gymapi.UP_AXIS_Z
    params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    params.physx.use_gpu = not args.cpu
    params.physx.solver_type = 1
    params.physx.num_position_iterations = 4
    params.physx.num_velocity_iterations = 0
    params.physx.contact_offset = 0.01
    params.physx.rest_offset = 0.0
    params.use_gpu_pipeline = not args.cpu
    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, params)

    opt = gymapi.AssetOptions()
    opt.default_dof_drive_mode = 3      # effort
    opt.collapse_fixed_joints = True
    opt.replace_cylinder_with_capsule = True
    opt.flip_visual_attachments = False
    opt.fix_base_link = False
    opt.density = 0.001
    opt.thickness = 0.01
    opt.angular_damping = 0.0
    opt.linear_damping = 0.0
    opt.max_angular_velocity = 1000.0
    opt.max_linear_velocity = 1000.0
    opt.armature = 0.0
    asset = gym.load_asset(sim, root, fname, opt)

    dofs = gym.get_asset_dof_names(asset)
    props = gym.get_asset_dof_properties(asset)
    bodies = gym.get_asset_rigid_body_names(asset)
    shapes = gym.get_asset_rigid_shape_properties(asset)
    env = gym.create_env(sim, gymapi.Vec3(-1.0, -1.0, -1.0), gymapi.Vec3(1.0, 1.0, 1.0), 1)
    actor = gym.create_actor(env, asset, gymapi.Transform(), "v11_train", 0, 0)
    rb_props = gym.get_actor_rigid_body_properties(env, actor)
    fields = props.dtype.names
    print(f"\nDOF 顺序 ({len(dofs)}):")
    for i, n in enumerate(dofs):
        info = " ".join(f"{f}={props[f][i]:.4g}" for f in
                        ("lower", "upper", "effort", "velocity", "friction", "damping", "armature")
                        if f in fields)
        print(f"  [{i}] {n:26s} {info}")
    total = sum(p.mass for p in rb_props)
    print(f"\n刚体 ({len(bodies)}), 总质量 {total:.4f} kg:")
    for i, n in enumerate(bodies):
        print(f"  [{i}] {n:26s} m={rb_props[i].mass:.4f} kg com={rb_props[i].com}")
    print(f"\n碰撞几何 {len(shapes)} 个, 显示几何 "
          f"{gym.get_asset_rigid_body_count(asset)} 刚体")
    gym.destroy_sim(sim)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
