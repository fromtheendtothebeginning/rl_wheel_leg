#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v11 训练环境冒烟测试：小规模建环境 + 跑若干步，检查
虚拟腿 L0/theta0、默认站姿高度、PD 力矩、奖励项、终止率、观测维度。

用法:
  export PYTHONPATH=$PWD:$PYTHONPATH
  export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
  python smoke_test_v11.py --steps 200 --action zero|sine
"""

from __future__ import annotations

import argparse
import sys

import isaacgym  # noqa: F401  必须最先导入
import torch

from wheel_legged_gym.envs import *  # noqa: F401,F403
from wheel_legged_gym.utils import get_args, task_registry


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--num_envs", type=int, default=64)
    ap.add_argument("--action", choices=("zero", "sine"), default="zero")
    ap.add_argument("--headless", action="store_true", default=True)
    # 兼容 get_args()
    ap.add_argument("--task", default="wheel_legged")
    args_cli, _ = ap.parse_known_args()

    argv_backup = sys.argv
    sys.argv = [argv_backup[0], "--task", args_cli.task, "--headless"]
    try:
        args = get_args()
    finally:
        sys.argv = argv_backup
    args.headless = True

    env_cfg, _ = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = args_cli.num_envs
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.num_rows = 2
    env_cfg.terrain.num_cols = 2
    env_cfg.terrain.max_init_terrain_level = 0
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_inertia = False
    env_cfg.domain_rand.randomize_Kp = False
    env_cfg.domain_rand.randomize_Kd = False

    env, cfg = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    print(f"\n== 环境建好: {cfg.env.num_envs} envs, dof={env.num_dof}, "
          f"obs={cfg.env.num_observations}, actions={cfg.env.num_actions}")
    print(f"   dof 顺序: {env.dof_names}")
    print(f"   关节索引: wheel={env.feet_indices}, termination_contact={env.termination_contact_indices}")
    print(f"   PD: kp={env.p_gains[0].tolist()} kd={env.d_gains[0].tolist()}")
    print(f"   力矩上限: {env.torque_limits.tolist()}")
    print(f"   默认角: {[round(v, 4) for v in env.default_dof_pos[0].tolist()]}")
    print(f"   总质量: {env.base_mass.mean().item():.4f} kg (base)")

    # 注意：先 reset（构造后的状态是 PhysX 的中间态，不是初始站姿）
    env.reset()
    obs, _, _, _, _, obs_history = env.step(torch.zeros(env.num_envs, env.num_actions, device=env.device))
    print(f"\n== 初始状态: obs={tuple(obs.shape)} obs_history={tuple(obs_history.shape)}")
    print(f"   L0={env.L0[0].tolist()} (m)  theta0={torch.rad2deg(env.theta0[0]).tolist()} (deg)")
    print(f"   base_height={env.base_height[0].item():.4f} m, root_z={env.root_states[0,2].item():.4f} m")
    print(f"   dof_pos={[round(v,4) for v in env.dof_pos[0].tolist()]}")

    worst = dict(term=0.0, h=0.0, reward=0.0)
    for i in range(args_cli.steps):
        if args_cli.action == "sine":
            t = i * cfg.control.decimation * cfg.sim.dt
            actions = 0.15 * torch.sin(
                torch.tensor(2 * 3.1416 * 0.5 * t, device=env.device)
            ) * torch.ones(env.num_envs, env.num_actions, device=env.device)
        else:
            actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
        obs, _, rew, done, _, obs_history = env.step(actions)
        worst["term"] = max(worst["term"], done.float().mean().item())
        worst["h"] = max(worst["h"], env.base_height.mean().item())
        worst["reward"] = max(worst["reward"], rew.mean().item())
        if i % max(1, args_cli.steps // 5) == 0:
            print(f"  step {i:4d}: h={env.base_height.mean().item():+.4f} m  "
                  f"L0={env.L0.mean().item():.4f} m  theta0={torch.rad2deg(env.theta0).mean().item():+.2f} deg  "
                  f"vx={env.base_lin_vel[:,0].mean().item():+.3f} m/s  rew={rew.mean().item():+.4f}  "
                  f"done={done.float().mean().item():.3f}")
    print(f"\n== {args_cli.steps} 步结束: 最大终止率 {worst['term']:.3f}, "
          f"最大 base 高度 {worst['h']:.4f} m, 最大单步奖励 {worst['reward']:.4f}")
    print("   奖励项均值:")
    for k, v in sorted(env.episode_sums.items()):
        m = v[~torch.isnan(v)].mean().item()
        if abs(m) > 1e-9:
            print(f"     {k:26s} {m:+.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
