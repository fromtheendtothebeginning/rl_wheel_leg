#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 wxk's-branch 的平地策略（v11_flat_slow.onnx）蒸馏进本工程的 actor，
作为 PPO 的训练起点（"用他的策略开始训练"）。

流程：
1. 在我们的 v12 闭链 mjlab 环境里跑 v11_flat_slow（25 维当前观测 + 5×25 帧序
   历史，与 onnx_mj_v11 的口径一致；动作加小噪声覆盖更多状态）；
2. 收集 (125 维 term-major 观测 → 6 维动作) 样本；
3. 监督训练我们的 actor MLP（125→256³→6），保存为 RSL-RL 兼容 checkpoint
   （logs/rsl_rl/wheelleg_v12_mjlab/distill_wxk/model_0.pt，以现有同架构
   checkpoint 为模板，只替换 actor 权重并重置探索 std）；
4. 之后用 `train.py --run-name <new> --resume` 从这份 checkpoint 开始 PPO。

用法（mjlab/ 目录，训练需先停止以独占 GPU）：
    .venv/bin/python scripts/distill_wxk_policy.py --num-envs 1024 --steps 1200
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
from torch import nn

MJLAB_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(MJLAB_DIR, "src"))

ONNX_PATH = os.path.join(
    MJLAB_DIR, "..", "mujoco", "v11_flat_slow.onnx"
)
HIST = 5
DIMS = (3, 3, 3, 4, 6, 6)  # ang_vel, gravity, cmd, hip_knee, dof_vel, action


def split_obs125(obs: torch.Tensor):
    """term-major 125 -> 当前帧 25（各 term 的最新一帧）。"""
    out, i = [], 0
    for d in DIMS:
        block = obs[:, i : i + d * HIST]
        out.append(block[:, -d:])  # 每个 term 块的最后 d 维 = 最新帧
        i += d * HIST
    return torch.cat(out, dim=-1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-envs", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--noise", type=float, default=0.15, help="动作噪声幅度（探索覆盖）")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--out-run", default="distill_wxk")
    args = ap.parse_args()

    import onnxruntime as ort

    import mjlab_wheelleg  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg
    from mjlab_wheelleg import TASK_ID

    # ---------- 1. 收集数据 ----------
    env_cfg = load_env_cfg(TASK_ID)
    env_cfg.scene.num_envs = args.num_envs
    # 关键：禁用课程。否则 CommandStandingCurriculum 构造时就把 stage 0 的
    # "全站定 + 固定高度 0.19" 应用回来，学生永远见不到走行/高度指令
    # （实测：蒸馏出的策略对 vx=+0.5 指令倒退 -0.32 m/s）。
    env_cfg.curriculum = {}
    env_cfg.commands["cmd"].standing_prob = 0.3  # 70% 走行，覆盖机动
    env_cfg.commands["cmd"].lin_vel_range = (-1.2, 1.2)
    env_cfg.commands["cmd"].ang_vel_range = (-1.0, 1.0)
    env_cfg.commands["cmd"].height_range = (0.14, 0.24)
    env_cfg.observations["actor"].enable_corruption = False  # 教师基于干净观测
    env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0")

    providers = ["CPUExecutionProvider"]
    sess = ort.InferenceSession(ONNX_PATH, providers=providers)
    in_names = [i.name for i in sess.get_inputs()]
    print(f"teacher ONNX inputs: {in_names}", flush=True)
    assert len(in_names) >= 2, "预期 25 + 125 双输入策略"

    def collect(policy_fn, steps: int, tag: str):
        """policy_fn(obs125)->numpy actions；返回 (X, Y) 样本对。"""
        obs, _ = env.reset()
        n = args.num_envs
        hist = np.zeros((n, HIST * 25), dtype=np.float32)  # 帧序 5×25，旧→新
        xs, ys = [], []
        t0 = time.time()
        for t in range(steps):
            o = obs["actor"].detach()
            cur25 = split_obs125(o).cpu().numpy().astype(np.float32)
            hist[:, :-25] = hist[:, 25:]
            hist[:, -25:] = cur25
            # DAgger：教师对**当前状态**标注（无论状态是谁走出来的）
            act = sess.run(None, {in_names[0]: cur25, in_names[1]: hist})[0].astype(np.float32)
            xs.append(o.cpu())
            ys.append(torch.from_numpy(act))
            executed = np.clip(act + np.random.randn(*act.shape) * args.noise, -100, 100)
            if policy_fn is not None:
                executed = np.clip(policy_fn(o), -100, 100)
                executed = np.clip(executed + np.random.randn(*executed.shape) * args.noise, -100, 100)
            obs, rew, term, trunc, infos = env.step(torch.from_numpy(executed).cuda())
            if (t + 1) % 400 == 0:
                print(f"  {tag} {t+1}/{steps} ({time.time()-t0:.0f}s)", flush=True)
        return torch.cat(xs, 0), torch.cat(ys, 0)

    # 第一轮：教师驱动（覆盖率）
    X1, Y1 = collect(None, args.steps, "collect-teacher")
    print(f"教师数据: {tuple(X1.shape)}", flush=True)

    # ---------- 2. 监督训练 actor ----------
    class Actor(nn.Module):
        def __init__(self):
            super().__init__()
            dims = [125, 256, 256, 256, 6]
            layers = []
            for i in range(4):
                layers.append(nn.Linear(dims[i], dims[i + 1]))
                if i < 3:
                    layers.append(nn.ELU())
            self.mlp = nn.Sequential(*layers)

        def forward(self, x):
            return self.mlp(x)

    dev = "cuda:0"
    actor = Actor().to(dev)
    opt = torch.optim.Adam(actor.parameters(), lr=1e-3)
    Xd, Yd = X1.to(dev), Y1.to(dev)
    ntrain = int(len(Xd) * 0.95)
    perm = torch.randperm(len(Xd))
    tr, va = perm[:ntrain], perm[ntrain:]
    for ep in range(args.epochs):
        actor.train()
        perm = torch.randperm(ntrain)  # CPU 上打乱，索引 tr（CPU）
        for i in range(0, ntrain, args.batch):
            idx = tr[perm[i : i + args.batch]].to(dev)
            loss = torch.mean(torch.square(actor(Xd[idx]) - Yd[idx]))
            opt.zero_grad(); loss.backward(); opt.step()
        actor.eval()
        with torch.no_grad():
            vl = torch.mean(torch.square(actor(Xd[va]) - Yd[va])).item()
        print(f"  epoch {ep+1}/{args.epochs}: train_mse={loss.item():.5f} val_mse={vl:.5f}", flush=True)
    actor.eval()
    with torch.no_grad():
        pred = actor(Xd[va])
    err = (pred - Yd[va]).abs()
    print(f"BC-1 验证集 |误差|: mean={err.mean().item():.4f} p95={err.flatten().quantile(0.95).item():.4f}", flush=True)

    # ---------- 2b. DAgger 第二轮：学生驱动 + 教师标注 ----------
    @torch.no_grad()
    def student_act(o: torch.Tensor) -> np.ndarray:
        return actor(o.to(dev)).cpu().numpy().astype(np.float32)

    X2, Y2 = collect(student_act, max(300, args.steps // 2), "collect-student")
    X = torch.cat([X1, X2], 0)
    Y = torch.cat([Y1, Y2], 0)
    print(f"DAgger 合并数据: {tuple(X.shape)}", flush=True)
    Xd, Yd = X.to(dev), Y.to(dev)
    ntrain = int(len(Xd) * 0.95)
    perm = torch.randperm(len(Xd))
    tr, va = perm[:ntrain], perm[ntrain:]
    for ep in range(args.epochs):
        actor.train()
        perm = torch.randperm(ntrain)
        for i in range(0, ntrain, args.batch):
            idx = tr[perm[i : i + args.batch]].to(dev)
            loss = torch.mean(torch.square(actor(Xd[idx]) - Yd[idx]))
            opt.zero_grad(); loss.backward(); opt.step()
        actor.eval()
        with torch.no_grad():
            vl = torch.mean(torch.square(actor(Xd[va]) - Yd[va])).item()
        print(f"  DAgger epoch {ep+1}/{args.epochs}: val_mse={vl:.5f}", flush=True)
    actor.eval()
    with torch.no_grad():
        pred = actor(Xd[va])
    err = (pred - Yd[va]).abs()
    print(f"DAgger 验证集 |误差|: mean={err.mean().item():.4f} p95={err.flatten().quantile(0.95).item():.4f}", flush=True)

    # ---------- 3. 写 RSL-RL checkpoint ----------
    import glob

    exp_dir = os.path.join(MJLAB_DIR, "logs", "rsl_rl", "wheelleg_v12_mjlab")
    # 模板：任意一个 256³ 架构的 checkpoint（取最新的 flat_v6 model_*.pt）
    tmpl = sorted(
        glob.glob(os.path.join(exp_dir, "*flat_v6", "model_*.pt")),
        key=lambda p: int(os.path.basename(p)[6:-3]),
    )[-1]
    ckpt = torch.load(tmpl, map_location="cpu", weights_only=False)
    sd = ckpt["actor_state_dict"]
    net = actor.mlp.state_dict()
    for k, v in net.items():
        sd[f"mlp.{k}"] = v.detach().cpu()
    if "distribution.std_param" in sd:
        sd["distribution.std_param"][:] = 0.5
    ckpt["iter"] = 0
    ckpt["infos"] = {"distilled_from": os.path.basename(ONNX_PATH)}
    out_dir = os.path.join(exp_dir, args.out_run)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "model_0.pt")
    torch.save(ckpt, out_path)
    print(f"wrote {out_path}（模板 {os.path.basename(tmpl)}）", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
