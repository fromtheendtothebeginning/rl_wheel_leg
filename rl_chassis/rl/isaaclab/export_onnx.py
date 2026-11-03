#!/usr/bin/env python3
"""把 Isaac Lab (RSL-RL) 训练出的策略导出成 sim2sim 用的 ONNX。

这一轮 Stage 1 策略是单输入：``obs`` (N, 25) → ``actions`` (N, 6)。
旧的 v11 策略是 25 + 125 双输入；``onnx_mj_v11.py`` 已经按输入个数自动适配。

用法（仓库根目录）::

    ~/IsaacLab-3.0/.venv/bin/python rl_chassis/rl/isaaclab/export_onnx.py \\
        --checkpoint ~/IsaacLab-3.0/logs/rsl_rl/wheel_legged_isaaclab/<run>/model_600.pt \\
        --out rl_chassis/rl/mujoco/v12_stage1_flat.onnx
"""

from __future__ import annotations

import argparse

import torch
from torch import nn


class PolicyMLP(nn.Module):
    """RSL-RL 的 actor 均值网络：25 -> 128 -> 64 -> 32 -> 6，ELU。"""

    def __init__(self, num_obs: int = 25, num_actions: int = 6):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(num_obs, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, 32),
            nn.ELU(),
            nn.Linear(32, num_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mlp(obs)


def build_from_checkpoint(path: str) -> PolicyMLP:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt.get("actor_state_dict", ckpt.get("model_state_dict", ckpt))
    policy = PolicyMLP()
    mapped = {
        f"mlp.{i}.weight": state[f"mlp.{i}.weight"]
        for i in (0, 2, 4, 6)
    }
    mapped.update({f"mlp.{i}.bias": state[f"mlp.{i}.bias"] for i in (0, 2, 4, 6)})
    policy.load_state_dict(mapped, strict=True)
    policy.eval()
    print(f"loaded {path} (iter={ckpt.get('iter', '?')})")
    return policy


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    policy = build_from_checkpoint(args.checkpoint)
    dummy = torch.zeros(1, 25, dtype=torch.float32)

    torch.onnx.export(
        policy,
        dummy,
        args.out,
        input_names=["obs"],
        output_names=["actions"],
        opset_version=17,
        dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}},
        dynamo=False,
    )
    print(f"wrote {args.out}")

    # 快速自检：ONNX 与 PyTorch 输出要一致
    try:
        import numpy as np
        import onnxruntime as ort

        sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
        sample = torch.randn(3, 25, dtype=torch.float32)
        with torch.no_grad():
            ref = policy(sample).numpy()
        got = sess.run(None, {"obs": sample.numpy()})[0]
        err = float(np.abs(ref - got).max())
        print(f"onnx check: shape={got.shape} max_abs_err={err:.2e}")
    except ImportError as exc:
        print(f"skip onnx check: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
