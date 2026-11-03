#!/usr/bin/env python3
"""把 mjlab (RSL-RL 5.x) 训出的策略导出成 sim2sim 用的 ONNX。

输出与 isaaclab/export_onnx.py 相同的图：单输入 ``obs`` (N, 25) → ``actions``
(N, 6)，actor 25→128→64→32→6（ELU，确定性均值）。
``onnx_mj_v12.py`` 按输入个数自动适配，直接加载即可。

    uv run python export_onnx.py --checkpoint logs/rsl_rl/wheelleg_v12_mjlab/<run>/model_600.pt \
        --out ../mujoco/v12_mjlab_flat.onnx
"""

from __future__ import annotations

import argparse

import torch
from torch import nn


class PolicyMLP(nn.Module):
  """RSL-RL 的 actor 均值网络。flat_hist 版：125(=25×5 历史) -> 256³ -> 6，ELU。"""

  def __init__(self, num_obs: int = 125, num_actions: int = 6,
               hidden: tuple[int, ...] = (256, 256, 256)):
    super().__init__()
    dims = [num_obs, *hidden, num_actions]
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
      layers.append(nn.Linear(dims[i], dims[i + 1]))
      if i < len(dims) - 2:
        layers.append(nn.ELU())
    self.mlp = nn.Sequential(*layers)

  def forward(self, obs: torch.Tensor) -> torch.Tensor:
    return self.mlp(obs)


def build_from_checkpoint(path: str, num_obs: int, hidden: tuple[int, ...]) -> PolicyMLP:
  ckpt = torch.load(path, map_location="cpu", weights_only=False)
  state = ckpt.get("actor_state_dict", ckpt.get("model_state_dict", ckpt))
  policy = PolicyMLP(num_obs=num_obs, hidden=hidden)
  linear_idx = [i for i, m in enumerate(policy.mlp) if isinstance(m, nn.Linear)]
  mapped = {}
  for i in linear_idx:
    mapped[f"mlp.{i}.weight"] = state[f"mlp.{i}.weight"].float()
    mapped[f"mlp.{i}.bias"] = state[f"mlp.{i}.bias"].float()
  policy.load_state_dict(mapped, strict=True)
  policy.eval()
  print(f"loaded {path} (iter={ckpt.get('iter', '?')})")
  return policy


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--checkpoint", required=True)
  parser.add_argument("--out", required=True)
  parser.add_argument("--num-obs", type=int, default=125)
  parser.add_argument("--hidden", type=int, nargs="+", default=[256, 256, 256])
  args = parser.parse_args()

  policy = build_from_checkpoint(args.checkpoint, args.num_obs, tuple(args.hidden))
  dummy = torch.zeros(1, args.num_obs, dtype=torch.float32)

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

  try:
    import numpy as np
    import onnxruntime as ort

    sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    sample = torch.randn(3, args.num_obs, dtype=torch.float32)
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
