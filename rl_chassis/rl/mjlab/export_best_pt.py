#!/usr/bin/env python3
"""把 mjlab (RSL-RL 5.x) 训练出的最终 actor 导出成真机部署用的 TorchScript best.pt。

输入输出与 ``v12_flat_solid_mass16_v10.onnx`` 完全一致：

* 输入 ``obs``：125 维 = 25 维本体观测 × 5 帧历史（term 分块、帧旧→新）
* 输出 ``actions``：6 维原始策略动作（范围约 ±1）
  ``[左髋, 左膝, 左轮, 右髋, 右膝, 右轮]``

输出**不做**动作裁剪/缩放，部署端按现有 ONNX 路径
（``onnx_mj_v11.py`` / 真机控制器）先 ``clip(±1.0)`` 再映射成
虚拟腿目标角（±0.5 rad）和轮速目标（±10 rad/s）。

    .venv/bin/python export_best_pt.py \
        --checkpoint logs/rsl_rl/wheelleg_v12_solid/<run>/model_2999.pt \
        --out ../deploy/best.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from export_onnx import build_from_checkpoint


class DeployPolicy(nn.Module):
  """可接受 (125,) 或 (N,125) 的 actor 均值网络（ELU，确定性输出）。"""

  def __init__(self, mlp: nn.Module):
    super().__init__()
    self.mlp = mlp

  def forward(self, obs: torch.Tensor) -> torch.Tensor:
    squeeze = obs.dim() == 1
    if squeeze:
      obs = obs.unsqueeze(0)
    out = self.mlp(obs)
    if squeeze:
      out = out.squeeze(0)
    return out


def _find_latest_solid_checkpoint(root: Path) -> Path:
  files = sorted(root.glob("logs/rsl_rl/wheelleg_v12_solid/*/model_*.pt"),
                 key=lambda p: p.stat().st_mtime)
  if not files:
    raise FileNotFoundError(f"{root} 下没有 solid checkpoint")
  return files[-1]


def main() -> int:
  here = Path(__file__).resolve().parent
  parser = argparse.ArgumentParser()
  parser.add_argument("--checkpoint", type=str, default=None,
                      help="缺省 = solid 实验下最新 checkpoint")
  parser.add_argument("--out", type=str, default=str(here.parent / "deploy" / "best.pt"),
                      help="输出 best.pt（默认 rl_chassis/rl/deploy/best.pt）")
  parser.add_argument("--num-obs", type=int, default=125)
  parser.add_argument("--hidden", type=int, nargs="+", default=[256, 256, 256])
  parser.add_argument("--onnx", type=str,
                      default=str(here.parent / "mujoco" / "v12_flat_solid_mass16_v10.onnx"),
                      help="可选：同源 ONNX，用来对拍数值")
  args = parser.parse_args()

  ckpt_path = Path(args.checkpoint) if args.checkpoint else _find_latest_solid_checkpoint(here)
  out_path = Path(args.out)
  out_path.parent.mkdir(parents=True, exist_ok=True)

  policy = build_from_checkpoint(str(ckpt_path), args.num_obs, tuple(args.hidden))
  deploy = DeployPolicy(policy.mlp).eval()

  sample = torch.randn(4, args.num_obs, dtype=torch.float32)
  with torch.no_grad():
    ref = deploy(sample)
  # 单样本接口也验证一次
  with torch.no_grad():
    ref1 = deploy(sample[0])
  if not torch.allclose(ref1, ref[0], atol=1e-6):
    raise RuntimeError("单样本 forward 与批量 forward 不一致")

  scripted = torch.jit.script(deploy)
  scripted.eval()
  scripted.save(str(out_path))

  # 重新加载，确认 TorchScript 与 eager 逐元素一致
  loaded = torch.jit.load(str(out_path))
  with torch.no_grad():
    got = loaded(sample)
  err = float((ref - got).abs().max())
  print(f"wrote {out_path}  ({out_path.stat().st_size / 1024:.0f} KB)")
  print(f"torchscript check: shape={tuple(got.shape)} max_abs_err={err:.2e}")
  if err > 1e-5:
    raise RuntimeError(f"TorchScript 与 eager 输出偏差过大: {err}")

  # 与现有 ONNX 对拍（可选）
  onnx_path = Path(args.onnx)
  if onnx_path.exists():
    try:
      import numpy as np
      import onnxruntime as ort

      sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
      onnx_out = sess.run(None, {"obs": sample.numpy()})[0]
      max_err = float(np.abs(onnx_out - ref.numpy()).max())
      print(f"onnx check: {onnx_path.name} max_abs_err={max_err:.2e}")
    except ImportError as exc:
      print(f"skip onnx check: {exc}")
  else:
    print(f"skip onnx check: {onnx_path} 不存在")

  ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
  meta = {
    "format": "torchscript",
    "file": out_path.name,
    "checkpoint": str(ckpt_path),
    "iter": int(ckpt.get("iter", -1)),
    "obs_dim": args.num_obs,
    "obs_history": 5,
    "obs_term_dims": [3, 3, 3, 4, 6, 6],
    "obs_terms": ["ang_vel", "gravity", "commands", "hip_knee_pos",
                  "joint_vel", "applied_action"],
    "obs_layout": "term-major, oldest frame first: "
                  "[ang_vel(15)][gravity(15)][commands(15)][hip_knee(20)]"
                  "[joint_vel(30)][applied_action(30)]",
    "obs_scales": {
      "ang_vel": 0.25,
      "commands": [2.0, 0.25, 5.0],
      "joint_vel": 0.05,
    },
    "action_dim": 6,
    "action_order": ["left_hip", "left_knee", "left_wheel",
                     "right_hip", "right_knee", "right_wheel"],
    "action_clip": 1.0,
    "pos_action_scale": 0.5,
    "vel_action_scale": 10.0,
    "virtual_default": {
      "left": [-0.274884, 0.517721],
      "right": [0.274883, -0.517722],
    },
    "control": {
      "step_dt": 0.01,
      "kp": 40.0,
      "kd": 3.0,
      "wheel_kd": 0.3,
      "actuator_tau_s": 0.015,
      "delay_steps": 1,
      "effort_limits": [40.0, 40.0, 20.0, 40.0, 40.0, 20.0],
      "motor_order": ["Left_front", "Left_rear", "Left_Wheel",
                      "Right_front", "Right_rear", "Right_Wheel"],
    },
    "note": "best.pt 输出原始动作，不裁剪；部署端需 clip 到 ±1 后再做任务映射。"
            "策略输出顺序是 [髋,膝,轮]×左右，真机电机力矩顺序是"
            " [front, rear, wheel]×左右，经腿雅可比虚功映射。",
  }
  meta_path = out_path.with_suffix(out_path.suffix + ".meta.json")
  meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
  print(f"wrote {meta_path}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
