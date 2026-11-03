# 真机部署文件（v12 solid v10）

`best.pt` = mjlab 训练出的最终 actor（`flat_solid_mass16_v10` / `model_2999`），
TorchScript 格式，`torch.jit.load()` / `torch::jit::load()` 可直接加载。

## 接口

| 项 | 值 |
|---|---|
| 输入 | `obs` float32，`(125,)` 或 `(N, 125)` |
| 观测构成 | 25 维本体观测 × 5 帧历史（term 分块、帧旧→新）：`[ang_vel(15)][gravity(15)][commands(15)][hip_knee(20)][joint_vel(30)][applied_action(30)]` |
| 输出 | `actions` float32，6 维原始策略动作，约 ±1 |
| 动作顺序 | `[左髋, 左膝, 左轮, 右髋, 右膝, 右轮]` |

完整标定（观测缩放、动作映射、PD/滞后/延迟、电机顺序、默认虚拟角）见
`best.pt.meta.json`，与训练/ONNX 口径逐项一致。

## Python

```python
import torch

policy = torch.jit.load("best.pt", map_location="cpu").eval()
act = policy(torch.zeros(125, dtype=torch.float32))  # -> (6,)
act = torch.clamp(act, -1.0, 1.0)
```

## C++（libtorch）

```cpp
auto policy = torch::jit::load("best.pt");
policy.eval();
auto obs = torch::zeros({1, 125}, torch::kFloat32);
auto act = policy.forward({obs}).toTensor().squeeze(0);  // (6,)
act = act.clamp(-1.0, 1.0);
```

## 部署注意

* **输出是原始动作，未裁剪**：真机端先 `clip(±1.0)`，再按
  `pos_action_scale=0.5`（虚拟髋/膝目标 = 默认角 ±0.5 rad）、
  `vel_action_scale=10.0`（轮速目标 ±10 rad/s）映射。
* **控制周期 10 ms**（100 Hz），执行器一阶滞后 15 ms，动作延迟 1 拍。
* 力矩下发顺序是 `[front, rear, wheel]×左右`，策略动作是 `[髋,膝,轮]×左右`；
  中间经腿雅可比虚功映射（与 `onnx_mj_v11.py` 的 `tau_real` 计算一致）。
* 观测里的 `applied_action` 用**最新下发动作**（不是延迟后的），
  与训练/ONNX 一致；`gravity` 直立 = `(0,0,-1)`。
* `best.pt` 与 `v12_flat_solid_mass16_v10.onnx` 在随机输入上
  `max_abs_err ≈ 1.9e-6`，两条路径数值等价。

## 重新导出

```bash
cd rl_chassis/rl/mjlab
.venv/bin/python export_best_pt.py \
  --checkpoint logs/rsl_rl/wheelleg_v12_solid/2026-11-03_15-19-56_flat_solid_mass16_v10/model_2999.pt
```
