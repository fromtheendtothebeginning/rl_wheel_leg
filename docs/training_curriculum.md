# 分阶段训练路线（机动 → 台阶 → 跳跃 → 自救）

本文记录现有 ONNX 策略的实测边界、训练后端现状，以及后续分阶段课程的目标与验收线。
测试环境：`rl_chassis/mjcf/rl_chassis_v12_scene.xml`（闭链 + 4 个 POM 导轮），
Python `~/IsaacLab-3.0/.venv`（mujoco 3.15 / onnxruntime 1.26）。

## 1. 现有 ONNX 实测

### 1.1 平地指令跟踪（6 s，`v11_terrain_tall.onnx`）

| 指令 | 实测 | 误差 |
|---|---|---|
| 站立，h=0.15 | h=0.1517 m，倾角 0.5° | +1.7 mm |
| vx=+1.0 m/s | 平均 +0.884 m/s，末速 +0.982 | −12% / −2% |
| vx=−1.0 m/s（`v11_terrain`） | 平均 −0.903 m/s，末速 −0.870 | −10% / −13% |
| ωz=+1.0 rad/s | 末速 +0.922 | −8% |
| ωz=−1.0 rad/s（`v11_terrain`） | 末速 −0.923 | −8% |
| h=0.20 | h=0.2008 m | +0.8 mm |
| h=0.24 | h=0.2391 m | −0.9 mm |
| vx=0.5 + ωz=0.3 | 平均 vx +0.276，末 ωz +0.205 | 组合指令跟踪明显变差 |

`v11_terrain_tall` 比 `v11_terrain` 更适合高度指令：h=0.20/0.24 时后者只有
0.1945/0.2255 m。平地上推荐 `v11_terrain_tall`，直立推荐 `v11_terrain`（零指令漂移最小）。

### 1.2 台阶测试（`v11_terrain_tall.onnx`，台阶进深 0.7 m）

| 台阶高度 | 结果 |
|---|---|
| 3 cm | 连续爬 4 级（x=3.39 m，h=0.271 m） |
| 4 cm | 卡在第一级（x=1.06 m） |
| 5 cm | 连续爬 5 级（x=3.98 m，h=0.403 m） |
| 6 cm | 卡住（x=0.70 m） |
| 7 cm | 卡住（x=0.28 m） |
| 14 cm | 失败（卡住或倾倒） |
| 18 cm | 失败（倾倒） |

关键结论：

* **不是单调的高度极限**：3/5 cm 能爬多级，4/6/7 cm 反而卡住。说明现有策略对台阶
  的接触时序很敏感，缺乏地形感知，靠本体感觉“撞运气”。
* 关掉 4 个 POM 导轮的碰撞体后，7 cm 和 14 cm 仍然上不去（平均前进速度 +0.03/−0.04 m/s），
  所以导轮不是唯一原因；但导轮在台阶面的先接触确实会改变时序，训练模型里必须体现它们，
  或者明确决定把导轮碰撞关掉。
* 现有 ONNX 只能在平地和很小的台阶上使用。**上台阶必须重新训练**，不能靠调参数解决。

## 2. 训练后端现状

`rl_chassis/rl/plane_v11/` 是 **Isaac Gym** 工程（`train.py` 直接 `import isaacgym`）。
本机没有安装 `isaacgym`：

```
ModuleNotFoundError: No module named 'isaacgym'
```

Isaac Gym Preview 已停止维护，且 RTX 5060（Blackwell / sm_120）与旧 CUDA 二进制的兼容性
不可靠。本机唯一验证过的训练栈是 **Isaac Lab 3.0**：

* `~/IsaacLab-3.0/.venv`，Python 3.12，torch cu130；
* `docs/isaac_sim_setup.md` 记录过 `Isaac-Cartpole-Direct` 训练跑通与原位渲染。

因此训练侧需要把 `wheel_legged_gym` 的环境移植到 Isaac Lab 的 `DirectRLEnv`，
而不是继续在 Isaac Gym 上加功能。

已经完成的移植前置：

* 训练用串联 URDF `rl_chassis/urdf/rl_chassis_train.urdf` 存在；
* 已转成 USD：`out/train_urdf/rl_chassis_v11.usda`（2.2 MB，`out/` 已 gitignore）；
* 校验：7 个 body（base + 左右各 大腿/小腿/轮）、6 个 revolute 关节、
  articulation root 在 `/rl_chassis_v11/Geometry/base_link`，关节限位 ±360°。

重新生成 USD：

```bash
mkdir -p out/train_urdf
~/IsaacLab-3.0/.venv/bin/urdf_usd_converter \
  rl_chassis/urdf/rl_chassis_train.urdf out/train_urdf
```

## 3. 分阶段课程

原则：**每个阶段只在前一阶段策略上做增量**，每阶段都有可量化的验收线，不达标不进入下一阶段。
训练模型用串联腿（和现有 sim2sim 一致），但第 2 阶段起必须把 POM 导轮纳入训练资产，
否则策略学到的台阶时序无法迁移到 v12 闭链模型。

### Stage 1：平地机动（复现 + 对齐接口）

* 环境：平地，6 个主动关节，PD 与现有 `plane_v11` 一致（髋/膝 kp=40 kd=3；轮 kd=0.3）。
* 观测 25 维、动作 6 维，与 `onnx_mj_v11.py` 完全一致；先训练无 history 的 MLP，
  验证通后再加 5×25 history encoder。
* 指令：vx ∈ [−2, 2]、ωz ∈ [−2, 2]、h ∈ [0.14, 0.24]。
* 训练量：4096 envs，2000–3000 iter（对齐现有 v11_plane_3000 的量级）。
* 验收：|vx 误差| < 10%（±1.0）、|ωz 误差| < 10%、|h 误差| < 5 mm、摔倒率 < 1%。

### Stage 2：台阶 / 粗糙地形

* 资产改动：把 4 个 POM 导轮（被动转动 + 圆柱碰撞）加入训练 URDF；
  如果决定部署时关闭导轮碰撞，则训练资产同步关闭，保持一致。
* 地形：Isaac Lab 的 `HfPyramidStairsTerrainCfg`，进深 0.7 m，高度 3→14 cm 课程；
  可混入 5–10° 斜坡和 stepping stones。
* 奖励：前进速度、机身高度、俯仰/横滚、轮子接地、台阶接触、能量；对底盘/导轮撞台阶立面加惩罚。
* 课程：台阶高度按成功率自动升降；先低速（0.3 m/s）后 0.5–0.8 m/s。
* 验收：14 cm 台阶以 0.5 m/s 连续爬 6 级，20 次里 ≥ 90% 成功，底盘不刮台阶。

### Stage 3：跳跃

* 指令层新增“跳”触发（或用高度指令的快速阶跃）；记录离地相（四轮接触力为 0）。
* 奖励：起跳前的蹲伏、离地时间、最高点高度、落地时的速度/姿态稳定；落地摔倒重罚。
* 需要先确认执行器力矩上限（现有 ±40 N·m 髋/膝、±20 N·m 轮）能否提供足够冲量；
  可能需要提高轮/髋力矩或增加前馈。
* 验收：原地跳过 10 cm 障碍/台阶，落地后 1 s 内恢复站立，连续 10 次不摔。

### Stage 4：自救（跌倒恢复）

* 初始化：随机 roll ±90°、pitch ±60°、随机关节角，轮子可能朝上。
* 奖励：机身竖直度、四轮重新接地、恢复时间；对关节力矩/速度设上限，避免“甩起来”。
* 课程：从接近站立的小角度开始，逐步扩大初始倾倒角；成功后加入“从任意姿态恢复”。
* 验收：80% 的跌倒初始状态在 3 s 内自行站起，不依赖外部扶正。

## 4. 移植清单（Isaac Lab）

1. `rl_chassis/rl/isaaclab/` 新增 `DirectRLEnv`：scene（地面/机器人/接触传感器）、
   `_pre_physics_step`（PD 力矩）、`_get_observations`（25 维）、`_get_rewards`、`_get_dones`、
   `_reset_idx`（指令重采样、姿态随机化）。
2. 用 `ArticulationCfg` + `UsdFileCfg` 加载 `out/train_urdf/rl_chassis_v11.usda`；
   6 个关节用 effort 控制，PD 在 env 内按子步计算（与 `onnx_mj_v11.py` 一致）。
3. 地形：Stage 1 用 `GroundPlaneCfg`；Stage 2 换 `TerrainImporterCfg` + 台阶课程。
4. 奖励/终止：先对齐 `wheel_legged_config.py` 的量级，再加台阶/跳跃/自救项。
5. 训练：`isaaclab train --task <注册名> --rl_library rsl_rl`，日志与 ckpt 放 `logs/`。
6. 导出 ONNX（obs 25 + history 125 → actions 6）后，用 `ona_mj_v12.py` 在闭链 v12 上做 sim2sim，
   每阶段都跑一遍第 1 节的平地/台阶指标。

## 5. 复现命令

```bash
PY=~/IsaacLab-3.0/.venv/bin/python

# 平地：前进 / 转向 / 高度
$PY rl_chassis/rl/mujoco/onnx_mj_v12.py --onnx rl_chassis/rl/mujoco/v11_terrain_tall.onnx \
    --no-viewer --duration 6 --cmd-lin-vel 1.0
$PY rl_chassis/rl/mujoco/onnx_mj_v12.py --onnx rl_chassis/rl/mujoco/v11_terrain_tall.onnx \
    --no-viewer --duration 6 --cmd-height 0.24

# 台阶场景（进深 0.7 m，起点 x=1 m）
$PY rl_chassis/rl/mujoco/make_stairs_scene.py --step-height 0.07 --n 6 \
    --step-depth 0.7 --start-x 1.0 --out /tmp/stairs_0.07.xml
$PY rl_chassis/rl/mujoco/onnx_mj_v12.py --xml /tmp/stairs_0.07.xml \
    --onnx rl_chassis/rl/mujoco/v11_terrain_tall.onnx --no-viewer \
    --duration 15 --cmd-height 0.15 --cmd-lin-vel 0.6

# 训练资产
mkdir -p out/train_urdf
~/IsaacLab-3.0/.venv/bin/urdf_usd_converter \
    rl_chassis/urdf/rl_chassis_train.urdf out/train_urdf
```

## 6. 验收总表

| 阶段 | 指标 | 达标线 |
|---|---|---|
| 1 平地 | 速度/转向/高度跟踪 | 误差 < 10% / 10% / 5 mm，摔倒率 < 1% |
| 2 台阶 | 14 cm × 6 级 | 成功率 ≥ 90%，0.5 m/s，不刮底盘 |
| 3 跳跃 | 10 cm 障碍 | 10 次连续成功，落地 1 s 内稳定 |
| 4 自救 | 随机跌倒 | 80% 在 3 s 内站起 |
