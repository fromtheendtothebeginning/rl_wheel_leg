# mjlab 轮腿训练（v12 闭链，MuJoCo Warp GPU）

用 [mjlab](https://mujocolab.github.io/mjlab/)（MuJoCo 官方生态的 RL 框架：
MuJoCo Warp GPU 后端 + Isaac Lab 式 manager-based 环境 + RSL-RL 训练）
直接在 **v12 真机闭链 MJCF** 上训练轮腿策略。

与 `isaaclab/`（串联 USD + PhysX）路线的本质区别：**训练/部署同一套物理**。
训练模型就是 sim2sim 的目标模型（`mjcf/rl_chassis_v12.xml`，含 4 个 connect
等式约束与 4 个 POM 导轮），目标是把 sim2sim gap 整体消掉——策略的力矩映射
（闭链雅可比、虚功原理）与 `onnx_mj_v12.py` 逐项一致，导出的 ONNX 无需任何
改口径就能直接跑 sim2sim。

## 1. 结构

```
mjlab/
├── pyproject.toml            # uv 工程（mjlab 1.6 + torch + rsl-rl 5.4）
├── train.py                  # 训练入口（复用 mjlab.scripts.train 管线）
├── export_onnx.py            # checkpoint -> ONNX（25 obs -> 6 actions）
├── scripts/
│   ├── warp_fidelity.py      # GPU(mujoco-warp) vs CPU(MuJoCo) 闭链保真度对拍
│   ├── check_leg_torch.py    # 闭链运动学 torch 实现的数值校验
│   └── smoke_env.py          # 环境冒烟（构建/reset/step/维度）
└── src/mjlab_wheelleg/
    ├── __init__.py           # 任务注册 WheelLegged-Flat-v12-Closed
    ├── leg_torch.py          # 闭链运动学（批量解析雅可比 + 牛顿逆解）
    ├── mdp.py                # 动作/观测/奖励/终止/指令/reset 事件
    ├── env_cfg.py            # ManagerBasedRlEnvCfg 组装
    └── rl_cfg.py             # PPO 配置（对齐 isaaclab 版）
```

## 2. 关键设计

* **策略接口不变**：观测 25 维（角速度×0.25、重力投影、指令×(2,0.25,5)、
  虚拟髋/膝角偏差、6 关节速度×0.05、上一步动作），动作 6 维
  （虚拟髋/膝位置目标 ±0.5、轮速目标 ±10）。
* **闭链上的力矩映射**：逐物理步在虚拟空间算 PD（kp=40 kd=3，轮 kd=0.3），
  用解析雅可比 `∂q_shank/∂(q_front,q_rear)` 按虚功原理映射到真机
  (后髋, 前髋曲柄) 力矩，再过 15 ms 一阶滞后与 10 ms 动作延迟。
  与 `onnx_mj_v11.py` 的部署数学完全同构（`leg_torch.py` 的常数
  vendor 自 `rl/tools/leg_kinematics_ref.py`，校验到 1e-8）。
* **闭链一致性 reset**：初始关节角不能独立扰动（否则 connect 约束开局猛拉，
  实测角速度冲到 4 rad/s）。`mdp.ResetLegJoints` 采样虚拟 (rear, shank) 扰动
  后牛顿解出一致的前链 4 关节（残差 <1e-12）。
* **力矩下发必须走 `set_joint_effort_target`**：mjlab 每个物理子步用
  actuator 内部 target 缓冲重算并覆盖 `ctrl`，直接 `write_ctrl_to_sim`
  会被清零（踩过：机器人无力矩瘫倒）。
* **obs 的"上一步动作"= 最新动作**：对齐 `onnx_mj_v11.py` 的
  `last_actions = act`（注释同款）。用延迟一拍的 applied action 会让
  平滑度惩罚与观测对不上，PPO 把动作学爆（实测 ±100 震荡、
  action_smooth 罚 -4 万主导，episode 全部 1 s 内死于 bad_contact）。

## 3. 验证记录（2026-10-25）

| 项 | 结果 |
|---|---|
| warp_fidelity（闭链 GPU vs CPU，5 s 混沌开环） | t=0.05 s 偏差 7e-7 → 混沌放大后饱和 0.13 rad，零控 base_z 差 1e-4 m，无系统漂移 |
| check_leg_torch | 常数 0 偏差；雅可比 vs 有限差分 5e-4、vs 参考实现 2e-8；theta0 1e-16 |
| 零动作站立（GPU env，同配置） | z 稳在 0.147~0.19，与 CPU 参考（0.1416）同水平 |
| 冒烟训练 256 env × 30 iter | 3.5 s/iter，checkpoint 正常 |
| 正式训练 1024 env | ~4.7 s/iter，600 iter ≈ 47 min |

## 4. 怎么跑

```bash
cd rl_chassis/rl/mjlab

# 训练（课程文档 stage1 预算：4096 env × 2000~3000 iter）
uv run python train.py --num-envs 1024 --max-iterations 600 --run-name flat_v2

# 导出 ONNX（obs 25 -> actions 6，onnx_mj_v12.py 直接可载）
uv run python export_onnx.py \
    --checkpoint logs/rsl_rl/wheelleg_v12_mjlab/<run>/model_600.pt \
    --out ../mujoco/v12_mjlab_flat.onnx

# sim2sim 验证（仓库根 .venv，见 mujoco/README_sim2sim.md）
cd ../mujoco && python onnx_mj_v12.py --onnx v12_mjlab_flat.onnx
```

## 5. 实心碰撞变体（`WheelLegged-Flat-v12-Solid`）

**最终采用 hybrid**：12 个运动 link 用原始网格凸包当实体碰撞（MuJoCo 网格碰撞
= 凸包），`base_link` **保留 coacd 凸分解**（腿坑）；车轮仍用解析圆柱（本来就是
实心、滚动光滑）。碰撞 geom 310 → 70。

**为什么 base 不能也用单个凸包**：base 的凸包体积 27 L，会把 6 条腿 link 和车轮
全吞进腹腔。autogen 为了消除装配位姿的互相顶住，会把 `base_link ↔ 6 腿`
（最深 108 mm）、`base_link ↔ 车轮`（28 mm）等 14 对加进 `<exclude>`，腿就能
穿过底盘。实测全凸包策略（`flat_solid_v1`）收敛到折叠腿贴地：目标 0.190 m，
稳定在 **0.066 m**（轮半径 0.06 m），track_lin 0.97、`bad_contact` -0.54。
保留腿坑后这些排除全部消失。

```bash
# 生成 hybrid（weld/free 各一份）
cd rl_chassis
../.venv/bin/python tools/make_mjcf_pins.py \
  --urdf urdf/rl_chassis_v12_std.urdf --outdir mjcf --base free \
  --name rl_chassis_v12_solid --collision-mode solid --solid-skip base_link \
  --passive pom_wheel1_joint pom_wheel2_joint pom_wheel3_joint pom_wheel4_joint

# 训练：warm start + 实心模型精修
# 从零训练（v2/v3）会卡在 0.136 m 的低伏解；已知 flat_bc4 策略在 hybrid 上
# 直接站 0.190 m，用它做 warm start 后曲线立即正确。
cd rl/mjlab
mkdir -p logs/rsl_rl/wheelleg_v12_solid/flat_bc4_seed
cp logs/rsl_rl/wheelleg_v12_mjlab/<flat_bc4 目录>/model_3149.pt \
   logs/rsl_rl/wheelleg_v12_solid/flat_bc4_seed/
.venv/bin/python train.py --task solid --num-envs 2048 --max-iterations 4149 \
  --seed 42 --run-name flat_solid_warm_v1 --resume --load-run flat_bc4_seed
# 日志：logs/rsl_rl/wheelleg_v12_solid/<run>/
```

solid 任务比凸分解版多两项高度约束：`rew_height_under`（低于指令 5 mm
以上按 -5/m 线性罚；2 cm 死区会让策略卡在 0.131/0.150 不往上站）和
`BaseHeightTermination`（低于任务阈值持续 0.3 s 判失败；15.641 kg 模型
用 0.07 m）。只对 `--task solid` 生效，凸分解主任务不变。

体检结果（2026-11-02，`rl_chassis_v12_solid_weld_scene.xml`）：

| 项 | hybrid 实心版 | 凸分解版（baseline） |
|---|---|---|
| 闭链残差（静态/动态） | 0.004 µm / 0.008 mm | 同 |
| 每腿自由度 | 3 | 3 |
| 装配位姿自碰撞 | 0 对 | 0 对 |
| 可用站姿（121 点） | **52** | 52 |
| 轮心 x 可达 | **−0.130 ~ +0.099 m** | −0.130 ~ +0.098 m |
| `Left_front_joint` 压 −1.2 停在 | **−0.403 rad** | −0.391 rad |

**动态验证**：已知的 `v12_flat_bc4_final.onnx` 直接在 hybrid 上跑 sim2sim，
cmd 0.19 m → 稳定 0.1902~0.1907 m、倾斜 0.6°、只有车轮接地，说明 hybrid 的
接触力学没有引入新的不稳定。

**warm start 结果（2026-11-02，iter 3212）**：`stand_height 0.193 m`（指令
0.190）、`rise 0.197 s`（目标 ≤2 s）、`settle 0.029 s`（目标 ≤0.2 s）、
`height_err 0.004 m`、ep_len 2000、`base_too_low 0`。sim2sim：站立 0.190 m；
cmd 1.0 m/s → 0.93 m/s、h 0.182 m、倾斜 1.8°，只有车轮接地。从零跑的
`flat_solid_v2/v3` 均稳在 0.136 m，是 exp 高度核 + 低伏稳定解，已由
`height_under` 与 warm start 解决。

**额外保护**：solid 任务注册时加 `min_base_height=0.07`（`mdp.BaseHeightTermination`，
低于 0.07 m 持续 0.3 s 判失败）。15.641 kg 模型零动作会自然沉到 0.073 m，
所以不能沿用 5.641 kg 时代的 0.12 m；0.07 m 仍能挡住全凸包版 0.056~0.066 m
的折腿贴地解。另外启用 `mdp.SustainedBadContact`：底盘/腿/POM 接触力 >30 N
持续 0.5 s 判失败。v8 曾用后部 POM 导轮撑出 27° 前倾、ep_len 2000 但
`bad_contact -0.975`——二值接触惩罚饱和后没有梯度，只能靠终止打断这个伪站姿。
20 N/0.3 s 会把早期普通落地接触也判死（v9 ep_len 60），所以放宽到 30 N/0.5 s。
凸分解主任务不受影响。

训练时 `BAD_CONTACT_GEOMS_SOLID`（`(?!(?:Left|Right)_Wheel_link_col$).*_col`）
负责把 hybrid 的碰撞 geom（base 分块 `*_cNN_col` + 腿 `<link>_col`）接进蹭地
惩罚传感器。全凸包版留在 `mjcf/*_allhull*` 作为失败对照，不用于训练。

## 6. 已知问题 / 备注

* **base_link 质量（2026-11-03 调整）**：所有 v12 MJCF/URDF 的 base_link 标称
  **15.641 kg**（CAD 5.641 + 10 kg 载荷），训练 startup 再 ±3 kg，整机
  14.88~20.88 kg。`dr.body_mass` 只改质量、不改惯量，等价于在质心加点质量；
  旧 ONNX（按 5.641 kg 训练）在这套质量下已不适用，需要重训后再导出。
* **训练预算还差得远**：Isaac Gym v1（`v11_plane_3000`）用了 2048 env × 3000
  iter 才站稳会走；本目录 600 iter 只是管线验证。`docs/training_curriculum.md`
  的 stage1 验收线（±1.0 m/s 误差 <10%、摔倒率 <1%）按 4096×2000+ 跑。
* **`uv run` 会锁 venv**：训练进行中别再开 uv 命令（会用文件锁卡住）；
  监控用 `.venv/bin/python` 直连。
* mujoco-warp 的接触/约束缓冲是静态分配，凸包多（base 32 块 ×2）摔倒时接触
  暴增会非法访存，`SimulationCfg(nconmax=256, njmax=2048)` 必须给足。
* RTX 5060 Laptop 8G：1024 env 用 3.9 GB，4096 env 需实测（可能要降 decimation
  或分块）。
