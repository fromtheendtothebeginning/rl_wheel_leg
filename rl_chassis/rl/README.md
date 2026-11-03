# rl_chassis_v11 强化学习复现（对标 fudan_rl_wheel_leg）

用 v11 的 URDF/MJCF 复现复旦那套「Isaac Gym 训练 + MuJoCo sim2sim」的轮腿 RL。
参考仓库：`fudan_rl_wheel_leg/`（原地克隆，未改动）。

## 1. 核心思路：训练用串联腿，部署用闭链腿

复旦的机器人是真机闭链（五连杆），但**训练模型是串联腿**——因为 Isaac Gym 不支持闭链。
他们训练用 `infantry_V4_increase.urdf`（只有 base + 大腿 + 小腿 + 轮 7 个 link），
部署时再通过虚拟腿映射把策略的 (髋, 膝) 力矩搬到真机 (髋, 曲柄) 上。

v11 完全同构，所以照搬这个套路：

```
训练（Isaac Gym）                     部署（MuJoCo / 真机）
base → rear_link(大腿)                base → front_Link(曲柄) → child1/2/3 ┐
     → rear_child1(小腿)                                         两个销 ├→ rear_link → rear_child1
     → Wheel                                                           └→ Wheel
  前链整条删掉                        6 个电机：前髋(曲柄)/后髋(大腿)/轮
```

即：**v11 的 rear 链就是复旦那条串联腿**（同样的关节次序、同样的 l1/l2 语义），
多出来的 `front_Link` 曲柄链在训练模型里删掉，部署时它是驱动「虚拟膝」的输入。

## 2. 标定结果（`tools/calibrate_leg.py` 输出）

| 量 | 左腿 | 右腿 |
|---|---|---|
| 大腿长 l1（髋→膝） | 0.215000 m | 0.215000 m |
| 小腿长 l2（膝→轮心） | 0.254000 m | 0.254000 m |
| 平面角零点 a1 / a2 | 179.870° / −212.108° | 同 |
| 关节→平面角符号 sgn | −1 | +1 |
| 车轮半径 / 半宽 | 0.06000 / 0.02918 m | 同 |
| 装配位姿 (q=0) | L0 = 0.1350 m，θ0 = 0°（轮正在髋正下方） | 同 |

校验：

* **闭环逆解 vs 真机**：用 `rl_chassis/mjcf/verify_report.txt` 里焊死基座自由衰减 2 s
  后的真机状态 (front=−0.4976, rear=−0.7726 → child1=−0.1521, child2=+0.0292,
  child3=−0.1521, shank=+0.1521) 反解，四个关节全对、销残差 0；
* **虚拟腿公式 vs 闭环正解**：偏差 < 1 nm；
* **站姿可达域**（物理分支，轮心在髋下方）：L0 ∈ [0.051, 0.469] m，θ0 ∈ [−90°, +46°]；
* **多解**：机构平面上有 1~3 个装配分支，部署时必须靠连续性锁死物理分支
  （`leg_model.V11Leg.solve_loop` 用上一步的解做种子，并限步长）。

虚拟腿口径（与复旦的区别：他们 θ0 = atan2(ey,ex) − π/2，我们用 θ0 = atan2(ex, −ey)，
因为 v11 装配位姿正好是腿竖直向下，这样 0 就是竖直、不会有 ±π 跳变）：

```
theta1 = leg_a1 + leg_sign*q_rear
theta2 = leg_a2 + leg_sign*q_shank
end_x  = offset + l1*cos(theta1) + l2*cos(theta1+theta2)     # +x 前, +z 上
end_z  = l1*sin(theta1) + l2*sin(theta1+theta2)
L0 = hypot(end_x, end_z)      theta0 = atan2(end_x, -end_z)
```

## 3. 目录

```
rl/
├── tools/
│   ├── leg_model.py        # 单腿闭环正/逆解、虚拟腿雅可比、虚拟/真机力矩映射
│   ├── calibrate_leg.py    # 标定 + 校验（对真机状态、公式、可达域、多解）
│   ├── pose_tool.py        # (L0, theta0) <-> 关节角，含真机曲柄角
│   ├── tune_default_pose.py# 算「重力静平衡默认角」= q_des + τ_gravity/kp
│   ├── make_train_urdf.py  # 生成训练用串联 URDF
│   └── check_isaac_asset.py# 在 Isaac Gym 里试载，打印 dof 顺序/质量/力矩上限
└── plane_v11/              # Isaac Gym 训练工程（从 fudan plane/ 移植）
    ├── wheel_legged_gym/   # 环境/配置/PPO
    ├── export_onnx/        # 导出 ONNX（obs 25 + history 125 -> actions 6）
    ├── smoke_test_v11.py   # 冒烟测试
    └── logs/               # 训练日志与 ckpt
```

## 4. 和复旦的差异（都是有意为之）

| 项 | 复旦 | 我们 | 原因 |
|---|---|---|---|
| 机器人 URDF | `infantry_V4_increase.urdf` | `urdf/rl_chassis_train.urdf`（脚本生成） | 只保留 rear 链 |
| 训练总质量 | 7.86 kg | 7.3956 kg | 删掉前链，每腿少 0.20 kg（复旦训练模型同样缺并联杆质量） |
| kp / kd | 20 / 1.0 | 40 / 2.0 | v11 的腿更折：站 0.205 m 需要 ~7.5 N·m 膝力矩（复旦只需 ~3 N·m），kp=20 静差 0.38 rad 会把腿折进轮舱 |
| 默认角 | 装配姿态附近 | **重力静平衡点** `q_des + τ/kp` | 零动作即稳态，腿不会折穿 |
| θ0 口径 | `atan2(ey,ex)−π/2` | `atan2(ex,−ey)` | v11 零位即竖直向下，避免 ±π 跳变 |
| 高度指令范围 | [0.10, 0.20] m | [0.11, 0.19] m | 高度 = base_link 离地 = L0 + R_wheel − 0.11508；对应 L0 ∈ [0.165, 0.245] |
| num_envs | 8192（云 GPU） | 2048（RTX 4060 8G，实测 0.57 s/轮） | |
| 地形 | 部署那版用 trimesh 台阶 | 先 `plane` | 第一步先站住/走稳 |

其余（奖励权重、动作尺度 0.5/10、obs 25 维、history 5×25、latent 3、
`ActorCriticSequence`、decimation 2 / dt 0.005、域随机化）**与复旦逐项对齐**。

## 5. 怎么跑

```bash
cd rl_chassis/rl/plane_v11
export PYTHONPATH=$PWD:$PYTHONPATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH     # env_isaacgym

# 训练（本地 2048 环境约 0.57 s/轮，3000 轮 ≈ 25 min）
python wheel_legged_gym/scripts/train.py --task=wheel_legged --headless \
       --num_envs=2048 --max_iterations=3000 --run_name=v11_plane

# 导出 ONNX（sim2sim 用）
cd ..; python plane_v11/export_onnx/export_onnx.py \
       --log_root plane_v11/logs/wheel_legged_v11 \
       --load_run=v11_plane --checkpoint=3000 --out v11_plane.onnx
```

注：`wheel_legged_gym/__init__.py` 里加了 numpy>=1.24 的兼容补丁
（isaacgym Preview 4 的 `torch_utils.py` 用了被删掉的 `np.float`）。

## 6. 已知问题 / 待办

1. **PhysX 伪自碰撞**：base/thigh 的网格凸包会把轮舱、腿摆动的空腔填平，零动作静置时
   `base_link` 上会报 20~30 N 的接触力。复旦原版一样（他们 base 报 30~47 N，
   root_z 0.071），属于这套训练模型的固有现象，暂按复旦的做法不管；真机/部署模型用
   coacd 凸分解没有这个问题（v11 MJCF 已验证自碰撞 0 对）。若训练不稳可考虑给
   base_link 换 coacd 分解碰撞（注意 32 块 × 2048 env 的开销）。
2. **小毛病**：起步瞬间有约 0.1 m/s 的后退过渡（1 s 内收敛到 ~0）；站姿高度稳在
   0.1436 m（指令 0.15 m，差 6 mm）。属于 3000 轮的精度水平，多训/加跟踪权重可收紧。
3. **还没做**：台阶地形版本（对标复旦 `上台阶3`）、真机力矩上限核对（v11 MJCF ±40 N·m，
   训练 URDF 一致）、真机驱动器的轮子正负号核对（见第 7 节的注意事项）。

## 7. MuJoCo sim2sim（已完成，能跑）

## 8. 训练记录（三版，都是续训）

| 版本 | 从 | 轮数 | 地形 | 高度指令 | 备注 |
|---|---|---|---|---|---|
| `v11_plane`（v1） | 从头 | 3000 | 平地 | 0.11~0.19 | 基准版：平地站/走/转/调高都跟得上 |
| `v11_terrain`（v2） | v1 | 2500 | trimesh 混合地形 | 0.12~0.21 | 加地形；惩罚加大（torque ×4、action_rate ×5、action_smooth ×5、dof_acc ×4、dof_vel ×3、ang_vel_xy ×2、dof_pos_limits ×3）；加**动作延迟 0~10 ms 随机** + **执行器一阶滞后 15 ms**；kp/kd 随机化放宽到 ±20% |
| `v11_terrain_tall`（v3） | v2 | 2500 | 同上 | 0.16~0.27 | 抬高站姿：高度跟踪误差收到 ~1 cm；纯台阶地形 5~19 cm、256 环境×15 s 各档高度指令**零摔倒** |
| `v11_flat_slow`（v4，当前推荐） | v3 | 2000 | **平地** | 0.14~0.24 | 回到平地，专治"腿抖"：**kd 2→3**（轮 0.2→0.3、kd 随机化上限 ×1.4）+ 稳定性惩罚再加大一档（dof_vel ×2.7、dof_acc ×3、action_rate/action_smooth ×2.4、torques ×2、ang_vel_xy ×1.5、lin_vel_z ×2、nominal_state ×2、dof_pos_limits ×1.7）。抖指标（膝加速度 rms，MuJoCo 前进 0.6 m/s）从 v3 的 **0.22 m 处 209** 降到 **10**；速度跟踪 0.5/1.0/1.5/2.0 → 0.63/1.10/1.76/2.26 m/s；30 s 长跑各档站姿零摔 |

ONNX：`mujoco/v11_plane_3000.onnx`(v1)、`v11_terrain.onnx`(v2)、`v11_terrain_tall.onnx`(v3)、
`mujoco/v11_flat_slow.onnx`(v4)。
**平地作业/骑行推荐 v4**；需要上台阶时用 v3（v4 没训过地形），文档见 `mujoco/README_sim2sim.md`。

`mujoco/onnx_mj_v11.py`：把 ONNX 策略跑在 v11 的**闭链** MJCF 上，真机口径
（不用仿真里的膝角，而是像真机一样从 (前髋, 后髋) 反解虚拟膝）。

> **单片机移植看 [`mujoco/README_sim2sim.md`](mujoco/README_sim2sim.md)**：那里逐维写了
> 观测/动作的含义、单位、缩放、限幅，以及真机闭链映射的公式、几何常数和验证流程；
> 纯 numpy 的闭链运动学参考实现在 `tools/leg_kinematics_ref.py`（不依赖 MuJoCo）。

```bash
cd rl_chassis/rl/mujoco
# env_isaacgym 里已装好 mujoco 3.2.3 + onnxruntime（和 isaacgym/torch 同环境）
python onnx_mj_v11.py --onnx v11_plane_3000.onnx                    # 开窗口 + 键盘操控
python onnx_mj_v11.py --onnx v11_plane_3000.onnx --cmd-lin-vel 1.5 --duration 8 --no-viewer
```

键盘：按住 1 前进 / 2 后退 / 3 左转 / 4 右转，5 抬高 / 6 降低，0 停，b 卸力，q/Esc 退出。
鼠标可转视角，机位默认跟随整车。

> 用仓库根的 `.venv` 跑（而不是 `env_isaacgym`）时要补两个包：`onnxruntime` 和键盘后端。
> `pynput` 在 Linux 上的元数据依赖 `evdev`，而 `evdev` 要编译、需要 `python3-dev`；
> 没装头文件时用 `pip install --no-deps pynput python-xlib six` —— 键盘监听走 Xlib，
> 本来就不需要 `evdev`，装不上它不会影响键盘。

映射细节：

* 状态：(q_front, q_rear) 闭环逆解出虚拟膝角 q_shank，膝角速度由雅可比
  `∂q_shank/∂(q_front, q_rear)` 合成；
* 力矩：`τ_rear_real = τ_hip + τ_knee·∂q_shank/∂q_rear`，
  `τ_front_real = τ_knee·∂q_shank/∂q_front`（虚功原理），车轮力矩直通；
* 观测/动作尺度、kp/kd=40/2、默认角、指令范围都与训练配置逐项一致。

实测（`v11_plane_3000.onnx`，实时倍率约 6~8×）：

| 指令 | 实测 |
|---|---|
| 站立，height 0.15 | h = 0.1436 m，两轮着地，倾斜 1.4° |
| lin_vel +1.5 | +1.69 m/s |
| lin_vel −1.0 | −1.32 m/s |
| yaw_vel +1.5 | +1.37 rad/s |
| height 0.18 / 0.12 | 0.1804 / 0.1179 m |

踩过的两个坑（写下来免得再踩）：

1. **MuJoCo 自由关节角速度 `qvel[3:6]` 已经是机体系**，不要再旋转一次；Isaac 的
   `root_states[10:13]` 才是世界系。搞错的话策略直接失控。
2. **自由关节四元数是 (w,x,y,z)**。按 (x,y,z,w) 赋值等于给整车加 180° 偏航，
   现象是"策略看着倒着走"（我们的 MJCF 里 `Left_*` 在 +y、轴指向 +y，跟 URDF 一致，
   当时一度以为是轮子轴反了）。
