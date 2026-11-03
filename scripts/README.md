# scripts/

仿真侧的运行脚本。和 `rl_chassis/tools/` 里那套 **建模流水线**（坐标对齐 → 销孔 →
凸分解 → 生成 MJCF → 体检）是两件事：那边生产模型，这边把模型丢进仿真器跑。

**所有可调项都在 [`config/project.ini`](../config/project.ini)**：路径、PD 增益、
渲染分辨率、灯光强度、取景余量。脚本里不再写死数字，命令行参数可以覆盖配置；
`--config other.ini` 就能换一套参数（比如做对照实验）。

## 需要什么

| 脚本 | 跑在哪个环境 | 依赖 |
| --- | --- | --- |
| `run_mujoco.py` | 仓库根的 `.venv`（或 Docker） | mujoco、numpy、Pillow |
| `run_isaac_sim.py` / `smoke_isaac_sim.py` | 装了 Isaac Sim 的环境（配置里的 `[paths] isaac_venv`） | isaacsim、trimesh、Pillow |
| `convert_robot_usd.py` | 同上 | urdf-usd-converter / mujoco-usd-converter（随 Isaac Sim 带） |
| `check_env.py` | 系统 python 即可 | 只调 subprocess |

`requirements.txt` 是 MuJoCo + 流水线那套的依赖。环境怎么搭、版本怎么配（驱动 +
Isaac Sim + Isaac Lab 三者对应关系）见 [`../docs/isaac_sim_setup.md`](../docs/isaac_sim_setup.md)。

## MuJoCo

```bash
.venv/bin/python scripts/run_mujoco.py --view                        # 开窗口看机构
.venv/bin/python scripts/run_mujoco.py --headless --png out/mujoco.png
```

会打印闭环残差、基座高度、竖直度。`--wheel 5` 给轮子常值力矩能看它往前跑。

## Isaac Sim

```bash
# 1) 机器人转 USD（MJCF weld / MJCF free / URDF 三份）
$ISAAC_VENV/bin/python scripts/convert_robot_usd.py

# 2) 丢进 Isaac Sim 渲染一张（默认用 MJCF weld 版，它保留 4 条闭链）
OMNI_KIT_ACCEPT_EULA=YES $ISAAC_VENV/bin/python scripts/run_isaac_sim.py

# 3) 想知道渲染器本身工不工作，跑最小冒烟
OMNI_KIT_ACCEPT_EULA=YES $ISAAC_VENV/bin/python scripts/smoke_isaac_sim.py

# 4) 环境四项自检（驱动 / torch 的 sm_120 / 版本 / 渲染）
python3 scripts/check_env.py
```

`$ISAAC_VENV` 就是 `config/project.ini` 里的 `[paths] isaac_venv`，默认 `~/IsaacLab-3.0/.venv`。

## Docker

不装 Python 依赖，直接进容器：

```bash
docker-compose -f docker/docker-compose.yml up --build mujoco
```

见 [`../docker/README.md`](../docker/README.md)。

## 两条转换路径的取舍

| | MJCF 路 | URDF 路 |
| --- | --- | --- |
| 层级 | link 平铺在一个 `Geometry` Scope 下 | base_link 下嵌套成树 |
| **闭链** | **保留**（4 个 `<connect>` → `PhysicsSphericalJoint`） | 丢掉（URDF 表达不了） |
| 电机 | 6 个 `MjcActuator` | 无 |

**要做这台车的仿真，走 MJCF 路** —— 闭链就是它的核心机构。URDF 路只在需要一棵干净的
运动学树时用（例如做 IK / 可视化）。

## 出图上的已知瑕疵

STL 是 SolidWorks 导出的薄板开口壳体，没有法线。`scene_utils.repair_robot_meshes()` 会用
trimesh 算平面法线写回、设 doubleSided、绑材质，能压掉大部分黑斑，但开放壳体判断不了
内外，**根除得修网格本身**。物理和训练不受影响。

## 踩过的坑（都写在代码注释里）

- 场景里**不能**用 `add_default_ground_plane()`：它要去 NVIDIA 在线资产服务器拉贴图，本机 404。
- headless 下**没有默认光照**，不加灯渲染是纯黑。
- 相机要按包围盒自动取景，手写机位容易贴太近、看着像"模型散架"。
- 量闭环残差必须量 **joint 两端锚点**，量两个 body 的原点没有意义。
- 厘米级场景别铺太大的地面，阴影贴图分辨率会被浪费掉。
