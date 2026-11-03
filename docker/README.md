# Docker

把项目拆成三类容器，各管一件事。**代码都是运行时挂载**（`..:/workspace`），改代码不用重建镜像。

| 服务 | 镜像 | 用途 | 大小 | 状态 |
| --- | --- | --- | --- | --- |
| `mujoco` | 自建 `kaiwu-robomaster:mujoco` | MuJoCo 跑模型、渲染 PNG | 1.1 GB | **已验证** |
| `tools` | 同上 | 跑 `rl_chassis/tools/` 里的建模体检 | 同上 | **已验证** |
| `ros` | `osrf/ros:noetic-desktop-full` | rviz 看 URDF（ROS 1，匹配仓库里的 catkin 包） | ~4 GB | 未验证（需 X11） |
| `isaac` | NVIDIA 官方 `nvcr.io/nvidia/isaac-sim:6.2.0` | Isaac Sim（GPU） | ~15 GB | 未验证（需 GPU） |

> 注意 `docker.io` 不带 Compose v2，Ubuntu 源里只有 v1（`docker-compose`）。
> 装了 v2 的话把命令里的 `docker-compose` 换成 `docker compose`。

## 快速开始

```bash
# 1) MuJoCo 跑一遍仿真：打印闭环残差，并在 out/ 出一张 PNG
docker-compose -f docker/docker-compose.yml up --build mujoco

# 2) 建模体检（闭环残差 / 自由度 / 落地高度）
docker-compose -f docker/docker-compose.yml run --rm tools
```

不给 compose 也行：

```bash
docker build -f docker/Dockerfile.mujoco -t kaiwu-robomaster:mujoco .
docker run --rm -v "$PWD":/workspace --user "$(id -u):$(id -g)" -e HOME=/tmp \
    kaiwu-robomaster:mujoco scripts/run_mujoco.py --headless --png out/mujoco.png
```

`--user` 是为了让 `out/` 里的产物属于你而不是 root。

## ROS 1（rviz）

仓库里的 `package.xml` / `CMakeLists.txt` 是 **catkin（ROS 1）**，`launch/` 也是 ROS 1 写法。
Ubuntu 22.04 上装不了 Noetic，容器里反而最省事（镜像是 20.04 底）。

```bash
docker build -f docker/Dockerfile.ros -t kaiwu-robomaster:ros .
xhost +local:docker                     # 放行 X11
docker-compose -f docker/docker-compose.yml run --rm ros
# 进容器后：
cd /ws && catkin_make && source devel/setup.bash
roslaunch rl_chassis display.launch
```

> `display.launch` 里 rviz 的配置指向 `urdf.rviz`，仓库里没有这个文件（SolidWorks 导出继承
> 下来的老问题），会报找不到配置。要么去掉 `-d`，要么自己存一份 rviz 配置。

## Isaac Sim

不自建镜像：Isaac Sim 依赖 25 GB 且和显卡驱动强绑定，**用 NVIDIA 官方镜像**。

前提：宿主机装好 NVIDIA 驱动（Blackwell / RTX 50 系必须用 **open** 内核模块，
见 [`../docs/isaac_sim_setup.md`](../docs/isaac_sim_setup.md)）+ `nvidia-container-toolkit`。

```bash
docker-compose -f docker/docker-compose.yml run --rm isaac
# 容器里用 Isaac Sim 自带的 python：
/isaac-sim/python.sh /workspace/scripts/run_isaac_sim.py --steps 60
```

容器内 `config/project.ini` 的 `[paths] isaac_venv`（默认宿主机上的 `~/IsaacLab-3.0/.venv`）
是不存在的，需要覆盖：

```bash
/isaac-sim/python.sh /workspace/scripts/convert_robot_usd.py --isaac-venv /isaac-sim
```

（转换器在镜像里的位置以镜像版本为准，`--isaac-venv` 指到它的父目录即可。）

**Isaac Lab 3.0 训练没有放进 Docker**：它是 `uv` 管理的独立环境（约 25 GB），
且必须和驱动分支配套。换机器按 [`../docs/isaac_sim_setup.md`](../docs/isaac_sim_setup.md) 重装更可靠。

## 常见问题

- **`--png` 报 EGL 初始化失败**：镜像里已装 `libegl1 libegl-mesa0 libosmesa6`；
  自定义镜像要补上，或改用 `MUJOCO_GL=osmesa`。
- **rviz 起不来**：检查宿主机 `DISPLAY`，并 `xhost +local:docker`。
- **out/ 里的文件变成 root 所有**：少了 `--user "$(id -u):$(id -g)"`。
- **Isaac 里渲染全黑**：八成是驱动版本和 Isaac Sim 版本不配套，
  在宿主机跑 `python3 scripts/check_env.py` 看（见 `../docs/isaac_sim_setup.md`）。
