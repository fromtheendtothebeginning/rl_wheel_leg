# Isaac Sim / Isaac Lab 环境搭建（本机实测）

记录这台机器（Ubuntu 22.04 + RTX 5060 Laptop，Blackwell / sm_120）上把 Isaac Sim 和
Isaac Lab 跑起来的过程。**版本三者必须配套**，这是折腾最久的地方。

## 结论：当前可用组合

| 组件 | 版本 | 位置 |
| --- | --- | --- |
| NVIDIA 驱动 | **595.99.02**（**open** 内核模块） | 系统 |
| Isaac Sim | **6.2.0.0** | `~/IsaacLab-3.0/.venv`（Python 3.12） |
| Isaac Lab | **3.0.0** | 同上 |
| torch | **2.11.0+cu130**（arch 含 sm_120） | 同上 |

验证方式：`scripts/check_env.py`（驱动 → 版本 → RTX 渲染冒烟）。实测渲染出
`rgb shape=(480, 640, 4)`、显存 1.56 GB / 8.08 GB；训练 `Isaac-Cartpole-Direct` 跑通。

## 三条铁律

### 1. Blackwell 必须用 **open** 内核模块

闭源模块根本初始化不了 RTX 50 系，`nvidia-smi` 会报 "No devices were found"，内核日志里是：

```
NVRM: installed in this system requires use of the NVIDIA open kernel modules.
NVRM: GPU 0000:01:00.0: RmInitAdapter failed! (0x22:0x56:1017)
```

所以要装 `nvidia-driver-<ver>-open`。**别被 sysfs 的 `power/control=auto` 误导** ——
那是表象，设备压根没初始化成功。

### 2. 驱动分支决定哪个 Isaac Sim 版本能用

| Isaac Sim | 期望驱动 | 另一个驱动上的表现 |
| --- | --- | --- |
| **6.2** | **≥ 595.58.03** | 580 上：物理能跑，**渲染静默返回空帧**（`rgb shape=(0,)`） |
| 5.1 | 580.x | 595 上：启动时**段错误**，崩在 `librtx.scenedb.plugin.so` |

所以换 Isaac Sim 版本前先确认驱动分支，反之亦然。切换命令（都要重启）：
```bash
sudo apt-get install -y nvidia-driver-595-open linux-modules-nvidia-595-open-$(uname -r)   # 走 6.2
sudo apt-get install -y nvidia-driver-580-open linux-modules-nvidia-580-open-$(uname -r)   # 走 5.1
```

### 3. Isaac Lab 必须配 Isaac Sim 的对应版本

| Isaac Lab | 锁定的 Isaac Sim | Python |
| --- | --- | --- |
| **3.0**（GitHub main） | **6.2** | 3.12 |
| 2.3.2（PyPI 最新） | 5.1 | 3.11 |

两个 Python 版本互斥，所以 5.1 和 6.2 不能共用一个环境。**PyPI 上的 `isaaclab` 发布滞后**
（停在 2.3.2，仍锁 5.1），3.0 只在 GitHub 上 —— 这就是为什么"PyPI 上找不到支持 6.2 的
Isaac Lab"，但 main 分支其实已经支持了。

## 装 Isaac Lab 3.0（uv）

```bash
git clone --depth 1 https://github.com/isaac-sim/IsaacLab.git ~/IsaacLab-3.0
python3 -m pip install --user uv
cd ~/IsaacLab-3.0 && uv sync --extra isaacsim
```

`uv sync --extra isaacsim` 解析出 `isaacsim[all,extscache]==6.2.0.0` + `torch==2.11.0`
（CUDA 13.0 源）+ Python 3.12。**这条路不用手动换 torch**。

### 3.0 的 CLI 和 2.x 不一样

```bash
cd ~/IsaacLab-3.0
OMNI_KIT_ACCEPT_EULA=YES .venv/bin/isaaclab train --rl_library rsl_rl --task Isaac-Cartpole-Direct --max_iterations 20
```

- **没有 `--headless` 了**：改成 `--viz/--visualizer` 选可视化器，默认就是 headless。
- 任务名变了：`Isaac-Cartpole-v0` → `Isaac-Cartpole-Direct`（用 `isaaclab list_envs` 查）。
- 默认物理后端是 Newton；要 Isaac Sim 走 `--extra isaacsim`。
- 其它：`--num_envs`、`--device`、`--video`。

## 两个会误导人的坑

### torch 的 sm_120 内核（只在 5.1 那条线遇到）

Isaac Sim 5.1 默认装的 torch 是 cu126 构建，`torch.cuda.get_arch_list()` 里只有
`sm_50…sm_90`，**没有 sm_120**，训练一跑就报：

```
CUDA error: no kernel image is available for execution on the device
```

必须换 cu128 构建。**注意 pip 会骗你**：`pip install torch==2.7.0 --index-url .../cu128`
看到版本号相同就输出 "Requirement already satisfied" 然后什么都不做，必须**先 uninstall**。

（6.2 + Isaac Lab 3.0 这条路的 torch 是 cu130，天然带 sm_120，没这个问题。）

### 在线资产服务器取不到

NVIDIA 的 `omniverse-content-production.s3-us-west-2.amazonaws.com` 在本机返回 404，
所以 `add_default_ground_plane()` 会抛 `Could not find assets root folder`。
`scripts/scene_utils.py` 里的地面/材质/灯光全部用本地 USD prim 手搭，绕开它。
headless 下**没有默认光照**，不加灯渲染出来是纯黑。

## 磁盘

Isaac Sim 全套约 12–25 GB（取决于 extras）。注意 micromamba / uv 都用**硬链接**，
`du` 会把同一份数据在环境目录和包缓存里各算一次 —— 删环境时 `du` 显示 44 GB，
实际可能只释放 8 GB。`uv cache clean` 也基本回收不了什么（都是硬链接）。
