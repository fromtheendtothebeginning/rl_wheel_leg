# KaiWuRoboMaster

存放机器人强化学习（RL）模型资产的仓库：**URDF** 与 **MuJoCo MJCF**。

```
KaiWuRoboMaster/
├── rl_chassis/            机器人包：模型 + tools/（建模流水线）+ rl/（强化学习）
├── scripts/               仿真侧：MuJoCo 运行 / URDF·MJCF → USD / Isaac Sim 渲染 / 环境自检
├── config/project.ini     所有可调项：路径、PD 增益、渲染与场景参数
├── docker/                容器：MuJoCo + 工具链 / ROS 1 / Isaac Sim
├── docs/                  环境搭建记录（Isaac Sim / Isaac Lab / 驱动版本配套）
└── <robot>/               每个机器人一个包，各自一套完整描述
    ├── urdf/     URDF 描述（.urdf，含 SolidWorks 导出表 .csv）
    ├── mjcf/     MuJoCo 原生模型（.xml，含 scene、预览图、体检报告）
    ├── meshes/   STL 网格（`meshes/convex/` 里是凸分解出来的碰撞体）
    ├── config/   关节名等配置
    └── launch/   ROS 启动文件
```

> `COD-2026RoboMaster-Balance-Simulation_File-main/`（COD 战队的开源参考资产）只在本地保留，已 gitignore —— 里面有两个 27 MB 的 USD，不适合进版本库。代码里只把它当对照模型引用，不依赖这些文件本身。

## 快速开始

```bash
# MuJoCo：跑一遍并出图（参数全在 config/project.ini）
.venv/bin/python scripts/run_mujoco.py --headless --png out/mujoco.png

# 不想在本机装 Python 依赖就直接进容器
docker-compose -f docker/docker-compose.yml up --build mujoco

# Isaac Sim：先转 USD，再渲染（默认 MJCF 版，保留 4 条闭链）
$ISAAC_VENV/bin/python scripts/convert_robot_usd.py
OMNI_KIT_ACCEPT_EULA=YES $ISAAC_VENV/bin/python scripts/run_isaac_sim.py
```

环境怎么搭（驱动 / Isaac Sim / Isaac Lab 的版本配套、踩过的坑）见
[`docs/isaac_sim_setup.md`](docs/isaac_sim_setup.md)；容器见 [`docker/README.md`](docker/README.md)。

## 现有模型

| 目录 | 说明 |
|---|---|
| `rl_chassis/` | 步兵轮腿（v12）：在 v11 基础上加 4 个 POM 导轮 (`pom_wheel1_joint` … `pom_wheel4_joint`)。`urdf/rl_chassis_v12_std.urdf` 已做坐标/销轴/惯量修正，`urdf/rl_chassis_train.urdf` 是 RL 训练用的串联腿模型；`mjcf/rl_chassis_v12{,_weld}.xml` 各带 `_scene.xml`，6 个主动电机、4 个导轮为被动铰链、4 条闭链；`meshes/` 含凸分解；`tools/` 是建模流水线；`rl/` 是强化学习代码（见 `rl/README.md`） |

## 强化学习

`rl_chassis/rl/` 是 v11 的 RL 复现代码（对标 `fudan_rl_wheel_leg`）：**训练用串联腿、部署用闭链腿**
（Isaac Gym 不支持闭链，靠虚拟腿映射把 (髋, 膝) 力矩搬到真机的 (髋, 曲柄)）。
`plane_v11/` 是基于 wheel_legged_gym 的单腿训练，`mujoco/` 是 ONNX 策略的 sim2sim（含台阶/斜坡场景），
`tools/` 负责闭链标定与生成训练用 URDF。详见 `rl_chassis/rl/README.md` 与 `rl_chassis/rl/mujoco/README_sim2sim.md`。

仿真侧的运行脚本（跑 MuJoCo、URDF/MJCF → USD、Isaac Sim 渲染、环境自检）在 `scripts/`；
环境搭建记录（驱动 / Isaac Sim / Isaac Lab 版本配套）见 `docs/isaac_sim_setup.md`。

## 闭链轮腿模型的制作流程（成功流程）

目标：把 SolidWorks/sw2urdf 导出的 URDF，做成 MuJoCo 里**闭链精确闭合、关节轴正确、
每个 link 都有实体碰撞**、可以直接跑仿真/训练的 MJCF。每一步都有校验数字，
做不到就说明这步没做对。适用范围：每条腿是平行四边形/五连杆这类闭链、所有销轴与
髋轴平行的轮腿机器人（v6/v7/v9/v10/v11、COD-2026 都是这一类）。

> **一句话**：坐标系先对齐 → **把每个关节轴对到它的物理销孔上** → 从 STL 几何识别销轴、
> 在孔心放 site、用 `<connect>` 闭链 → 每个 link 凸分解出实体碰撞体 →
> 用残差 / 自由度秩 / 落地高度 / 阻挡测试四组数字验收。
>
> 坑都集中在第 ① ② 步：坐标不对、关节轴落错位置，后面怎么调都不对。

### 1. 处理顺序（照这个顺序跑，别跳）

```bash
# 以下命令在 rl_chassis/ 目录下执行（脚本都在它的 tools/ 里）
# ① 整体坐标系对齐 infantry_V4（x 前 / y 左 / z 上），所有 joint axis -> 0 0 1
python3 tools/align_frame.py  <robot>/urdf/<raw>.urdf \
        -o <robot>/urdf/<name>_std.urdf --fix-inertia

# ② 把每个父子关节的轴线挪到它的物理销孔上（sw2urdf 会把 origin 放错位置）
python3 tools/snap_pin_axes.py --urdf <robot>/urdf/<name>_std.urdf --check   # 先看
python3 tools/snap_pin_axes.py --urdf <robot>/urdf/<name>_std.urdf           # 再改（自动备份 .bak）

# ③ 每个 link 做凸分解碰撞体（v11：13 个 link 285 块 2.8 MB，约 90 s）
python3 tools/make_convex_collision.py --urdf <robot>/urdf/<name>_std.urdf

# ④ URDF -> MJCF（site/connect/圆柱轮/电机/keyframe/排除对）
python3 tools/make_mjcf_pins.py --urdf <robot>/urdf/<name>_std.urdf \
        --outdir <robot>/mjcf --base weld --name <name>    # 基座焊死，看/调机构
python3 tools/make_mjcf_pins.py --urdf <robot>/urdf/<name>_std.urdf \
        --outdir <robot>/mjcf --base free --name <name>    # 基座带 freejoint，训练用

# ⑤ 验收
python3 tools/check_mjcf_loop.py      --xml <robot>/mjcf/<name>_weld_scene.xml
python3 tools/check_self_collision.py --xml <robot>/mjcf/<name>_weld_scene.xml \
        --joint Left_front_joint --to -1.2
python3 tools/stance_sweep.py         --xml <robot>/mjcf/<name>_weld_scene.xml \
        --hold-seconds 5 --n 11 --span 1.0 --plot <robot>/mjcf/stance_workspace.png
```

脚本都在 `rl_chassis/tools/`。`make_mjcf_pins.py` 生成时会自动体检关节轴是否落在
销孔上，不在就直接告警并打印该跑的命令 —— **重新从 SolidWorks 导出后，第 ② 步一定要
重跑**（sw2urdf 每次都会把那几个 origin 导错）。

### 2. 每一步在解决什么问题

**① 坐标系对齐（`align_frame.py`）**

CAD 惯例是「横向 = X、向上 = −Y、向前 = Z」，MuJoCo/ROS 惯例是「x 前 / y 左 / z 上」，
映射是 `(x,y,z)_old -> (−z, x, −y)`。这一步顺便做三件事：

* 每个 link 的坐标系重投影，让所有 `joint axis = (0,0,1)`（左右镜像体现在 joint origin
  的 rpy 上，和 infantry_V4 一致）；关节正方向也对齐；
* `--fix-inertia`：SolidWorks 的惯性张量**缺平行轴项**。校验方法：按网格几何重算的
  惯量 vs URDF 里的值 —— v6/v9/v11 的底盘小了 6~13 倍，腿上的小零件约 1.7 倍，
  车轮/小块 ≈ 1.0（本来就对）。重算后总质量 7.8012 kg 与网格体积×密度吻合。

判据：所有 `joint/axis/@xyz = 0 0 1`；根关节 rpy = ∓90°；几何校验残差 < 1e-6 m。

**② 关节轴对到销孔（`snap_pin_axes.py`）★ 最容易踩的坑**

sw2urdf 有时会把 link 的坐标系放在它**远端**的销孔上，于是该 link 的 joint origin 也
落在远端，关节轴线和真正的销孔差十几厘米。装配位姿完全看不出来（网格摆位是对的），
但关节一转动，两个零件就在销孔处撕开：

| 关节（v11） | 轴线（错） | 真正销孔 | 转 0.3 rad 撕开 |
|---|---|---|---|
| `Left_front_child1_joint` | (−0.00202, +0.05229) | (+0.09805, +0.11508) | **35.3 mm** |
| `Left_front_child3_joint` | (−0.25369, +0.16301) | (−0.13869, +0.16274) | **34.4 mm** |
| `Right_front_child1_joint` / `Right_front_joint3_joint` | 镜像 | 镜像 | 35.3 / 34.4 mm |

脚本对每个「父子关节 + 两者存在同轴孔」的组合转 ±0.3 rad，量两孔中心的分开量；超过
0.5 mm 判为错位，于是 1) 把 joint origin 平移到销孔轴线（只改 (x,z)，y 不动）；
2) **反向补偿挂在 child 上的一切**：child 自己的 visual / inertial origin、以及 child
的所有子关节 origin。这样只动轴、不动几何（校验：网格世界位姿变化 < 0.001 µm，
惯量分量差 0）。同一关节上有 ≥3 个同轴孔对时自动判定为**螺栓孔圈**（轮毂），跳过不动。

> 修完**不要**再在销孔处补 `<connect>`：轴错的时候补，会把每条腿的自由度从 3 压到 2
> （约束秩 8→10）、驱动互相打架；轴对了以后补则是纯冗余（秩 8→8、自由度 6→6，
> site 恰好落在转动轴线上，雅可比那几行为 0）。

**③ 闭链销轴识别 + site（`pin_loops.py`）**

闭链的 site **必须落在销轴轴线上**。把 site 放在"两网格最近点"是错的：最近点很容易落在
平面或倒角上，站点一偏，`<connect>` 就变成把两个零件硬拽在一起，多出来的自由度会把腿
锁死或乱飞。这类 SolidWorks 零件是**棱柱件**（法向只有平行/垂直于销轴两个方向），于是：

1. 丢掉上下大平面，只留侧壁，投影到垂直于销轴的平面上；
2. 一个圆孔在投影里就是**一圈彼此相连的侧壁**（它和零件其它部分只在上下平面处相连，
   而那部分已被丢掉）→ 按共享边做并查集，把每个孔单独切出来；
3. 每个多边形环做最小二乘圆拟合，圆度 `rms/r` 达标才算圆孔；
4. 两个零件上**位置重合**的孔 = 一条销轴（v11 识别出 4 条：`child2↔rear_link`、
   `child3↔rear_child1`，与 COD 官方 MJCF 的配对完全一致）。

销轴上取一点，分别在两个 link 的坐标系里写成 site：

```xml
<site name="pin1_A" pos="…" size="0.005"/><!-- 在 Left_front_child2_link 上 -->
<site name="pin1_B" pos="…" size="0.005"/><!-- 在 Left_rear_link 上 -->
<connect name="L_loop1" site1="pin1_A" site2="pin1_B"
         solref="0.002 1" solimp="0.99 0.99 0.01 0.5 2"/>
```

**y 坐标要对齐**：这类机构所有关节轴都平行（垂直于机构平面），所以任何一条 connect 的
y 分量在运动中是**常量**，两个 site 的 y 不同就等于埋了一个恒定约束误差。做法：site 的
世界 y 取两零件孔中心的中值，两个 site 用**同一个世界点**转成各自局部坐标（脚本已做）。

**④ 每个 link 的实体碰撞（`make_convex_collision.py`）**

MuJoCo 的网格碰撞体是**凸包**，而这些零件都不凸 —— 用「凸包体积 / 真实体积」一量就知道
多离谱：

| 零件 | 凸包/真实 | 零件 | 凸包/真实 |
|---|---|---|---|
| `base_link` | **22.7×** | 曲柄 `front_Link` | 5.6× |
| 大腿 `rear_link` | 3.7× | 小腿 `rear_child1` | 2.9× |
| 连杆 `child2` | 1.77× | 车轮 | 1.96× |

直接拿原网格开自碰撞，底盘会变成实心大盒子，腿在它肚子里全是伪碰撞 —— 早期版本只能把
`conaffinity` 设成 0（代价就是腿会穿过底盘）。现在的做法：

* 用 **coacd** 把每个 link 凸分解成若干块（13 个 link、285 块、2.8 MB），每块写成独立
  STL 放在 `meshes/convex/`，每块一个 geom 参与碰撞（`contype=1 conaffinity=1`），
  原始网格只做显示；
* **车轮不做分解**：碰撞体是精确的解析圆柱（半径 0.06、半宽 0.0292），滚动才光滑；
* 接触刚度比默认硬：`solref="0.008 1"`、`solimp="0.95 0.99 0.001"`
  （MuJoCo 默认 solref 0.02 偏软，硬怼时会压进去 8 mm）；
* `<exclude>` 只出现在两类地方：1) 本来就铰接/销连在一起的零件对（父子关节 + 4 条闭链
  销轴）；2) **装配位姿下就已经互相顶住的零件对** —— 凸分解只能逼近（每块是凸包，会比
  真实形状胖一点），紧配合的零件会互相插进去。生成脚本自动找出（压入 > 0.2 mm）并打印
  名单，v11 上是 `front_Link↔rear_link`（两个髋部同心圆环，孔被凸包填了）、
  `Wheel↔front_child1_link`。

**⑤ 生成 MJCF（`make_mjcf_pins.py`）**

直接从 URDF 机械式转换（body pos/quat = joint origin，geom quat = visual rpy），不重新
猜几何，另外做这些事：

* 两条闭链各 2 个 site（共 8 个）+ 4 条 `<connect>`；
* 车轮碰撞体换成解析圆柱，网格只显示；
* 6 个力矩电机（每腿：前髋 = 连杆根部、后髋 = 大腿根部、轮）；
* `--base weld`：`base_link` 这个 body 上**不加 joint**（焊死在世界系），整机抬高 `h`
  让车轮解析圆柱最低点正好落在 z=0（v11 是 +0.079926 m）；
* `--base free`：加 `<freejoint>`，并且 **keyframe 里必须写基座位姿**（`0 0 h 1 0 0 0 …`）
  —— 否则 `mj_resetDataKeyframe` 会把基座拉回原点，车轮掉到地面以下 80 mm；
* 两遍生成：先写一份不带额外排除的，量出装配位姿互相顶住的零件对，再带上 `<exclude>` 重写。

### 3. 验收标准（四组数字，缺一不可）

| 检查 | 命令 | 达标线（v11 实测） |
|---|---|---|
| 闭链闭合 | `check_mjcf_loop.py` | 装配位姿残差 **0.004 µm**（≈0） |
| 自由度 | 同上（约束雅可比奇异值） | 每腿 **3**（2 个闭链 DOF + 1 轮），秩 8，谱 8 个大值后直接掉 8 个量级 |
| 落地 | 同上 | 车轮碰撞体最低点 **z = 0.000 mm**（焊死版），不穿模 |
| 保位 | `stance_sweep.py` | PD 压 5 s：闭链残差 **2.5 µm**、驱动漂移 **0.17°**、轮子法向力 ~7 N |
| 站姿可达 | 同上 | 121 个 (前髋, 后髋) 目标里 **52** 个能让轮子落在地面上，轮心 x ∈ [−0.13, +0.10] m |
| 自碰撞 | `check_self_collision.py` | 装配位姿自碰撞 **0 对**；曲柄硬压到 −1.2 rad 时停在 **−0.36 rad**（早期版本一路压到底 = 零件穿过去了） |
| 关节轴 | `snap_pin_axes.py --check` | 所有真实销孔「转 0.3 rad 撕开 **0.000 mm**」 |

原始输出见 [`rl_chassis_v11/mjcf/verify_report.txt`](rl_chassis_v11/mjcf/verify_report.txt)。

**为什么"每腿 3 个自由度"是对的**：平面机构 Kutzbach，7 个构件（6 个动件 + 机架）、8 个
转动副 → `3(7−1) − 2×8 = 2` 个闭链自由度；`<connect>` 是球铰，在平面机构里每个只贡献
2 个独立约束，两条闭链 = 4 个约束 → 6 个闭链关节 − 4 = 2。再加轮子自转 = 3。驱动就是
前髋 + 后髋（和 COD 官方 MJCF 一致）。

诊断小技巧：**残差不随步长变小**（0.001 → 0.00025 都是 0.10 mm）说明它是软约束在重力
载荷下的正常挠度，不是离散化误差；残差只出现在 (x,z) 而 y 恒为 0，就印证了上面那条
「所有关节轴平行 → y 行退化」。

### 4. 换一个新机器人时的清单

1. `align_frame.py` 对齐坐标系 + 重算惯量，检查总质量与网格体积吻合；
2. `snap_pin_axes.py --check`：**只要它报出关节，就一定要先修**；
3. `pin_loops.py` 核对销轴配对是否符合机构图（每腿应有：上销 = 连杆↔大腿、下销 =
   连杆↔小腿；父子关节上也有同轴孔对，那是铰链不是闭链，别重复加 connect）；
4. `make_convex_collision.py` + `make_mjcf_pins.py`（weld / free 两份）；
5. 跑第 3 节那四组验收；
6. 想真拿去训练，再把关节行程换成 CAD 真实限位（sw2urdf 导出的是 continuous ±6.28，
   等于没限位），并按需要调 `solref / solimp / armature`。

**v12 这种"在旧版上只加导轮"的导出**：先跑一次 `tools/repair_export.py`（补零轴、把按 CAD
全局坐标导出的导轮网格移到轮轴中心），再按上面的 ①～⑤ 执行。生成 MJCF 时用
`--passive pom_wheel1_joint ...` 把导轮排除在电机之外；导轮仍保留实体圆柱碰撞，
所以斜坡/台阶时能真实接触。

```bash
python3 tools/repair_export.py \
  --urdf urdf/rl_chassis_v12.urdf \
  --base-mesh meshes/base_link_full.STL \
  --copy-inertial Right_rear_link \
  -o urdf/.rl_chassis_v12_repaired.urdf
python3 tools/align_frame.py urdf/.rl_chassis_v12_repaired.urdf \
  -o urdf/rl_chassis_v12_std.urdf --fix-inertia --relative-meshes
python3 tools/snap_pin_axes.py --urdf urdf/rl_chassis_v12_std.urdf
python3 tools/make_convex_collision.py --urdf urdf/rl_chassis_v12_std.urdf
python3 tools/make_mjcf_pins.py --urdf urdf/rl_chassis_v12_std.urdf \
  --outdir mjcf --base weld --name rl_chassis_v12 \
  --passive pom_wheel1_joint pom_wheel2_joint pom_wheel3_joint pom_wheel4_joint
python3 tools/make_mjcf_pins.py --urdf urdf/rl_chassis_v12_std.urdf \
  --outdir mjcf --base free --name rl_chassis_v12 \
  --passive pom_wheel1_joint pom_wheel2_joint pom_wheel3_joint pom_wheel4_joint
```

### 5. 踩过的工具坑（写下来免得再踩）

* **MuJoCo 网格碰撞体 = 凸包**，不凸的零件必须凸分解，否则只能用 `conaffinity=0` 关掉
  自碰撞（代价就是穿模）；
* `<inertial>` 的 `fullinertia` 和 `quat` **不能同时给**。URDF 的惯性张量是在 inertial
  自己的坐标系里，要写成 `fullinertia` 就得先旋转到 body 系（`I_body = R I Rᵀ`）；
* URDF 根 link 若直接转 MuJoCo（`MjSpec.from_file`），不给 freejoint 会**丢掉底盘质量**
  （7.73 → 2.12 kg）；手写 MJCF 里"没有 joint 的 body"则会正常计入；
* 二进制 STL 每条记录 50 字节 = 法向 12 + 三顶点 36 + **属性 2**，属性必须逐条内嵌
  （把 n×2 字节一次性追加到文件尾会让所有记录错位，MuJoCo 报 "exceeds maximum bounds"）；
  写完务必按 `84 + n*50` 校验长度并回读数据；
* free 基座的 keyframe 一定要写基座位姿（见 ⑤）；
* 车轮碰撞体用解析圆柱：网格轮的最低顶点比轮心低 1.7 mm，会让平衡点带偏置；左右轮要按
  **关节轴**定义圆柱（圆心只取轴向分量、半径取绕轴最大距离），否则用网格 bbox 左右差
  0.17 mm；
* 凸分解块放进 `group="3"` 并给半透明颜色，写渲染脚本时记得 `opt.geomgroup[3]=0`，
  否则预览图里全是绿块。

## 预览

| 预览 | 文件 |
|---|---|
| 整机（iso） | `rl_chassis_v11/mjcf/preview_iso.png` |
| 正视图（左右对称、轮子触地） | `rl_chassis_v11/mjcf/preview_front.png` |
| 站姿可达域扫描 | `rl_chassis_v11/mjcf/stance_workspace.png` |
| 关节轴修正前：`Left_front_child1_joint` 转 0.3 rad，销孔撕开 35.3 mm | `rl_chassis_v11/mjcf/preview_pin1_before.png` |
| 修正后：两点重合（0.000 mm） | `rl_chassis_v11/mjcf/preview_pin1_after.png` |
| 修正前：`Left_front_child3_joint`，撕开 23.0 mm | `rl_chassis_v11/mjcf/preview_pin2_before.png` |
| 修正后：重合 | `rl_chassis_v11/mjcf/preview_pin2_after.png` |
