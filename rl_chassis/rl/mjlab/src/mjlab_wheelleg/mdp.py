"""轮腿任务的 MDP 组件：动作项、观测、奖励、终止、指令、指标。

数值全部对齐 Isaac Lab 版 ``rl/isaaclab/wheel_env.py``（v12 stage1 平地任务），
但动作接口按真机口径实现：

* 策略输出 6 维 = [左髋(虚拟), 左膝(虚拟), 左轮, 右髋, 右膝, 右轮]，
  髋/膝是「串联腿口径」的位置目标（默认角 + 0.5·a），轮是速度目标（10·a）；
* 逐物理步 PD 在虚拟空间算出虚拟 (髋, 膝) 力矩，再用闭链雅可比
  ∂q_shank/∂(q_front, q_rear) 按虚功原理映射到真机 (后髋, 前髋曲柄) 力矩；
* 力矩限幅（虚拟空间，随机化 0.8~1.1×）→ 一阶执行器滞后（真机空间，15 ms）→
  写 6 个电机 ctrl（MJCF ctrlrange ±40/±20 兜底）；
* 动作延迟 1 个策略周期（fifo）。

观测/奖励里用的 q_rear / q_shank 直接读闭链仿真的真机关节角——仿真本身保证
闭链一致，不需要部署时那套牛顿逆解。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.manager_base import ManagerTermBaseCfg
from mjlab.managers.termination_manager import TerminationTermCfg

from .leg_torch import (
    CLOSED_DEFAULT,
    JOINT_NAMES,
    MOTOR_NAMES,
    VIRTUAL_DEFAULT,
    WHEEL_JOINT_NAMES,
    LegTorch,
)

HIP_KNEE_DIMS = [0, 1, 3, 4]


# ---------------------------------------------------------------------------
# 指令：(vx, wz, height)，5 s 重采样，50% 概率站定
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class LocomotionCommandCfg(CommandTermCfg):
  lin_vel_range: tuple[float, float] = (-1.2, 1.2)
  ang_vel_range: tuple[float, float] = (-1.0, 1.0)
  height_range: tuple[float, float] = (0.14, 0.24)
  standing_prob: float = 0.5

  def build(self, env) -> LocomotionCommand:  # noqa: ANN001
    return LocomotionCommand(self, env)


class LocomotionCommand(CommandTerm):
  def __init__(self, cfg: LocomotionCommandCfg, env):  # noqa: ANN001
    super().__init__(cfg, env)
    self._vel = torch.zeros(self.num_envs, 3, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self._vel

  def _update_metrics(self) -> None:
    self.metrics["standing"] = (torch.linalg.norm(self._vel[:, :2], dim=1) < 1e-6).float()

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    n = len(env_ids)
    cfg: LocomotionCommandCfg = self.cfg
    stand = torch.rand(n, device=self.device) < cfg.standing_prob
    vx = torch.empty(n, device=self.device).uniform_(*cfg.lin_vel_range)
    wz = torch.empty(n, device=self.device).uniform_(*cfg.ang_vel_range)
    height = torch.empty(n, device=self.device).uniform_(*cfg.height_range)
    vx = torch.where(stand, torch.zeros_like(vx), vx)
    wz = torch.where(stand, torch.zeros_like(wz), wz)
    self._vel[env_ids] = torch.stack((vx, wz, height), dim=-1)

  def _update_command(self, env_ids: torch.Tensor | None) -> None:
    pass


# ---------------------------------------------------------------------------
# 动作项：虚拟 PD + 闭链力矩映射 + 执行器滞后 + 动作延迟
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class WheelLegActionCfg(ActionTermCfg):
  entity_name: str = "robot"

  kp: float = 40.0
  kd: float = 3.0
  wheel_kd: float = 0.3
  pos_action_scale: float = 0.5
  vel_action_scale: float = 10.0
  effort_limit: tuple[float, ...] = (40.0, 40.0, 20.0, 40.0, 40.0, 20.0)
  actuator_tau: float = 0.015
  delay_steps: int = 1

  # 启动时一次的控制器随机化（对齐 plane_v11 domain_rand）
  kp_range: tuple[float, float] = (0.8, 1.2)
  kd_range: tuple[float, float] = (0.7, 1.4)  # 部署端可能上调 kd 压抖
  effort_range: tuple[float, float] = (0.8, 1.1)

  def build(self, env) -> WheelLegAction:  # noqa: ANN001
    return WheelLegAction(self, env)


class WheelLegAction(ActionTerm):
  """详见模块 docstring；动作维 6。"""

  cfg: WheelLegActionCfg

  def __init__(self, cfg: WheelLegActionCfg, env):  # noqa: ANN001
    super().__init__(cfg, env)
    entity = self._entity
    n = self.num_envs
    dev = self.device

    # 关节索引（entity 内部序）：每腿 (front, c1, c2, c3, rear, shank) + 轮
    self._leg_idx = {}
    self._wheel_idx = {}
    for side in ("left", "right"):
      ids, names = entity.find_joints(JOINT_NAMES[side], preserve_order=True)
      assert tuple(names) == JOINT_NAMES[side], f"{side}: 关节解析 {names}"
      self._leg_idx[side] = torch.tensor(ids, device=dev)
      (wid,), wnames = entity.find_joints((WHEEL_JOINT_NAMES[side],))
      assert wnames[0] == WHEEL_JOINT_NAMES[side]
      self._wheel_idx[side] = wid

    # 电机 ctrl 索引（XML motor，ctrl = 力矩）
    ctrl_ids, act_names = entity.find_actuators(list(MOTOR_NAMES), preserve_order=True)
    assert tuple(act_names) == MOTOR_NAMES, f"电机解析 {act_names}"
    self._ctrl_ids = torch.tensor(ctrl_ids, device=dev)

    # 被驱动关节的 entity 局部索引（与 tau 列序 [F,R,W]x2 一致）。
    # 力矩必须经 set_joint_effort_target 下发：env 每个子步 write_data_to_sim()
    # 会用 actuator 内部 target 缓冲重算并覆盖 ctrl，直接写 ctrl 会被清零。
    driven = (
        "Left_front_joint",
        "Left_rear_joint",
        "Left_Wheel_joint",
        "Right_front_joint",
        "Right_rear_joint",
        "Right_Wheel_joint",
    )
    ids, jnames = entity.find_joints(list(driven), preserve_order=True)
    assert tuple(jnames) == driven, f"驱动关节解析 {jnames}"
    self._driven_joint_ids = torch.tensor(ids, device=dev)

    # 真机电机力矩上限（MJCF ctrlrange）
    self._motor_limit = torch.tensor(
      (40.0, 40.0, 20.0, 40.0, 40.0, 20.0), device=dev
    ).repeat(n, 1)

    self._legs = {s: LegTorch(s) for s in ("left", "right")}

    # 虚拟默认角（策略口径）：[L髋, L膝, L轮(0), R髋, R膝, R轮(0)]
    self._default_virtual = torch.tensor(
      (
        VIRTUAL_DEFAULT["left"][0],
        VIRTUAL_DEFAULT["left"][1],
        0.0,
        VIRTUAL_DEFAULT["right"][0],
        VIRTUAL_DEFAULT["right"][1],
        0.0,
      ),
      device=dev,
    ).repeat(n, 1)

    # 启动随机化（每个 env 采样一次，不随 reset 变）
    def ur(lo, hi, shape=(n, 1)):
      return torch.empty(*shape, device=dev).uniform_(lo, hi)

    self._kp = cfg.kp * ur(*cfg.kp_range)
    self._kd = cfg.kd * ur(*cfg.kd_range)
    self._wheel_kd = cfg.wheel_kd * ur(*cfg.kd_range)
    self._effort_limit = (
      torch.tensor(cfg.effort_limit, device=dev).repeat(n, 1)
      * ur(*cfg.effort_range, (n, 6))
    )

    # 动作 fifo（策略周期整数倍延迟）
    self._fifo: list[torch.Tensor] = [
      torch.zeros(n, 6, device=dev) for _ in range(max(0, cfg.delay_steps))
    ]
    # 本策略周期实际下发的动作（= 延迟后的），观测用它
    self._applied_actions = torch.zeros(n, 6, device=dev)
    # 执行器滞后后的真机力矩（奖励里罚力矩用它）
    self._tau_real = torch.zeros(n, 6, device=dev)
    # 最近一次的闭链雅可比（诊断用）
    self._d = {
      s: (torch.zeros(n, device=dev), torch.zeros(n, device=dev))
      for s in ("left", "right")
    }

  @property
  def action_dim(self) -> int:
    return 6

  @property
  def raw_action(self) -> torch.Tensor:
    return self._applied_actions

  @property
  def applied_actions(self) -> torch.Tensor:
    return self._applied_actions

  @property
  def tau_real(self) -> torch.Tensor:
    return self._tau_real

  def process_actions(self, actions: torch.Tensor) -> None:
    # 延迟队列：队首是「现在该下发」的（delay_steps 步前的动作）
    if self._fifo:
      self._fifo.append(actions.clone())
      self._applied_actions = self._fifo.pop(0)
    else:
      self._applied_actions = actions.clone()

  def apply_actions(self) -> None:
    cfg = self.cfg
    entity = self._entity
    qpos = entity.data.joint_pos
    qvel = entity.data.joint_vel

    # 物理动作硬裁到设计范围 ±1（target ±0.5 rad、轮速 ±10 rad/s）。
    # 只靠 PPO 的 clip_actions=100 不够：重质量下 raw action 偶发 ±3~5 就会把
    # 虚拟目标推到 ±2.5 rad，shank 雅可比奇异，关节速度/obs 直接 NaN。
    a = torch.clamp(self._applied_actions, -1.0, 1.0)
    target = self._default_virtual + cfg.pos_action_scale * a
    tau_cmd = torch.zeros_like(a)

    for i, side in enumerate(("left", "right")):
      leg = self._legs[side]
      q6 = qpos[:, self._leg_idx[side]]
      qd6 = qvel[:, self._leg_idx[side]]
      d_front, d_rear = leg.shank_jacobian(q6)
      self._d[side] = (d_front, d_rear)

      q_hip, q_knee = q6[:, 4], q6[:, 5]
      qd_hip, qd_knee = qd6[:, 4], qd6[:, 5]
      qd_wheel = qvel[:, self._wheel_idx[side]]

      # 虚拟空间 PD
      tau_hip = self._kp.squeeze(-1) * (target[:, 3 * i] - q_hip) - self._kd.squeeze(-1) * qd_hip
      tau_knee = self._kp.squeeze(-1) * (target[:, 3 * i + 1] - q_knee) - self._kd.squeeze(-1) * qd_knee
      lim_hk = self._effort_limit[:, 3 * i : 3 * i + 2]
      tau_hip = torch.clamp(tau_hip, -lim_hk[:, 0], lim_hk[:, 0])
      tau_knee = torch.clamp(tau_knee, -lim_hk[:, 1], lim_hk[:, 1])

      tau_wheel = self._wheel_kd.squeeze(-1) * (
        cfg.vel_action_scale * a[:, 3 * i + 2] - qd_wheel
      )
      lim_w = self._effort_limit[:, 3 * i + 2]
      tau_wheel = torch.clamp(tau_wheel, -lim_w, lim_w)

      # 虚功映射到真机 (后髋, 前髋曲柄)
      tau_cmd[:, 3 * i + 1] = tau_hip + tau_knee * d_rear
      tau_cmd[:, 3 * i] = tau_knee * d_front
      tau_cmd[:, 3 * i + 2] = tau_wheel

    tau_cmd = torch.clamp(tau_cmd, -self._motor_limit, self._motor_limit)

    # 一阶执行器滞后（真机力矩空间）
    alpha = min(1.0, self._env.physics_dt / max(cfg.actuator_tau, 1e-6))
    self._tau_real += (tau_cmd - self._tau_real) * alpha
    self._tau_real = torch.clamp(self._tau_real, -self._motor_limit, self._motor_limit)

    # 经 actuator target 缓冲下发（XML motor = 力矩直通）
    self._entity.set_joint_effort_target(
        self._tau_real, joint_ids=self._driven_joint_ids
    )

  def reset(self, env_ids: torch.Tensor | slice | None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._applied_actions[env_ids] = 0.0
    self._tau_real[env_ids] = 0.0
    for buf in self._fifo:
      buf[env_ids] = 0.0


# ---------------------------------------------------------------------------
# 观测（25 维，顺序与 Isaac 版逐维一致）
# ---------------------------------------------------------------------------


def _robot(env):  # noqa: ANN001, ANN202
  return env.scene["robot"]


def _wheel_leg_term(env):  # noqa: ANN001
  """取当前任务的轮腿动作项（平地 "wheelleg" 或跳跃 "jump"）。"""
  am = env.action_manager
  for name in ("wheelleg", "jump"):
    try:
      return am.get_term(name)
    except KeyError:
      continue
  raise KeyError("找不到轮腿动作项（wheelleg / jump）")


def _leg_q_qd(env):  # noqa: ANN001, ANN202
  """返回虚拟口径的 (q6, qd6)：[L髋,L膝,L轮, R髋,R膝,R轮]。"""
  entity = _robot(env)
  term = _wheel_leg_term(env)
  qpos, qvel = entity.data.joint_pos, entity.data.joint_vel
  out_q, out_qd = [], []
  for side in ("left", "right"):
    idx = term._leg_idx[side]
    q6, qd6 = qpos[:, idx], qvel[:, idx]
    wheel = term._wheel_idx[side]
    out_q += [q6[:, 4], q6[:, 5], qpos[:, wheel]]
    out_qd += [qd6[:, 4], qd6[:, 5], qvel[:, wheel]]
  return torch.stack(out_q, dim=-1), torch.stack(out_qd, dim=-1)


def _virtual_default(env):  # noqa: ANN001, ANN202
  return _wheel_leg_term(env)._default_virtual


def obs_ang_vel(env):  # noqa: ANN001
  return _robot(env).data.root_link_ang_vel_b


def obs_gravity(env):  # noqa: ANN001
  # mjlab 口径直立 = (0,0,-1)，与策略口径一致，直接用
  return _robot(env).data.projected_gravity_b


def obs_joint_pos(env):  # noqa: ANN001
  q, _ = _leg_q_qd(env)
  default = _virtual_default(env)
  # [L髋, L膝, R髋, R膝]
  return q[:, [0, 1, 3, 4]] - default[:, [0, 1, 3, 4]]


def obs_joint_vel(env):  # noqa: ANN001
  _, qd = _leg_q_qd(env)
  return qd


def obs_applied_action(env):  # noqa: ANN001
  # 口径 = Isaac 版 / sim2sim：obs 里的"上一步动作"是**最新下发**的动作
  # （onnx_mj_v11.run 里 last_actions = act 先于延迟队列）。
  # 不能用 applied_actions（延迟一拍），否则平滑度惩罚与观测对不上，PPO 会把
  # 动作学爆（实测 ±100 震荡、action_smooth 罚分 -4 万主导）。
  return env.action_manager.action


# ---------------------------------------------------------------------------
# 奖励（权重与公式逐项对齐 v11/legged_gym；**每项按 legged_gym 语义裁剪**）
# ---------------------------------------------------------------------------


def clipped(fn, weight):
    """legged_gym 的奖励项裁剪语义：clip(weight * raw, -1, +1)。

    原实现（plane_v11/legged_robot.py compute_reward）对每个加权奖励项
    clip 到 ±clip_single_reward(=1)·dt，再由 RewardManager 的
    scale_rewards_by_dt=True 统一乘 dt 等价。没有这个裁剪时，-3e-6·Σq̈²、
    -0.12·ΣΔa² 这类项能冲到 -900/步，把跟踪奖励完全淹没，value 被迫拟合
    1e9 量级目标——所有训练异常的总根源。
    """

    def wrapped(env, **kwargs):  # noqa: ANN001
        return torch.clamp(weight * fn(env, **kwargs), -1.0, 1.0)

    return wrapped


def _cmd(env):  # noqa: ANN001, ANN202
  return env.command_manager.get_command("cmd")


def _contact_force(env):  # noqa: ANN001
  """底盘/腿/pom 轮与地面接触的最大法向合力 (n,)。"""
  sensor = env.scene["body_contact"]
  force = sensor.data.force  # (n, P, 3)
  if force is None:
    return torch.zeros(env.num_envs, device=env.device)
  return torch.linalg.norm(force, dim=-1).amax(dim=-1)


def rew_track_lin_vel(env):  # noqa: ANN001
  cmd = _cmd(env)
  err = torch.square(cmd[:, 0] - _robot(env).data.root_link_lin_vel_b[:, 0])
  return torch.exp(-err / 0.25) + (torch.exp(-err / 2.5) - 1.0)


def rew_track_yaw_vel(env):  # noqa: ANN001
  cmd = _cmd(env)
  err = torch.square(cmd[:, 1] - _robot(env).data.root_link_ang_vel_b[:, 2])
  return torch.exp(-err / 0.25) + (torch.exp(-err / 2.5) - 1.0)


def rew_stand_still(env):  # noqa: ANN001
  """零指令站定惩罚原始量：|vx|/0.2 + |ωz|/0.5（站定指令是精确 0，
  见 LocomotionCommand._resample_command）。

  治零指令漂移：速度跟踪核 exp(-err²/0.25) 对 0.08 m/s 的漂移只损失 2.5%
  奖励，训练学会了"漂着站"；此项让漂移直接掉奖励（漂 0.08 → -0.4/步，
  weight -1.0）。行走指令（非零）下门控为 0，不影响机动。
  """
  cmd = _cmd(env)
  standing = (cmd[:, 0].abs() < 1e-6) & (cmd[:, 1].abs() < 1e-6)
  vx = _robot(env).data.root_link_lin_vel_b[:, 0]
  wz = _robot(env).data.root_link_ang_vel_b[:, 2]
  return (vx.abs() / 0.2 + wz.abs() / 0.5) * standing.float()


def rew_yaw_hold(env):  # noqa: ANN001
  """零偏航指令时线性罚 |ωz|/0.2（治直行走弧线）。

  蒸馏基座直行带 -1.9°/s 的恒定偏航速率（8s 拐 13°），跟踪核
  exp(-err²/0.25) 在 0.03 rad/s 误差处梯度≈0 修不掉；线性项在任意小
  误差处都有梯度。偏航指令非零时门控为 0（转向跟踪交给 track_yaw_vel）。
  """
  cmd = _cmd(env)
  hold = (cmd[:, 1].abs() < 1e-6).float()
  wz = _robot(env).data.root_link_ang_vel_b[:, 2]
  return wz.abs() / 0.2 * hold


def rew_height(env):  # noqa: ANN001
  # 原版核宽（plane_v11: exp(-err²/0.001/10) = /0.01，σ≈10cm）。
  cmd = _cmd(env)
  err = torch.square(_robot(env).data.root_link_pos_w[:, 2] - cmd[:, 2])
  return torch.exp(-err / 0.01)


def rew_height_under(env):  # noqa: ANN001
  """单向欠高惩罚（实心变体专用）：只罚 z 低于指令 5 mm 以上的部分。

  exp 高度核在 5 cm 误差处仍有 0.76 的奖励，"低 5 cm 站稳"和"站到指令"
  只差 0.24/步，压不住低伏稳定解（实测 solid v2 稳在 0.136/0.190）。
  线性欠高项给恒定梯度。死区从 2 cm 收到 5 mm：15.641 kg 的 v5 实测
  稳在 0.131 / cmd 0.150，正好卡在 2 cm 死区边缘、惩罚为 0，高度曲线
  平着不涨；5 mm 死区只放过传感器噪声和 PD 正常压深。
  """
  cmd = _cmd(env)
  z = _robot(env).data.root_link_pos_w[:, 2]
  return torch.clamp(cmd[:, 2] - z - 0.005, min=0.0)


def rew_theta0_sync(env):  # noqa: ANN001
  term = _wheel_leg_term(env)
  q, _ = _leg_q_qd(env)
  th_l = term._legs["left"].theta0(q[:, 0], q[:, 1])
  th_r = term._legs["right"].theta0(q[:, 3], q[:, 4])
  return torch.square(th_l - th_r)


def rew_lin_vel_z(env):  # noqa: ANN001
  return torch.square(_robot(env).data.root_link_lin_vel_b[:, 2])


def rew_ang_vel_xy(env):  # noqa: ANN001
  return torch.sum(torch.square(_robot(env).data.root_link_ang_vel_b[:, :2]), dim=1)


def rew_upright(env):  # noqa: ANN001
  g = _robot(env).data.projected_gravity_b
  return torch.sum(torch.square(g[:, :2]), dim=1)


def rew_hip_knee_vel(env):  # noqa: ANN001
  _, qd = _leg_q_qd(env)
  return torch.sum(torch.square(qd[:, [0, 1, 3, 4]]), dim=1)


def rew_dof_acc(env):  # noqa: ANN001
  entity = _robot(env)
  term = _wheel_leg_term(env)
  acc = entity.data.joint_acc
  cols = []
  for side in ("left", "right"):
    idx = term._leg_idx[side]
    cols += [acc[:, idx[4]], acc[:, idx[5]], acc[:, term._wheel_idx[side]]]
  acc6 = torch.stack(cols, dim=-1)
  # MuJoCo 硬接触瞬间 q̈ 无界（PhysX 有 max_depenetration_velocity 钳制），
  # 不裁剪的话 -3e-6×Σq̈² 在落地冲击时会崩出 1e6 量级奖励，value 函数无法拟合。
  acc6 = acc6.clamp(-1000.0, 1000.0)
  return torch.sum(torch.square(acc6), dim=1)


def rew_torque(env):  # noqa: ANN001
  tau = _wheel_leg_term(env).tau_real
  return torch.sum(torch.square(tau), dim=1)


def rew_bad_contact(env):  # noqa: ANN001
  """底盘/腿 hull 蹭地惩罚（POM 导轮不在传感器集合内）。

  plane_v11 原版的接触项是空列表（等于没有），但那是因为串联训练模型的底盘
  离地高；闭链真机模型会前倾到用底盘蹭地，不罚的话策略学出"单轮支撑+蹭地"
  的低成本平衡态（实测 sim2sim：h 在 0.11~0.16 弹跳、底盘接地）。
  """
  return (_contact_force(env) > 5.0).float()


def rew_upright_fine(env):  # noqa: ANN001
  """小权重直立项：-2×Σg_xy²，裁剪在 45° 才饱和。

  upright 原项 -100×Σg_xy² 被 ±1 裁剪在 ~6° 就饱和，6°~90° 之间梯度为 0，
  策略可以前倾 25° 却与直立 6° 代价相同。加一个小权重的"细粒度"项把
  10°~45° 区间的梯度找回来（两项叠加：6° 内强约束，6°~45° 渐进约束）。
  """
  g = _robot(env).data.projected_gravity_b
  return torch.sum(torch.square(g[:, :2]), dim=1)


def rew_action_rate(env):  # noqa: ANN001
  am = env.action_manager
  return torch.sum(torch.square(am.action - am.prev_action), dim=1)


def rew_action_smooth(env):  # noqa: ANN001
  am = env.action_manager
  diff = am.action - 2.0 * am.prev_action + am.prev_prev_action
  return torch.sum(torch.square(diff[:, HIP_KNEE_DIMS]), dim=1)


# ---------------------------------------------------------------------------
# 终止
# ---------------------------------------------------------------------------


def term_bad_contact(env):  # noqa: ANN001
  return _contact_force(env) > 20.0  # 轻蹭（<20N）不算摔，给后仰回正留学习空间


class SustainedTilt:
  """倾角超限持续 1 s 才判失败（与 Isaac 版 fail_time 逻辑一致）。"""

  def __init__(self, cfg: TerminationTermCfg, env):  # noqa: ANN001
    self._env = env
    self._tilt_cos = float(cfg.params.get("tilt_cos", -0.1))
    self._duration = float(cfg.params.get("duration_s", 1.0))
    self._timer = torch.zeros(env.num_envs, device=env.device)

  def __call__(self, env, **kwargs):  # noqa: ANN001, ANN003
    grav_z = env.scene["robot"].data.projected_gravity_b[:, 2]  # 直立 = -1
    tilted = grav_z > self._tilt_cos
    self._timer = torch.where(
      tilted, self._timer + env.step_dt, torch.zeros_like(self._timer)
    )
    return self._timer > self._duration

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._timer[env_ids] = 0.0


class BaseHeightTermination:
  """base 高度低于阈值持续 duration_s 判失败（专治折叠腿/蹭地低伏）。

  实心碰撞变体的实测失败模式：策略把腿折进底盘、把 base 压到 0.06 m
  （轮半径 0.06 m）贴地滑行，track_lin 0.97 但站高完全丢失。高度奖励
  的 exp 核在 10 cm 以上误差处梯度很小，单靠奖励压不住；用硬终止把
  低伏姿态直接判死。阈值由任务传入：15.641 kg 模型零动作自然沉到
  0.073 m，用 0.07 m；5.641 kg 时代用 0.12 m。
  """

  def __init__(self, cfg: TerminationTermCfg, env):  # noqa: ANN001
    self._min = float(cfg.params.get("min_height", 0.12))
    self._duration = float(cfg.params.get("duration_s", 0.3))
    self._dt = float(getattr(env, "step_dt", 0.01))
    self._timer = torch.zeros(env.num_envs, device=env.device)

  def __call__(self, env, **kwargs):  # noqa: ANN001, ANN003
    z = env.scene["robot"].data.root_link_pos_w[:, 2]
    low = z < self._min
    self._timer = torch.where(low, self._timer + self._dt,
                              torch.zeros_like(self._timer))
    return self._timer >= self._duration

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._timer[env_ids] = 0.0


class SustainedBadContact:
  """底盘/腿/POM 接触力超过阈值持续 duration_s 判失败。

  专治 POM 导轮支撑的伪站姿：15.641 kg 的 v8 在 27° 前倾下靠后部 POM 轮
  撑出 ep_len 2000，但 bad_contact -0.975、upright_fine -0.416；二值
  bad_contact 奖励已饱和，给不出梯度。阈值由任务传（solid 用 30 N/0.5 s；
  20 N/0.3 s 会把普通落地接触也判死，v9 ep_len 只有 60）。
  只在 solid 任务启用。
  """

  def __init__(self, cfg: TerminationTermCfg, env):  # noqa: ANN001
    self._force = float(cfg.params.get("force_threshold", 20.0))
    self._duration = float(cfg.params.get("duration_s", 0.3))
    self._dt = float(getattr(env, "step_dt", 0.01))
    self._timer = torch.zeros(env.num_envs, device=env.device)

  def __call__(self, env, **kwargs):  # noqa: ANN001, ANN003
    bad = _contact_force(env) > self._force
    self._timer = torch.where(bad, self._timer + self._dt,
                              torch.zeros_like(self._timer))
    return self._timer >= self._duration

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._timer[env_ids] = 0.0


# ---------------------------------------------------------------------------
# reset：虚拟角扰动 + 闭链牛顿投影（保证初值在流形上，无约束弹跳）
# ---------------------------------------------------------------------------


class ResetLegJoints:
  """Isaac 版是「所有关节独立 ±0.02」；闭链上独立扰动会造出不一致构型，
  connect 约束开局一记猛拉（实测角速度冲到 ~4 rad/s）。这里改成：每条腿
  采样虚拟 (rear, shank) 扰动，牛顿解出一致的前链 4 关节；轮/pom 轮独立
  扰动（无约束）。
  """

  def __init__(self, cfg, env):  # noqa: ANN001
    self._env = env
    entity = env.scene["robot"]
    self._entity = entity
    self._leg_idx = {
      side: torch.tensor(entity.find_joints(JOINT_NAMES[side], preserve_order=True)[0])
      for side in ("left", "right")
    }
    self._legs = {s: LegTorch(s) for s in ("left", "right")}
    # 每条腿的默认 q6（entity 关节序），牛顿迭代种子
    self._seed = {}
    default = entity.data.default_joint_pos
    for side in ("left", "right"):
      self._seed[side] = default[:, self._leg_idx[side]].clone()

  def __call__(self, env, env_ids, position_range=(-0.02, 0.02), **kwargs):  # noqa: ANN001, ANN003
    entity = self._entity
    n = len(env_ids)
    lo, hi = position_range
    dev = env.device
    full_pos = entity.data.default_joint_pos[env_ids].clone()
    for side in ("left", "right"):
      idx = self._leg_idx[side]
      d_rear = torch.empty(n, device=dev).uniform_(lo, hi)
      d_shank = torch.empty(n, device=dev).uniform_(lo, hi)
      q6 = self._legs[side].solve_virtual_batch(
        self._seed[side][env_ids, 4] + d_rear,
        self._seed[side][env_ids, 5] + d_shank,
        seed_q6=self._seed[side][env_ids],
      )
      full_pos[:, idx] = q6
    # 轮与 pom 轮：无约束，直接独立扰动
    wheel_ids = torch.tensor(
      [
        entity.find_joints((WHEEL_JOINT_NAMES[s],))[0][0]
        for s in ("left", "right")
      ]
      + entity.find_joints(tuple(f"pom_wheel{i}_joint" for i in (1, 2, 3, 4)))[0],
      device=dev,
    )
    full_pos[:, wheel_ids] += torch.empty(n, len(wheel_ids), device=dev).uniform_(lo, hi)

    entity.write_joint_position_to_sim(full_pos, env_ids=env_ids)
    entity.write_joint_velocity_to_sim(
      torch.zeros_like(full_pos), env_ids=env_ids
    )


# ---------------------------------------------------------------------------
# 课程：先全站定学会站，站稳后放开速度指令（两段）
# ---------------------------------------------------------------------------


class CommandStandingCurriculum:
  """按近期平均 episode 长度推进阶段，改写 LocomotionCommand 的 standing_prob。

  stage 0: standing_prob=1.0（只练站/调高）——25 维观测里没有线速度，
  平衡本身就要先站住才谈得上跟踪；
  stage 1: standing_prob=0.5（v1 原配方的指令混合）。
  阈值 ep_len_threshold 步（默认 800 = 8 s）意味着多数 episode 是超时而非摔死。
  """

  def __init__(self, cfg, env):  # noqa: ANN001
    self._stages = [float(p) for p in cfg.params.get("stages", (1.0, 0.5))]
    self._height_ranges = list(cfg.params.get("height_ranges", ()))
    # 指令范围课程：stage 1 从窄指令起步，跟踪达标后逐步放宽（先窄后宽）
    self._cmd_lin_init = float(cfg.params.get("cmd_lin_init", 0.5))
    self._cmd_lin_max = float(cfg.params.get("cmd_lin_max", 1.2))
    self._cmd_ang_init = float(cfg.params.get("cmd_ang_init", 0.4))
    self._cmd_ang_max = float(cfg.params.get("cmd_ang_max", 1.0))
    self._cmd_step = float(cfg.params.get("cmd_step", 0.1))
    self._track_threshold = float(cfg.params.get("track_threshold", 0.85))
    self._cmd_cooldown_iters = int(cfg.params.get("cmd_cooldown_iters", 50))
    self._last_expand_step = -10**9
    self._lin_recent: list[float] = []
    self._ang_recent: list[float] = []
    self._cur_lin = self._cmd_lin_init
    self._cur_ang = self._cmd_ang_init
    self._threshold = float(cfg.params.get("ep_len_threshold", 800))
    self._window = int(cfg.params.get("window", 400))
    self._warmup_s = float(cfg.params.get("warmup_s", 300.0))
    # 从 stage 1 续训时用 start_stage=1：curriculum 不进 checkpoint，
    # resume 会回到 stage 0，把 stage 1 策略放回全站定分布会塌（实测
    # v6 续训 iter 359~382 掉到 0.111 m / ep_len 61）。
    self._stage = int(cfg.params.get("start_stage", 0))
    self._recent: list[float] = []
    # 预热防护按"本进程存活时间"计（resume 会恢复 common_step_counter，
    # 按它防护会被绕过；init_at_random_ep_len 的随机长度 episode 最长 20 s，
    # 进程跑满 5 min 后窗口里必然全是真实 episode）。
    import time as _time
    self._t0 = _time.time()
    # Stage 0 的设置（高度范围等）此前只在晋级时应用，初始从未生效——
    # 构造时立即应用一次。
    self._apply(env)

  def __call__(self, env, env_ids, **kwargs):  # noqa: ANN001, ANN003
    # 预热期：init_at_random_ep_len 会把首轮 episode 长度随机化，开局直接量出
    # 虚高均值误触晋级（resume 后尤其如此——common_step_counter 被恢复）。
    import time as _time
    if _time.time() - self._t0 < float(self._warmup_s):
      return {"stage": self._stage, "standing_prob": self._stages[self._stage],
              "mean_ep_len": 0}
    if env_ids is not None and len(env_ids) > 0:
      # reset 最先调 curriculum，episode_length_buf 还是刚结束 episode 的长度
      lens = env.episode_length_buf[env_ids].detach()
      self._recent.extend(lens.cpu().tolist())
      del self._recent[:-self._window]
    # stage 1：按最近一批 reset 的跟踪质量放宽指令范围
    if self._stage >= 1 and env_ids is not None and len(env_ids) > 0:
      sums = getattr(env.reward_manager, "_episode_sums", {})
      tl = sums.get("track_lin_vel")
      ty = sums.get("track_yaw_vel")
      lens = env.episode_length_buf[env_ids].detach().clamp_min(1).float()
      if tl is not None and ty is not None:
        # episode sum = Σ clip×dt → /ep_len/0.01 = clip 均值（1=完美）。
        # 单批 reset 的测量噪声很大（实测 8 批就冲满），用滑动均值 + 冷却，
        # 每次扩展至少间隔 cmd_cooldown_iters 个训练轮。
        lin_rate = (tl[env_ids].detach() / lens / 0.01).mean().item()
        ang_rate = (ty[env_ids].detach() / lens / 0.01).mean().item()
        self._lin_recent.extend([lin_rate] * min(len(env_ids), 32))
        self._ang_recent.extend([ang_rate] * min(len(env_ids), 32))
        del self._lin_recent[:-200]
        del self._ang_recent[:-200]
        steps_per_iter = 48  # num_steps_per_env
        cooled = (env.common_step_counter - self._last_expand_step) >= (
          self._cmd_cooldown_iters * steps_per_iter)
        mean_lin = sum(self._lin_recent) / len(self._lin_recent) if self._lin_recent else 0.0
        mean_ang = sum(self._ang_recent) / len(self._ang_recent) if self._ang_recent else 0.0
        if cooled and len(self._lin_recent) >= 100:
          if mean_lin > self._track_threshold and self._cur_lin < self._cmd_lin_max:
            self._cur_lin = min(self._cmd_lin_max, self._cur_lin + self._cmd_step)
            self._last_expand_step = env.common_step_counter
            self._lin_recent.clear()
            self._ang_recent.clear()
            print(f"[curriculum] 指令放宽: lin ±{self._cur_lin:.2f}"
                  f"（滑窗跟踪 {mean_lin:.2f}）", flush=True)
          elif mean_ang > self._track_threshold * 0.9 and self._cur_ang < self._cmd_ang_max:
            self._cur_ang = min(self._cmd_ang_max, self._cur_ang + self._cmd_step)
            self._last_expand_step = env.common_step_counter
            self._ang_recent.clear()
            print(f"[curriculum] 转向放宽: ang ±{self._cur_ang:.2f}"
                  f"（滑窗跟踪 {mean_ang:.2f}）", flush=True)
        self._apply(env)
    mean_len = (sum(self._recent) / len(self._recent)) if self._recent else 0.0
    if (
      self._stage + 1 < len(self._stages)
      and len(self._recent) >= self._window
      and mean_len >= self._threshold
    ):
      self._stage += 1
      self._apply(env)
      self._recent.clear()
      print(f"[curriculum] 进入 stage {self._stage}: standing_prob="
            f"{self._stages[self._stage]}", flush=True)
    return {"stage": self._stage, "standing_prob": self._stages[self._stage],
            "mean_ep_len": round(mean_len)}

  def _apply(self, env):  # noqa: ANN001
    term = env.command_manager.get_term("cmd")
    term.cfg.standing_prob = self._stages[self._stage]
    if self._height_ranges:
      term.cfg.height_range = self._height_ranges[min(self._stage, len(self._height_ranges) - 1)]
    if self._stage >= 1:
      term.cfg.lin_vel_range = (-self._cur_lin, self._cur_lin)
      term.cfg.ang_vel_range = (-self._cur_ang, self._cur_ang)

  def reset(self, env_ids=None):  # noqa: ANN001
    pass


# ---------------------------------------------------------------------------
# 指标（episode 平均，对应 Isaac 版的诊断输出）
# ---------------------------------------------------------------------------


def metric_height_err(env):  # noqa: ANN001
  cmd = _cmd(env)
  return torch.abs(_robot(env).data.root_link_pos_w[:, 2] - cmd[:, 2])


def metric_speed_abs(env):  # noqa: ANN001
  return torch.abs(_robot(env).data.root_link_lin_vel_b[:, 0])


def metric_stand_drift(env):  # noqa: ANN001
  """站定环境（速度指令精确 0）的机体 vx 均值，带符号（负=后漂）。

  把站定环境的当拍均值广播到所有环境返回——这样不管 metric 管理器按环境
  平均还是按 episode 累计，结果都等于站定漂移的时序平均，不被行走环境稀释。
  """
  cmd = _cmd(env)
  standing = (cmd[:, 0].abs() < 1e-6) & (cmd[:, 1].abs() < 1e-6)
  n = int(standing.sum())
  if n == 0:
    return torch.zeros(env.num_envs, device=env.device)
  vx = _robot(env).data.root_link_lin_vel_b[:, 0]
  return torch.full((env.num_envs,), float(vx[standing].mean()), device=env.device)


def _standing_mean(env, values):  # noqa: ANN001
  """站定环境（速度指令精确 0）的 values 均值，广播到所有环境返回。"""
  cmd = _cmd(env)
  standing = (cmd[:, 0].abs() < 1e-6) & (cmd[:, 1].abs() < 1e-6)
  n = int(standing.sum())
  if n == 0:
    return torch.zeros(env.num_envs, device=env.device)
  return torch.full((env.num_envs,), float(values[standing].mean()), device=env.device)


def metric_stand_height(env):  # noqa: ANN001
  """站定环境的 base 高度均值（"最终重心高度"，口径与高度指令一致 =
  base_link 离地 z）。广播返回，不被行走环境稀释。"""
  return _standing_mean(env, _robot(env).data.root_link_pos_w[:, 2])


def metric_stand_height_cmd(env):  # noqa: ANN001
  """站定环境的高度指令均值——站定高度的对数对照线。"""
  return _standing_mean(env, _cmd(env)[:, 2])


class RiseSettle:
  """启动分段计时 + 十次测试平方平均（params={"mode": "rise"|"settle"}）。

  单次测试口径（三段：落下 → 起立 → 站稳）：
  - 出生位姿比站姿高（INIT_HEIGHT 0.19 先自由落体到折叠 resting 高度），
    所以峰值跟踪必须从**落低点之后**开始，否则出生点就是"最高点"、起立时间
    恒为 ~0（实测踩过：rise 全体 0.07s）；
  - 落低点：z 的运行最小值；z 离开低点 5mm 以上 = 进入起立段；
  - 最高点 = 起立段内 base z 运行最大值时刻 → 起立时间 = 该时刻（含落下段，
    "从开始到最高点"口径）；
  - 站稳 = 起立段内 |z − 指令| < 3cm 且 |vz| < 0.15 m/s 持续 0.1 s 首刻
    （vz 条件防止快速穿过指令带被误判；只在起立段计时，出生低点静息不触发）；
    稳定时间 = 站稳 − 最高点（≥0）；
  - 一直没站稳：稳定时间随观察时长增长（大值 = 未站稳标记）。

  返回值 = 每 env 最近 10 次测试（含当前进行中的一次）的平方平均 RMS——
  平方平均放大离群的"猛起立"，一次摔蹲就把十次的读数带起来，符合
  "起立过猛要惩罚"的语义。窗口用环形缓冲 + 平方和增量维护。
  reduce="last"：episode 末值恰好 = 含本次在内的 10 次 RMS。
  reset 用 episode_length_buf 步数连续性检测（reset 钩子作双保险），
  init_at_random_ep_len 中段开局不污染窗口（首次观察不入窗）。
  """

  WINDOW = 10

  def __init__(self, cfg, env):  # noqa: ANN001
    self._dt = float(getattr(env, "step_dt", 0.01))
    self._mode = str(getattr(cfg, "params", {}).get("mode", "rise"))
    n = env.num_envs
    dev = env.device
    self._peak = torch.full((n,), -1.0, device=dev)      # 起立段内 z 运行最大值
    self._t_peak = torch.zeros(n, device=dev)            # 最高点时刻
    self._timer = torch.zeros(n, device=dev)             # 连续在带内时长
    self._t_stable = torch.full((n,), -1.0, device=dev)  # 站稳时刻（-1=未站稳）
    self._elapsed = torch.zeros(n, device=dev)           # 距上次 reset 的观察时长
    self._z_min = torch.full((n,), float("inf"), device=dev)  # 落低点
    self._risen = torch.zeros(n, dtype=torch.bool, device=dev)  # 已进入起立段
    self._prev = torch.full((n,), -2, dtype=torch.long, device=dev)
    self._started = torch.zeros(n, dtype=torch.bool, device=dev)
    # 最近 WINDOW 次测试的环形缓冲 + 平方和（跨 episode 持续，reset 不清）
    self._win = torch.zeros(n, self.WINDOW, device=dev)
    self._idx = torch.zeros(n, dtype=torch.long, device=dev)
    self._count = torch.zeros(n, dtype=torch.long, device=dev)
    self._sq = torch.zeros(n, device=dev)
    self._fin = torch.zeros(n, device=dev)               # 上一步记录的当前测试值

  def __call__(self, env, **kwargs):  # noqa: ANN001, ANN003
    buf = env.episode_length_buf
    cont = buf == (self._prev + 1)   # 步数连续；回跳 = 刚 reset（episode 结束）
    self._elapsed = torch.where(cont, self._elapsed + self._dt,
                                torch.full_like(self._elapsed, self._dt))
    self._prev = buf.clone()
    # episode 结束：把上一步记录的最终值入窗（首次观察不入窗，防零污染）
    ended = self._started & ~cont
    if bool(ended.any()):
      rows = ended.nonzero(as_tuple=True)[0]
      i = self._idx[rows]
      self._sq[rows] += self._fin[rows] ** 2 - self._win[rows, i] ** 2
      self._win[rows, i] = self._fin[rows]
      self._idx[rows] = (i + 1) % self.WINDOW
      self._count[rows] = (self._count[rows] + 1).clamp(max=self.WINDOW)
    self._started[:] = True

    unset = self._t_stable < 0
    z = _robot(env).data.root_link_pos_w[:, 2]
    cmd_z = _cmd(env)[:, 2]
    # 落低点跟踪（未起立时）；离低点 5mm 以上 = 进入起立段
    new_min = (~self._risen) & (z < self._z_min)
    self._z_min = torch.where(new_min, z, self._z_min)
    self._risen = self._risen | ((~self._risen) & (z > self._z_min + 0.005))
    better = self._risen & unset & (z > self._peak)
    self._peak = torch.where(better, z, self._peak)
    self._t_peak = torch.where(better, self._elapsed, self._t_peak)
    vz = _robot(env).data.root_link_lin_vel_b[:, 2]
    in_band = ((z - cmd_z).abs() < 0.03) & (vz.abs() < 0.15)
    self._timer = torch.where(self._risen & unset & in_band, self._timer + self._dt,
                              torch.zeros_like(self._timer))
    newly = self._risen & unset & (self._timer >= 0.1)
    self._t_stable = torch.where(newly, self._elapsed, self._t_stable)
    if self._mode == "rise":
      cur = self._t_peak
    else:
      cur = torch.where(self._t_stable >= 0,
                        torch.clamp(self._t_stable - self._t_peak, min=0.0),
                        self._elapsed - self._t_peak)
    self._fin = cur
    # RMS：窗口满时当前测试替换最旧槽位（恰 10 次）；未满时累加
    full = self._count >= self.WINDOW
    oldest_sq = torch.gather(self._win, 1, self._idx.unsqueeze(1)).squeeze(1) ** 2
    num = torch.where(full, self._sq - oldest_sq + cur ** 2, self._sq + cur ** 2)
    den = torch.where(full, self._count, self._count + 1).clamp(min=1).float()
    return torch.sqrt(num / den)

  def reset(self, env_ids=None):  # noqa: ANN001
    if env_ids is None:
      env_ids = torch.arange(self._peak.shape[0], device=self._peak.device)
    if len(env_ids) == 0:
      return
    self._peak[env_ids] = -1.0
    self._t_peak[env_ids] = 0.0
    self._timer[env_ids] = 0.0
    self._t_stable[env_ids] = -1.0
    self._elapsed[env_ids] = 0.0
    self._z_min[env_ids] = float("inf")
    self._risen[env_ids] = False
    self._prev[env_ids] = -2
