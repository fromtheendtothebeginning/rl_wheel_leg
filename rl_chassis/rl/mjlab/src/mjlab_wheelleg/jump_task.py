"""v12 闭链轮腿跳跃任务：冻结平地策略 + 残差 actor + 相位状态机 + 气弹簧。

设计对齐 wxk's-branch 的压缩版跳跃方案（docs 2026-10-17 jump-policy-design）：

* 平地策略（wxk 蒸馏的 ``distill_wxk/model_0.pt``）**冻结**，每个控制周期算
  a_flat（25 维观测 + 5×25 term-major 历史，与平地任务同口径）；
* 残差 actor（PPO 训练）输入 42 维，输出 6 维残差；
* 相位状态机（CROUCH→THRUST→FLIGHT→EXTEND→LAND→RECOVER→FLAT）决定
  高度指令、残差缩放和气弹簧力（FLAT 20 N / 跳跃窗口 350 N）；
* 最终动作 = clip(a_flat + scale(phase)·a_residual, ±100)，PD 链与平地任务
  完全相同；气弹簧按 v12 tendon 几何的膝轴力臂解析加到膝力矩上。
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from enum import IntEnum

import numpy as np
import torch

from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .leg_torch import JOINT_NAMES, LegTorch
from .mdp import WheelLegAction, WheelLegActionCfg, _cmd, _leg_q_qd, _robot

# ---------------------------------------------------------------------------
# 相位状态机（对齐 rl/tools/jump_manager.py，批量化并合并落地记账）
# ---------------------------------------------------------------------------


class JumpPhase(IntEnum):
  FLAT = 0
  CROUCH = 1
  THRUST = 2
  FLIGHT = 3
  EXTEND = 4
  LAND = 5
  RECOVER = 6


PHASE_DUR = {JumpPhase.CROUCH: 0.12, JumpPhase.THRUST: 0.15,
             JumpPhase.EXTEND: 0.10, JumpPhase.LAND: 0.25, JumpPhase.RECOVER: 0.25}
HEIGHT_CROUCH, HEIGHT_THRUST, HEIGHT_FLIGHT = 0.14, 0.24, 0.14
HEIGHT_EXTEND, HEIGHT_LAND = 0.24, 0.15
GAS_FLAT, GAS_JUMP = 20.0, 350.0


class JumpManager:
  """批量相位管理 + 落地/离地记账（所有量都在 tensor 上）。"""

  def __init__(self, num_envs: int, device, target_height: float = 0.10):
    self.n = int(num_envs)
    self.device = device
    self.target_height = float(target_height)
    self.flight_dur = 2.0 * math.sqrt(max(2.0 * self.target_height / 9.81, 1e-6)) + 0.04

    self.phase = torch.zeros(self.n, dtype=torch.long, device=device)
    self.timer = torch.zeros(self.n, device=device)
    self.warmup = torch.zeros(self.n, device=device)
    self.start_z = torch.zeros(self.n, device=device)
    self.peak_z = torch.zeros(self.n, device=device)          # 相对起跳点
    self.landing_impact = torch.zeros(self.n, device=device)
    self.landing_timer = torch.zeros(self.n, device=device)
    self.just_landed = torch.zeros(self.n, dtype=torch.bool, device=device)
    self.was_in_flight = torch.zeros(self.n, dtype=torch.bool, device=device)
    self.flight_peak = torch.zeros(self.n, device=device)     # 绝对高度峰值
    self.landed_peak = torch.zeros(self.n, device=device)
    self.air_time = torch.zeros(self.n, device=device)
    self.recovered = torch.zeros(self.n, dtype=torch.bool, device=device)
    self.success = torch.zeros(self.n, dtype=torch.bool, device=device)
    self.last_contact = torch.zeros(self.n, dtype=torch.bool, device=device)
    self.contact_filt = torch.zeros(self.n, dtype=torch.bool, device=device)

  def reset_envs(self, env_ids, warmup=(0.5, 1.5)):
    if env_ids is None or len(env_ids) == 0:
      return
    m = len(env_ids)
    self.phase[env_ids] = int(JumpPhase.FLAT)
    self.timer[env_ids] = 0.0
    self.warmup[env_ids] = torch.rand(m, device=self.device) * (warmup[1] - warmup[0]) + warmup[0]
    self.start_z[env_ids] = 0.0
    self.peak_z[env_ids] = 0.0
    self.landing_impact[env_ids] = 0.0
    self.landing_timer[env_ids] = 0.0
    self.just_landed[env_ids] = False
    self.was_in_flight[env_ids] = False
    self.flight_peak[env_ids] = 0.0
    self.landed_peak[env_ids] = 0.0
    self.air_time[env_ids] = 0.0
    self.recovered[env_ids] = False
    self.success[env_ids] = False
    self.last_contact[env_ids] = False
    self.contact_filt[env_ids] = False

  def advance(self, dt: float, contact: torch.Tensor, root_z: torch.Tensor,
              base_vz: torch.Tensor, gravity_z: torch.Tensor,
              ang_xy: torch.Tensor) -> None:
    """推进一个控制周期。contact: (n,) 布尔，轮地接触（去抖前）。"""
    dt = float(dt)
    self.timer += dt
    self.landing_timer = torch.clamp(self.landing_timer - dt, min=0.0)
    prev_flight = self.was_in_flight.clone()

    self.contact_filt = contact | self.last_contact
    self.last_contact = contact.clone()
    self.was_in_flight = ~self.contact_filt
    self.just_landed = prev_flight & (~self.was_in_flight)
    self.landing_impact = torch.where(self.just_landed, torch.clamp(-base_vz, min=0.0),
                                      self.landing_impact)
    self.landing_timer = torch.where(self.just_landed,
                                     torch.full_like(self.landing_timer, 0.8),
                                     self.landing_timer)
    self.landed_peak = torch.where(self.just_landed, self.flight_peak,
                                   torch.zeros_like(self.flight_peak))
    active = self.phase != int(JumpPhase.FLAT)
    self.flight_peak = torch.where(
      self.was_in_flight, torch.maximum(self.flight_peak, root_z),
      torch.where(active, self.flight_peak, torch.zeros_like(self.flight_peak)))
    # 鼓励跳跃项用的滞空时间加权
    self.air_time = self.air_time + dt * self.was_in_flight.float() * torch.clamp(
      root_z, min=0.0, max=0.5)

    flat = self.phase == int(JumpPhase.FLAT)
    self.warmup = torch.where(flat, torch.clamp(self.warmup - dt, min=0.0), self.warmup)
    ready = flat & (self.warmup <= 0.0) & self.contact_filt
    self.start_z = torch.where(ready, root_z, self.start_z)
    self.peak_z = torch.where(ready, torch.zeros_like(self.peak_z), self.peak_z)
    self.flight_peak = torch.where(ready, torch.zeros_like(self.flight_peak), self.flight_peak)
    self.phase = torch.where(ready, torch.full_like(self.phase, int(JumpPhase.CROUCH)), self.phase)
    self.timer = torch.where(ready, torch.zeros_like(self.timer), self.timer)
    active = self.phase != int(JumpPhase.FLAT)
    self.peak_z = torch.maximum(self.peak_z, torch.where(active, root_z - self.start_z, self.peak_z))

    def _to(mask, phase, reset_warmup=False):
      if not bool(mask.any()):
        return
      self.phase = torch.where(mask, torch.full_like(self.phase, int(phase)), self.phase)
      self.timer = torch.where(mask, torch.zeros_like(self.timer), self.timer)
      if reset_warmup:
        self.warmup = torch.where(
          mask, torch.rand_like(self.warmup) * 1.0 + 0.5, self.warmup)

    _to((self.phase == int(JumpPhase.CROUCH)) & (self.timer >= PHASE_DUR[JumpPhase.CROUCH]),
        JumpPhase.THRUST)
    _to((self.phase == int(JumpPhase.THRUST)) & ((~self.contact_filt) | (self.timer >= 0.20)),
        JumpPhase.FLIGHT)
    _to((self.phase == int(JumpPhase.FLIGHT)) & (self.timer >= max(self.flight_dur - 0.10, 0.15)),
        JumpPhase.EXTEND)
    land = (self.phase == int(JumpPhase.EXTEND)) & (self.contact_filt | (self.timer >= 0.12))
    _to(land, JumpPhase.LAND)
    _to((self.phase == int(JumpPhase.LAND)) & (self.timer >= PHASE_DUR[JumpPhase.LAND]),
        JumpPhase.RECOVER)
    stable = (gravity_z < -0.98) & (ang_xy < 0.5)
    rec = (self.phase == int(JumpPhase.RECOVER)) & (self.timer >= PHASE_DUR[JumpPhase.RECOVER]) & stable
    self.recovered = rec
    self.success = rec & (self.peak_z >= 0.8 * self.target_height)
    self.air_time = torch.where(rec, torch.zeros_like(self.air_time), self.air_time)
    _to(rec, JumpPhase.FLAT, reset_warmup=True)

  # ---------------- 相位相关量 ----------------
  def phase_onehot(self) -> torch.Tensor:
    out = torch.zeros(self.n, 6, device=self.device)
    active = self.phase != int(JumpPhase.FLAT)
    idx = torch.clamp(self.phase - 1, min=0, max=5)
    out[active, idx[active]] = 1.0
    return out

  def phase_progress(self) -> torch.Tensor:
    dur = torch.full((self.n,), 0.25, device=self.device)
    for phase, value in PHASE_DUR.items():
      dur = torch.where(self.phase == int(phase), torch.full_like(dur, value), dur)
    dur = torch.where(self.phase == int(JumpPhase.FLIGHT),
                      torch.full_like(dur, self.flight_dur), dur)
    return torch.clamp(self.timer / dur, 0.0, 1.0)

  def residual_scale(self) -> torch.Tensor:
    scale = torch.ones(self.n, device=self.device)
    scale = torch.where(self.phase == int(JumpPhase.FLAT), torch.zeros_like(scale), scale)
    crouch = self.phase == int(JumpPhase.CROUCH)
    scale = torch.where(crouch, torch.clamp(self.timer / 0.05, 0.0, 1.0), scale)
    rec = self.phase == int(JumpPhase.RECOVER)
    scale = torch.where(
      rec, torch.clamp(1.0 - self.timer / PHASE_DUR[JumpPhase.RECOVER], 0.0, 1.0), scale)
    return scale

  def height_command(self, flat_height: torch.Tensor) -> torch.Tensor:
    out = flat_height.clone()
    out = torch.where(self.phase == int(JumpPhase.CROUCH), torch.full_like(out, HEIGHT_CROUCH), out)
    out = torch.where(self.phase == int(JumpPhase.THRUST), torch.full_like(out, HEIGHT_THRUST), out)
    out = torch.where(self.phase == int(JumpPhase.FLIGHT), torch.full_like(out, HEIGHT_FLIGHT), out)
    out = torch.where(self.phase == int(JumpPhase.EXTEND), torch.full_like(out, HEIGHT_EXTEND), out)
    out = torch.where(self.phase == int(JumpPhase.LAND), torch.full_like(out, HEIGHT_LAND), out)
    out = torch.where(self.phase == int(JumpPhase.RECOVER), torch.full_like(out, HEIGHT_LAND), out)
    return out

  def gas_force(self) -> torch.Tensor:
    return torch.where(self.phase == int(JumpPhase.FLAT),
                       torch.full((self.n,), GAS_FLAT, device=self.device),
                       torch.full((self.n,), GAS_JUMP, device=self.device))

  def in_flight(self) -> torch.Tensor:
    return self.was_in_flight

  def landing_mask(self) -> torch.Tensor:
    return (self.landing_timer > 0.0).float()


def gas_moment_arm(q_knee: torch.Tensor) -> torch.Tensor:
  """气弹簧力对膝轴的力臂（v12 MJCF tendon 几何，两条腿）。"""
  a = q_knee.new_tensor([-0.019073331073, 0.044456634729])
  knee = q_knee.new_tensor([-0.21499951, -0.00048882211])
  b = q_knee.new_tensor([0.0051496210451, 0.049247716229])
  sign = q_knee.new_tensor([1.0, -1.0])
  a_s = a.unsqueeze(0) * sign.unsqueeze(1)
  b_s = b.unsqueeze(0) * sign.unsqueeze(1)
  c, s = torch.cos(q_knee), torch.sin(q_knee)
  bx, by = b_s[:, 0], b_s[:, 1]
  rb = torch.stack((c * bx - s * by, s * bx + c * by), dim=-1)
  drb = torch.stack((-s * bx - c * by, c * bx - s * by), dim=-1)
  r = knee.unsqueeze(0) + rb - a_s
  return torch.sum(r * drb, dim=-1) / torch.linalg.norm(r, dim=-1).clamp_min(1e-6)


# ---------------------------------------------------------------------------
# 冻结平地策略（torch 版，权重来自 distill_wxk 的 checkpoint）
# ---------------------------------------------------------------------------

FLAT_CKPT_DEFAULT = os.path.join(
  os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
  "logs", "rsl_rl", "wheelleg_v12_mjlab", "distill_wxk", "model_0.pt")


class FrozenFlatPolicy:
  """25 维观测 ×5 帧（term-major）→ 6 动作。权重加载后冻结。"""

  TERM_DIMS = (3, 3, 3, 4, 6, 6)
  HIST = 5

  def __init__(self, num_envs: int, device, ckpt_path: str = FLAT_CKPT_DEFAULT):
    from torch import nn

    dims = [125, 256, 256, 256, 6]
    layers = []
    for i in range(4):
      layers.append(nn.Linear(dims[i], dims[i + 1]))
      if i < 3:
        layers.append(nn.ELU())
    self.net = nn.Sequential(*layers).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = {k[len("mlp."):]: v.float() for k, v in ckpt["actor_state_dict"].items()
          if k.startswith("mlp.")}
    self.net.load_state_dict(sd, strict=True)
    for p in self.net.parameters():
      p.requires_grad_(False)
    self.net.eval()
    self.device = device
    self.buffers = [torch.zeros(num_envs, self.HIST, d, device=device) for d in self.TERM_DIMS]

  def reset(self, env_ids):
    for b in self.buffers:
      b[env_ids] = 0.0

  @torch.no_grad()
  def __call__(self, obs25: torch.Tensor) -> torch.Tensor:
    i = 0
    flat = []
    for buf, d in zip(self.buffers, self.TERM_DIMS):
      buf[:, :-1] = buf[:, 1:]
      buf[:, -1] = obs25[:, i : i + d]
      flat.append(buf.reshape(buf.shape[0], -1))
      i += d
    return self.net(torch.cat(flat, dim=-1)).clamp(-100.0, 100.0)


# ---------------------------------------------------------------------------
# 动作项：残差 + 冻结平地策略 + 气弹簧 + PD
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class JumpResidualActionCfg(WheelLegActionCfg):
  flat_ckpt: str = FLAT_CKPT_DEFAULT
  initial_target_height: float = 0.10

  def build(self, env) -> "JumpResidualAction":  # noqa: ANN001
    return JumpResidualAction(self, env)


class JumpResidualAction(WheelLegAction):
  """残差动作项。策略输出 6 维残差，动作 = clip(a_flat + scale·residual)。"""

  cfg: JumpResidualActionCfg

  def __init__(self, cfg: JumpResidualActionCfg, env):  # noqa: ANN001
    super().__init__(cfg, env)
    self.flat = FrozenFlatPolicy(self.num_envs, self.device, cfg.flat_ckpt)
    self.manager = JumpManager(self.num_envs, self.device,
                               target_height=cfg.initial_target_height)
    self._last_total = torch.zeros(self.num_envs, 6, device=self.device)

  # ---------------- 用于动作合成的 25 维观测 ----------------
  def _obs25(self) -> torch.Tensor:
    entity = _robot(self._env)
    q, qd = _leg_q_qd(self._env)
    cmd = _cmd(self._env)
    # 相位高度指令：CROUCH=0.14 / THRUST=0.24 等，驱动冻结平地策略升降重心
    cmd = torch.cat(
      (cmd[:, :2], self.manager.height_command(cmd[:, 2]).unsqueeze(-1)), dim=-1)
    return torch.cat(
      (
        entity.data.root_link_ang_vel_b * 0.25,
        entity.data.projected_gravity_b,
        cmd * torch.tensor([2.0, 0.25, 5.0], device=self.device),
        q[:, [0, 1, 3, 4]] - self._default_virtual[:, [0, 1, 3, 4]],
        qd * 0.05,
        self._last_total,
      ),
      dim=-1,
    ).clamp(-100.0, 100.0)

  def _advance_manager(self) -> None:
    entity = _robot(self._env)
    sensor = self._env.scene["wheel_contact"]
    data = sensor.data
    found = data.found  # (n, P)
    contact = (found.sum(dim=-1) > 0) if found is not None else torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device)
    vz = entity.data.root_link_lin_vel_w[:, 2]
    grav = entity.data.projected_gravity_b
    ang_xy = torch.linalg.norm(entity.data.root_link_ang_vel_b[:, :2], dim=-1)
    self.manager.advance(
      self._env.step_dt, contact,
      entity.data.root_link_pos_w[:, 2], vz, grav[:, 2], ang_xy)

  def process_actions(self, actions: torch.Tensor) -> None:  # residual
    # 1) 状态机先推进（用当前接触/姿态），相位决定高度指令与残差缩放
    self._advance_manager()
    # 2) 冻结平地策略（当前状态 + 历史；高度指令已被相位覆盖）
    obs25 = self._obs25()
    a_flat = self.flat(obs25)
    scale = self.manager.residual_scale().unsqueeze(-1)
    # 3) 残差延迟 1 个控制周期后叠加
    if self._fifo:
      self._fifo.append(actions.clone())
      residual = self._fifo.pop(0)
    else:
      residual = actions
    total = a_flat + scale * residual
    self._applied_actions = total.clamp(-100.0, 100.0)
    self._last_total = self._applied_actions.clone()

  def apply_actions(self) -> None:
    """与平地动作项相同的 PD 链，外加气弹簧膝力矩。"""
    cfg = self.cfg
    entity = self._entity
    qpos = entity.data.joint_pos
    qvel = entity.data.joint_vel

    a = self._applied_actions
    target = self._default_virtual + cfg.pos_action_scale * a
    tau_cmd = torch.zeros_like(a)

    gas_force = self.manager.gas_force()
    for i, side in enumerate(("left", "right")):
      leg = self._legs[side]
      q6 = qpos[:, self._leg_idx[side]]
      qd6 = qvel[:, self._leg_idx[side]]
      d_front, d_rear = leg.shank_jacobian(q6)
      self._d[side] = (d_front, d_rear)

      q_hip, q_knee = q6[:, 4], q6[:, 5]
      qd_hip, qd_knee = qd6[:, 4], qd6[:, 5]
      qd_wheel = qvel[:, self._wheel_idx[side]]

      tau_hip = self._kp.squeeze(-1) * (target[:, 3 * i] - q_hip) - self._kd.squeeze(-1) * qd_hip
      tau_knee = self._kp.squeeze(-1) * (target[:, 3 * i + 1] - q_knee) - self._kd.squeeze(-1) * qd_knee
      lim_hk = self._effort_limit[:, 3 * i : 3 * i + 2]
      tau_hip = torch.clamp(tau_hip, -lim_hk[:, 0], lim_hk[:, 0])
      tau_knee = torch.clamp(tau_knee, -lim_hk[:, 1], lim_hk[:, 1])

      # 气弹簧（只作用于膝轴，力臂与 v12 tendon 几何一致）
      tau_gas = gas_force * gas_moment_arm(torch.stack((q_hip, q_knee), dim=-1))[:, i]
      tau_knee = tau_knee + tau_gas

      tau_wheel = self._wheel_kd.squeeze(-1) * (
        cfg.vel_action_scale * a[:, 3 * i + 2] - qd_wheel)
      lim_w = self._effort_limit[:, 3 * i + 2]
      tau_wheel = torch.clamp(tau_wheel, -lim_w, lim_w)

      tau_cmd[:, 3 * i + 1] = tau_hip + tau_knee * d_rear
      tau_cmd[:, 3 * i] = tau_knee * d_front
      tau_cmd[:, 3 * i + 2] = tau_wheel

    tau_cmd = torch.clamp(tau_cmd, -self._motor_limit, self._motor_limit)
    alpha = min(1.0, self._env.physics_dt / max(cfg.actuator_tau, 1e-6))
    self._tau_real += (tau_cmd - self._tau_real) * alpha
    self._tau_real = torch.clamp(self._tau_real, -self._motor_limit, self._motor_limit)
    self._entity.set_joint_effort_target(self._tau_real, joint_ids=self._driven_joint_ids)

  def reset(self, env_ids: torch.Tensor | slice | None) -> None:
    super().reset(env_ids)
    if env_ids is None:
      env_ids = slice(None)
    self.flat.reset(env_ids)
    self.manager.reset_envs(
      env_ids if isinstance(env_ids, torch.Tensor)
      else torch.arange(self.num_envs, device=self.device))
    self._last_total[env_ids] = 0.0


# ---------------------------------------------------------------------------
# 观测项（actor 42 维）
# ---------------------------------------------------------------------------


def _term(env):  # noqa: ANN001
  return env.action_manager.get_term("jump")


def jump_obs_current(env):  # noqa: ANN001
  """与平地任务同口径的 25 维当前观测（平地策略自己也用这一份）。"""
  return _term(env)._obs25()


def jump_obs_phase(env):  # noqa: ANN001
  return _term(env).manager.phase_onehot()


def jump_obs_progress(env):  # noqa: ANN001
  return _term(env).manager.phase_progress().unsqueeze(-1)


def jump_obs_contact(env):  # noqa: ANN001
  """可部署接触置信度（轮加速度/IMU/电流近似），左右各一维。"""
  entity = _robot(env)
  term = _term(env)
  acc = entity.data.joint_acc[:, [term._wheel_idx["left"], term._wheel_idx["right"]]].abs()
  imu = torch.linalg.norm(entity.data.root_link_ang_vel_b[:, :2], dim=-1, keepdim=True)
  current = term.tau_real[:, [2, 5]].abs() / 20.0
  conf = 0.5 * torch.clamp(acc / 50.0, 0.0, 1.0)
  conf += 0.3 * torch.clamp(imu * 2.0, 0.0, 1.0)
  conf += 0.2 * torch.clamp(current, 0.0, 1.0)
  return torch.clamp(conf, 0.0, 1.0)


def jump_obs_torque(env):  # noqa: ANN001
  term = _term(env)
  return term.tau_real[:, [2, 5]] / 20.0


def jump_obs_last_residual(env):  # noqa: ANN001
  return env.action_manager.action


# ---------------------------------------------------------------------------
# 奖励
# ---------------------------------------------------------------------------


def _mgr(env):  # noqa: ANN001
  return _term(env).manager


def _leg_L0(env):  # noqa: ANN001
  term = _term(env)
  qpos = _robot(env).data.joint_pos
  out = []
  for side in ("left", "right"):
    wx, wz = term._legs[side].wheel_xz(qpos[:, term._leg_idx[side]])
    out.append(torch.sqrt(wx * wx + wz * wz))
  return torch.stack(out, dim=-1)


def jump_base_height(env):  # noqa: ANN001
  return _robot(env).data.root_link_pos_w[:, 2]


def rew_jump_flight(env):  # noqa: ANN001
  return _mgr(env).in_flight().float()


def rew_jump_encourage(env):  # noqa: ANN001
  m = _mgr(env)
  vz = _robot(env).data.root_link_lin_vel_w[:, 2]
  rew = (m.air_time - 5e-5) * (~m.in_flight()).float() * 0.15
  rew += torch.clamp(vz, min=0.0) * 0.5
  return rew


def rew_jump_height_flight(env):  # noqa: ANN001
  m = _mgr(env)
  err = torch.abs(jump_base_height(env) - 0.40)
  return torch.exp(-err * 6.0) * m.in_flight().float()


def rew_jump_leg_tuck(env):  # noqa: ANN001
  m = _mgr(env)
  err = torch.sum(torch.abs(_leg_L0(env) - 0.16), dim=1)
  return torch.exp(-err * 4.0) * m.in_flight().float()


def rew_jump_takeoff_extend(env):  # noqa: ANN001
  m = _mgr(env)
  vz = _robot(env).data.root_link_lin_vel_w[:, 2]
  mask = m.contact_filt & (vz > 0.05)
  err = torch.sum(torch.abs(_leg_L0(env) - 0.40), dim=1)
  return torch.exp(-err * 4.0) * mask.float()


def rew_jump_line_z(env):  # noqa: ANN001
  m = _mgr(env)
  vz = _robot(env).data.root_link_lin_vel_w[:, 2]
  return torch.clamp(vz, min=0.0) * m.in_flight().float()


def rew_landing_soft(env):  # noqa: ANN001
  m = _mgr(env)
  return m.just_landed.float() * torch.exp(-torch.square(m.landing_impact / 0.30))


def rew_landing_stable(env):  # noqa: ANN001
  m = _mgr(env)
  entity = _robot(env)
  upright = torch.exp(-6.0 * torch.sum(torch.square(entity.data.projected_gravity_b[:, :2]), 1))
  still = torch.exp(-4.0 * torch.sum(torch.square(entity.data.root_link_ang_vel_b[:, :2]), 1))
  height = torch.exp(-torch.square((jump_base_height(env) - 0.18) / 0.04))
  return m.landing_mask() * (0.50 * upright + 0.35 * still + 0.15 * height)


def rew_landing_ang_vel(env):  # noqa: ANN001
  m = _mgr(env)
  return m.landing_mask() * torch.sum(
    torch.square(_robot(env).data.root_link_ang_vel_b[:, :2]), 1)


def rew_landing_lin_vel_z(env):  # noqa: ANN001
  m = _mgr(env)
  return m.landing_mask() * torch.square(_robot(env).data.root_link_lin_vel_w[:, 2])


def rew_landing_height(env):  # noqa: ANN001
  m = _mgr(env)
  return m.landing_mask() * torch.abs(jump_base_height(env) - 0.18)


def rew_landing_peak(env):  # noqa: ANN001
  m = _mgr(env)
  frac = torch.clamp((m.landed_peak - 0.24) / (0.40 - 0.24), 0.0, 1.0)
  return m.just_landed.float() * frac


def rew_landing_low_peak(env):  # noqa: ANN001
  m = _mgr(env)
  frac = torch.clamp((0.24 - m.landed_peak) / 0.24, 0.0, 1.0)
  return m.just_landed.float() * frac


def rew_residual_magnitude(env):  # noqa: ANN001
  a = env.action_manager.action
  return torch.sum(torch.square(a), dim=1)


def rew_wheel_spin_flight(env):  # noqa: ANN001
  m = _mgr(env)
  term = _term(env)
  qd = _robot(env).data.joint_vel[:, [term._wheel_idx["left"], term._wheel_idx["right"]]]
  return torch.sum(torch.square(qd), dim=1) * m.in_flight().float()


def rew_wheel_speed_match(env):  # noqa: ANN001
  """LAND 阶段轮速匹配地面速度的鼓励项。"""
  m = _mgr(env)
  entity = _robot(env)
  term = _term(env)
  cmd = _cmd(env)
  vx = entity.data.root_link_lin_vel_w[:, 0]
  qd = entity.data.joint_vel[:, [term._wheel_idx["left"], term._wheel_idx["right"]]]
  target = torch.stack((vx / 0.06, -vx / 0.06), dim=-1)
  err = torch.sum(torch.square(qd - target), dim=1)
  land = (m.phase == int(JumpPhase.LAND)).float()
  return torch.exp(-err * 0.05) * land


# ---------------------------------------------------------------------------
# 终止
# ---------------------------------------------------------------------------


def jump_fall(env):  # noqa: ANN001
  return _robot(env).data.projected_gravity_b[:, 2] > -0.1


def jump_success(env):  # noqa: ANN001
  return _mgr(env).success
