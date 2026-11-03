"""v12 闭链轮腿平地任务：ManagerBasedRlEnvCfg 组装。

资产 = ``rl_chassis/mjcf/rl_chassis_v12.xml``（真机口径闭链模型，含 4 个
connect 等式约束与 6 个电机）。仿真层 MuJoCo Warp 在 GPU 上批量步进，
训练/部署同一套物理——目标是把 PhysX→MuJoCo 的 sim2sim gap 整体消掉。

控制/观测/奖励逐项对齐 Isaac Lab 版（见 mdp.py 的 docstring）。
"""

from __future__ import annotations

from pathlib import Path

import mujoco

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers import TerminationTermCfg as Termination
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg as EventTerm
from mjlab.managers.metrics_manager import MetricsTermCfg as Metric
from mjlab.managers.observation_manager import (
  ObservationGroupCfg,
  ObservationTermCfg as Obs,
)
from mjlab.managers.reward_manager import RewardTermCfg as Reward
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from . import mdp
from .leg_torch import CLOSED_DEFAULT, JOINT_NAMES, MOTOR_NAMES, WHEEL_JOINT_NAMES
from .mdp import LocomotionCommandCfg, WheelLegActionCfg

_REPO = Path(__file__).resolve().parents[4]  # .../rl_chassis
V12_XML = _REPO / "mjcf" / "rl_chassis_v12.xml"
assert V12_XML.exists(), f"missing {V12_XML}"

# 实心碰撞变体（hybrid）：所有运动 link 用原始网格凸包，但 base_link 保留
# coacd 凸分解（腿坑）。单个 base 凸包会把 6 条腿 link 吞进腹腔，实测策略
# 学会折腿贴地滑行（站高 0.06 m）；保留腿坑后可用站姿 52/121、已知的
# flat_bc4 策略在 hybrid 上仍能站 0.190 m。
# 全凸包版留在 mjcf/*_allhull*，作为失败对照，不用于训练。
V12_SOLID_XML = _REPO / "mjcf" / "rl_chassis_v12_solid.xml"
assert V12_SOLID_XML.exists(), f"missing {V12_SOLID_XML}"

# 默认初始高度：闭链默认姿态轮心离地 → base ≈ 0.188，稍抬一点落地
INIT_HEIGHT = 0.19

# 接触判失败的对象：底盘/腿的凸包。**不含 4 个 POM 导轮**——那是真机尾部的
# 被动支撑轮，车体后仰时接地是正常功能，不算摔。
# 注意 mjlab 的名字匹配是 re.fullmatch（全名匹配），模式必须带 .* 前缀
# （裸 "_c[0-9]+_col" 会匹配不到任何 geom，导致传感器只盯 pom 轮）。
# 奖励惩罚的对象 = 底盘/腿 hull + **POM 导轮**。POM 轮是真机被动支撑轮，
# 不判死（不终止），但必须罚——否则策略会把它当支架：实测 v6 用 POM 轮
# 支撑出前倾 27°、高度 0.134 的"静态稳定"站姿（两轮倒立摆应该靠主动平衡）。
BAD_CONTACT_GEOMS = r"(.*_c[0-9]+_col|pom_wheel[0-9]_Link_col)"
# 实心变体的碰撞 geom 叫 <link>_col（不再有 _cNN 分块）；排除两个主动轮，
# 它们本来就是唯一允许长期接地的部件。POM 导轮保留（罚但不终止）。
BAD_CONTACT_GEOMS_SOLID = r"(?!(?:Left|Right)_Wheel_link_col$).*_col"
WHEEL_GEOMS = ("Left_Wheel_link_col", "Right_Wheel_link_col")


def get_v12_robot_cfg(xml_path: Path = V12_XML) -> "object":  # EntityCfg
  from mjlab.actuator import XmlActuatorCfg
  from mjlab.entity import EntityCfg, EntityArticulationInfoCfg

  joint_pos: dict[str, float] = {}
  for side in ("left", "right"):
    for name, angle in zip(JOINT_NAMES[side], CLOSED_DEFAULT[side]):
      joint_pos[name] = angle
    joint_pos[WHEEL_JOINT_NAMES[side]] = 0.0

  def spec_fn() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(xml_path))
    if len(spec.equalities) != 4:
      raise RuntimeError(f"v12 闭链 equality 数量异常: {len(spec.equalities)}")
    return spec

  # 电机驱动的 6 个关节（与 MOTOR_NAMES 的电机一一对应）
  driven_joints = (
    "Left_front_joint",
    "Left_rear_joint",
    "Left_Wheel_joint",
    "Right_front_joint",
    "Right_rear_joint",
    "Right_Wheel_joint",
  )
  return EntityCfg(
    spec_fn=spec_fn,
    articulation=EntityArticulationInfoCfg(
      # XML 里的 6 个 motor（ctrl = 力矩），用 XmlActuatorCfg 包一层注册进实体；
      # target_names_expr 匹配的是传动目标（关节名）。
      actuators=tuple(
        XmlActuatorCfg(target_names_expr=(name,)) for name in driven_joints
      ),
    ),
    init_state=EntityCfg.InitialStateCfg(
      pos=(0.0, 0.0, INIT_HEIGHT),
      joint_pos=joint_pos,
      joint_vel={".*": 0.0},
    ),
  )


def wheelleg_flat_env_cfg(play: bool = False, num_envs: int = 1024,
                          xml_path: Path = V12_XML,
                          min_base_height: float | None = None,
                          height_ranges: tuple[tuple[float, float], ...] = (
                            (0.19, 0.19), (0.14, 0.24)),
                          cmd_height_range: tuple[float, float] = (0.14, 0.24),
                          bad_contact_termination: bool = False
                          ) -> ManagerBasedRlEnvCfg:
  ##
  # 观测（actor 25 维；critic 同维但无噪声）
  ##
  actor_terms = {
    "ang_vel": Obs(
      func=mdp.obs_ang_vel,
      scale=0.25,
      noise=Unoise(n_min=-0.2, n_max=0.2),  # 折算到输出 ±0.05
      clip=(-100.0, 100.0),
    ),
    "gravity": Obs(
      func=mdp.obs_gravity,
      noise=Unoise(n_min=-0.05, n_max=0.05),
      clip=(-100.0, 100.0),
    ),
    "commands": Obs(
      func=envs_mdp.generated_commands,
      params={"command_name": "cmd"},
      scale=(2.0, 0.25, 5.0),
      clip=(-100.0, 100.0),
    ),
    "hip_knee_pos": Obs(
      func=mdp.obs_joint_pos,
      noise=Unoise(n_min=-0.02, n_max=0.02),
      clip=(-100.0, 100.0),
    ),
    "joint_vel": Obs(
      func=mdp.obs_joint_vel,
      scale=0.05,
      noise=Unoise(n_min=-1.5, n_max=1.5),  # 折算到输出 ±0.075
      clip=(-100.0, 100.0),
    ),
    "applied_action": Obs(
      func=mdp.obs_applied_action,
      clip=(-100.0, 100.0),
    ),
  }
  critic_terms = {
    k: Obs(func=t.func, params=dict(t.params), scale=t.scale, clip=(-100.0, 100.0))
    for k, t in actor_terms.items()
  }

  body_contact = ContactSensorCfg(
    name="body_contact",
    primary=ContactMatch(
      mode="geom",
      pattern=(BAD_CONTACT_GEOMS if xml_path == V12_XML else BAD_CONTACT_GEOMS_SOLID),
      entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("force",),
    reduce="netforce",
    num_slots=1,
    history_length=2,
  )

  ##
  # 事件（reset + 域随机化，对齐 Isaac 版）
  ##
  events = {
    "reset_base": EventTerm(
      func=envs_mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {"yaw": (-0.3, 0.3), "z": (0.0, 0.02)},
        "velocity_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2), "yaw": (-0.2, 0.2)},
      },
    ),
    "reset_robot_joints": EventTerm(
      func=mdp.ResetLegJoints,
      mode="reset",
      params={"position_range": (-0.02, 0.02)},
    ),
    "wheel_friction": EventTerm(
      mode="startup",
      func=envs_mdp.dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=WHEEL_GEOMS),
        "operation": "abs",
        "axes": [0],
        "ranges": (0.6, 1.4),
      },
    ),
    # base_link 标称 15.641 kg = CAD 5.641 + 10 kg 载荷；启动时再 ±3 kg。
    # dr.body_mass 只改质量、不改惯量，等价于在质心加点质量（见 mjlab 文档）。
    "base_mass": EventTerm(
      mode="startup",
      func=envs_mdp.dr.body_mass,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=("base_link",)),
        "operation": "add",
        "ranges": (-3.0, 3.0),
      },
    ),
  }

  ##
  # 奖励（weight × 公式，逐项 = Isaac 版）
  ##
  # 每项 = clip(weight × 公式, ±1)（legged_gym 语义），再由 scale_rewards_by_dt
  # 统一乘 dt（= plane_v11 的 clip(·, ±1·dt)）。
  _W = {
    "track_lin_vel": (mdp.rew_track_lin_vel, 1.0),
    "track_yaw_vel": (mdp.rew_track_yaw_vel, 1.0),
    "stand_still": (mdp.rew_stand_still, -1.0),
    "yaw_hold": (mdp.rew_yaw_hold, -1.0),
    "height": (mdp.rew_height, 1.0),
    "theta0_sync": (mdp.rew_theta0_sync, -2.0),
    "lin_vel_z": (mdp.rew_lin_vel_z, -2.0),
    "ang_vel_xy": (mdp.rew_ang_vel_xy, -0.6),
    "upright": (mdp.rew_upright, -100.0),
    "upright_fine": (mdp.rew_upright_fine, -2.0),
    # 起立过猛的代价权衡（2026-11-01 晚）：1.5e-3/8e-4 的腿速惩罚 + lin_vel_z -3
    # 把稳定时间拖到 1.5s（目标 0.2s）、刹车拖到 1.6s、站姿蹲低 4~8°——运动
    # 惩罚过重连高度修正都变懒。回滚到原始 4e-4/-2.0，让修好的三段式指标
    # 诚实裁判起立猛度；若再过冲，用数据重新找平衡点。
    "hip_knee_vel": (mdp.rew_hip_knee_vel, -4.0e-4),
    "dof_acc": (mdp.rew_dof_acc, -3.0e-6),
    "torque": (mdp.rew_torque, -8.0e-4),
    "action_rate": (mdp.rew_action_rate, -0.12),
    "action_smooth": (mdp.rew_action_smooth, -0.12),
    "bad_contact": (mdp.rew_bad_contact, -1.0),
  }
  if min_base_height is not None:
    # 实心变体专用：单向欠高惩罚，治"低 5 cm 站稳"的稳定解（见 rew_height_under）。
    _W["height_under"] = (mdp.rew_height_under, -5.0)
  rewards = {
    name: Reward(func=mdp.clipped(fn, w), weight=1.0) for name, (fn, w) in _W.items()
  }

  terminations = {
    "time_out": Termination(func=envs_mdp.time_out, time_out=True),
    "sustained_tilt": Termination(
      func=mdp.SustainedTilt,
      params={"tilt_cos": -0.1, "duration_s": 1.0},
    ),
  }
  if min_base_height is not None:
    # 实心变体专用：低伏贴地（腿折进底盘）直接判死，见 BaseHeightTermination。
    terminations["base_too_low"] = Termination(
      func=mdp.BaseHeightTermination,
      params={"min_height": float(min_base_height), "duration_s": 0.3},
    )
  if bad_contact_termination:
    # 专治 POM 导轮支撑的 27° 伪站姿（v8 实测 ep_len 2000 但 bad_contact -0.975）。
    # 20 N/0.3 s 会把早期普通落地接触也判死（v9 ep_len 仅 60），放宽到 30 N/0.5 s；
    # POM 27° 支撑的接触力通常更高，仍会触发。
    terminations["bad_contact"] = Termination(
      func=mdp.SustainedBadContact,
      params={"force_threshold": 30.0, "duration_s": 0.5},
    )

  cfg = ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      entities={"robot": get_v12_robot_cfg(xml_path)},
      sensors=(body_contact,),
      num_envs=num_envs,
      extent=2.0,
    ),
    observations={
      # 25 维本体观测 × 5 帧历史 = 125 维（对齐 plane_v11 的 obs_history_length=5：
      # 25 维里没有机体线速度，欠驱动平衡必须靠时序自估速度）。
      # 组级 history_length 覆盖所有 term；展开序 = term 分块、帧旧→新。
      "actor": ObservationGroupCfg(
        terms=actor_terms,
        concatenate_terms=True,
        enable_corruption=True,
        history_length=5,
      ),
      "critic": ObservationGroupCfg(
        terms=critic_terms,
        concatenate_terms=True,
        enable_corruption=False,
        history_length=5,
      ),
    },
    actions={"wheelleg": WheelLegActionCfg(entity_name="robot")},
    commands={
      "cmd": LocomotionCommandCfg(
        resampling_time_range=(5.0, 5.0),
        lin_vel_range=(-1.2, 1.2),
        ang_vel_range=(-1.0, 1.0),
        height_range=cmd_height_range,
        standing_prob=0.5,
      )
    },
    events=events,
    rewards=rewards,
    terminations=terminations,
    metrics={
      "height_err": Metric(func=mdp.metric_height_err),
      "speed_abs": Metric(func=mdp.metric_speed_abs),
      "stand_drift": Metric(func=mdp.metric_stand_drift),
      "stand_height": Metric(func=mdp.metric_stand_height),
      "stand_height_cmd": Metric(func=mdp.metric_stand_height_cmd),
      "rise_time": Metric(func=mdp.RiseSettle, reduce="last", params={"mode": "rise"}),
      "settle_time": Metric(func=mdp.RiseSettle, reduce="last", params={"mode": "settle"}),
    },
    curriculum={
      # 两段课程：全站定 → 站稳（近期平均 ep_len ≥ 8 s）后放开速度指令
      "command_standing": CurriculumTermCfg(
        func=mdp.CommandStandingCurriculum,
        # 默认 Stage 0 固定 0.19（5.641 kg 模型），Stage 1 放开 0.14~0.24。
        # 15.641 kg 的 solid 任务由调用方传 (0.16,0.16)/(0.12,0.18)。
        params={"stages": (1.0, 0.5), "ep_len_threshold": 800.0, "window": 400,
                "height_ranges": height_ranges,
                # 指令范围课程（stage 1 起）：禁用收窄（init=max，坑14：收窄
                # 会让蒸馏策略忘掉快速走行，且跟踪指标被低速指令美化）
                "cmd_lin_init": 1.2, "cmd_lin_max": 1.2,
                "cmd_ang_init": 1.0, "cmd_ang_max": 1.0,
                "cmd_step": 0.1, "track_threshold": 0.85, "cmd_cooldown_iters": 50},
      ),
    },
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="base_link",
      distance=1.5,
      elevation=-10.0,
      azimuth=90.0,
    ),
    sim=SimulationCfg(
      # mujoco-warp 的接触/约束缓冲是静态分配（溢出=非法访存）。15.641 kg
      # 重模型摔倒时接触暴增，192/1024 不够稳，回到 256/2048。
      nconmax=256,
      njmax=2048,
      contact_sensor_maxmatch=128,
      mujoco=MujocoCfg(timestep=0.005, integrator="euler"),
    ),
    decimation=2,
    episode_length_s=20.0,
    scale_rewards_by_dt=True,  # legged_gym 语义：总和乘 dt（配合每项 ±1 裁剪）
  )

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("wheel_friction", None)
    cfg.events.pop("base_mass", None)

  return cfg


# ---------------------------------------------------------------------------
# 跳跃任务（冻结平地策略 + 残差）
# ---------------------------------------------------------------------------

V12_GAS_XML = _REPO / "mjcf" / "rl_chassis_v12_gas.xml"
assert V12_GAS_XML.exists(), f"missing {V12_GAS_XML}"


def get_v12_gas_robot_cfg() -> "object":
  """带气弹簧的 v12 实体（气弹簧电机不注册，训练中用解析力矩代替）。"""
  from mjlab.actuator import XmlActuatorCfg
  from mjlab.entity import EntityCfg, EntityArticulationInfoCfg

  joint_pos: dict[str, float] = {}
  for side in ("left", "right"):
    for name, angle in zip(JOINT_NAMES[side], CLOSED_DEFAULT[side]):
      joint_pos[name] = angle
    joint_pos[WHEEL_JOINT_NAMES[side]] = 0.0

  def spec_fn() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(V12_GAS_XML))
    if len(spec.equalities) != 4:
      raise RuntimeError(f"v12 gas 闭链 equality 数量异常: {len(spec.equalities)}")
    return spec

  driven_joints = (
    "Left_front_joint", "Left_rear_joint", "Left_Wheel_joint",
    "Right_front_joint", "Right_rear_joint", "Right_Wheel_joint",
  )
  return EntityCfg(
    spec_fn=spec_fn,
    articulation=EntityArticulationInfoCfg(
      actuators=tuple(XmlActuatorCfg(target_names_expr=(n,)) for n in driven_joints),
    ),
    init_state=EntityCfg.InitialStateCfg(
      pos=(0.0, 0.0, INIT_HEIGHT), joint_pos=joint_pos, joint_vel={".*": 0.0}),
  )


def wheelleg_jump_env_cfg(play: bool = False, num_envs: int = 1024) -> ManagerBasedRlEnvCfg:
  from mjlab.sensor import ContactSensorCfg as _CSC

  from . import jump_task as jt

  actor_terms = {
    "obs_current": Obs(func=jt.jump_obs_current, clip=(-100.0, 100.0)),
    "phase": Obs(func=jt.jump_obs_phase),
    "phase_progress": Obs(func=jt.jump_obs_progress),
    "contact_conf": Obs(func=jt.jump_obs_contact),
    "torque_fb": Obs(func=jt.jump_obs_torque),
    "last_residual": Obs(func=jt.jump_obs_last_residual, clip=(-100.0, 100.0)),
  }
  critic_terms = dict(actor_terms)

  body_contact = ContactSensorCfg(
    name="body_contact",
    primary=ContactMatch(mode="geom", pattern=BAD_CONTACT_GEOMS, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("force",),
    reduce="netforce",
    num_slots=1,
    history_length=2,
  )
  wheel_contact = ContactSensorCfg(
    name="wheel_contact",
    primary=ContactMatch(mode="geom", pattern=".*_Wheel_link_col", entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="maxforce",
    num_slots=1,
  )

  events = {
    "reset_base": EventTerm(
      func=envs_mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {"yaw": (-0.3, 0.3), "z": (0.0, 0.02)},
        "velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3), "yaw": (-0.2, 0.2)},
      },
    ),
    "reset_robot_joints": EventTerm(
      func=mdp.ResetLegJoints,
      mode="reset",
      params={"position_range": (-0.02, 0.02)},
    ),
    "wheel_friction": EventTerm(
      mode="startup",
      func=envs_mdp.dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=WHEEL_GEOMS),
        "operation": "abs", "axes": [0], "ranges": (0.6, 1.4),
      },
    ),
    # 与平地任务同一口径：base_link 15.641 kg ±3 kg（不改惯量）。
    "base_mass": EventTerm(
      mode="startup",
      func=envs_mdp.dr.body_mass,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=("base_link",)),
        "operation": "add", "ranges": (-3.0, 3.0),
      },
    ),
  }

  # 跳跃奖励表（对齐 jump_policy.py 的 scales；平地跟踪项保留）
  _W = {
    # 跳跃
    "flight": (jt.rew_jump_flight, 0.30),
    "encourage_jump": (jt.rew_jump_encourage, 1.5),
    "height_flight": (jt.rew_jump_height_flight, 15.0),
    "leg_tuck": (jt.rew_jump_leg_tuck, 1.7),
    "takeoff_extend": (jt.rew_jump_takeoff_extend, 0.5),
    "line_z": (jt.rew_jump_line_z, 12.0),
    "landing_soft": (jt.rew_landing_soft, 2.0),
    "landing_stable": (jt.rew_landing_stable, 3.0),
    "landing_ang_vel": (jt.rew_landing_ang_vel, -1.5),
    "landing_lin_vel_z": (jt.rew_landing_lin_vel_z, -3.0),
    "landing_height": (jt.rew_landing_height, -2.0),
    "landing_peak": (jt.rew_landing_peak, 25.0),
    "landing_low_peak": (jt.rew_landing_low_peak, -15.0),
    # 飞行控制
    "wheel_spin_flight": (jt.rew_wheel_spin_flight, -0.2),
    "wheel_speed_match": (jt.rew_wheel_speed_match, 0.5),
    "residual_magnitude": (jt.rew_residual_magnitude, -0.01),
    # 平地跟踪 / 稳定（复用平地任务项）
    "track_lin_vel": (mdp.rew_track_lin_vel, 1.0),
    "track_yaw_vel": (mdp.rew_track_yaw_vel, 1.0),
    "lin_vel_z": (mdp.rew_lin_vel_z, -2.0),
    "ang_vel_xy": (mdp.rew_ang_vel_xy, -0.6),
    "upright": (mdp.rew_upright, -100.0),
    "upright_fine": (mdp.rew_upright_fine, -2.0),
    "theta0_sync": (mdp.rew_theta0_sync, -2.0),
    "hip_knee_vel": (mdp.rew_hip_knee_vel, -4.0e-4),
    "dof_acc": (mdp.rew_dof_acc, -3.0e-6),
    "torque": (mdp.rew_torque, -5.0e-5),
    "bad_contact": (mdp.rew_bad_contact, -1.0),
    "action_rate": (mdp.rew_action_rate, -0.04),
    "action_smooth": (mdp.rew_action_smooth, -0.04),
  }
  rewards = {name: Reward(func=mdp.clipped(fn, w), weight=1.0)
             for name, (fn, w) in _W.items()}

  cfg = ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      entities={"robot": get_v12_gas_robot_cfg()},
      sensors=(body_contact, wheel_contact),
      num_envs=num_envs,
      extent=2.0,
    ),
    observations={
      "actor": ObservationGroupCfg(terms=actor_terms, concatenate_terms=True,
                                   enable_corruption=False),
      "critic": ObservationGroupCfg(terms=critic_terms, concatenate_terms=True,
                                    enable_corruption=False),
    },
    # 轮 kd 与部署一致（0.6）：站立轮微颤在部署端被调高阻尼压掉，训练同步
    actions={"jump": jt.JumpResidualActionCfg(entity_name="robot", wheel_kd=0.6)},
    commands={
      "cmd": LocomotionCommandCfg(
        resampling_time_range=(5.0, 5.0),
        lin_vel_range=(-1.2, 1.2),
        ang_vel_range=(-1.0, 1.0),
        height_range=(0.14, 0.24),
        standing_prob=0.5,
      )
    },
    events=events,
    rewards=rewards,
    terminations={
      "time_out": Termination(func=envs_mdp.time_out, time_out=True),
      "fall": Termination(func=jt.jump_fall),
    },
    metrics={
      "jump_success": Metric(func=jt.jump_success),
      "height_err": Metric(func=mdp.metric_height_err),
    },
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot", body_name="base_link",
      distance=1.5, elevation=-10.0, azimuth=90.0,
    ),
    sim=SimulationCfg(
      nconmax=256, njmax=2048, contact_sensor_maxmatch=128,
      mujoco=MujocoCfg(timestep=0.005, integrator="euler"),
    ),
    decimation=2,
    episode_length_s=20.0,
    scale_rewards_by_dt=True,
  )
  if play:
    cfg.episode_length_s = int(1e9)
    cfg.events.pop("base_mass", None)  # 与平地任务一致：play 用标称 15.641 kg
  return cfg
