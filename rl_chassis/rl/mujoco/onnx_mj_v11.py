#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v11 sim2sim：把 Isaac Gym 训练好的 ONNX 策略跑在 v11 的 MuJoCo **闭链**模型上。

对标 fudan_rl_wheel_leg/mujoco/python_tools/onnx_mj_binglian.py，差异：
  * 机器人换成 rl_chassis_v11（前髋=曲柄、后髋=大腿），状态/力矩映射用
    rl/tools/leg_model.py 里的闭环逆解与虚拟腿雅可比；
  * 我们没有气弹簧，去掉 gas spring 那部分；
  * 观测/动作尺度、增益、默认角全部与本仓库 plane_v11 的训练配置逐项一致。

用法：
  python onnx_mj_v11.py --onnx v11_plane.onnx                     # 策略
  python onnx_mj_v11.py --zero-action --duration 5                # 只做 PD 站立自检
  python onnx_mj_v11.py --onnx ... --cmd-lin-vel 1.0 --duration 8 --no-viewer
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

import mujoco
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
from leg_model import V11Leg, load as load_legs, wrap_to_pi  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROBOT_DIR = os.path.abspath(os.path.join(HERE, "..", ".."))
DEFAULT_XML = os.path.join(ROBOT_DIR, "mjcf", "rl_chassis_v12_scene.xml")

# ---- 与 plane_v11 训练配置逐项一致 ----------------------------------------
DOF_ORDER = ["rear_L", "knee_L", "wheel_L", "rear_R", "knee_R", "wheel_R"]
DEFAULT_DOF = np.array([-0.274884, 0.517721, 0.0, 0.274883, -0.517722, 0.0], dtype=np.float64)
KP = np.array([40.0, 40.0, 0.0, 40.0, 40.0, 0.0])
# 轮 kd 0.3 -> 0.6：站立时两轮主动平衡的轮速微颤（±5 rad/s）看起来像"抖"，
# 提高轮阻尼压掉 60%（实测 5.3->2.2 rad/s，高度稳定性反而更好，滚动跟踪不变）。
# 注意 1.0 会进极限环（10ms 延迟 + 高增益），0.6 是安全上限。
KD = np.array([3.0, 3.0, 0.3, 3.0, 3.0, 0.3])   # 第四版起：kd 2→3（压延迟引起的腿抖）
TAU_LIMIT = np.array([40.0, 40.0, 20.0, 40.0, 40.0, 20.0])
POS_ACTION_SCALE = 0.5
VEL_ACTION_SCALE = 10.0
# 与训练动作项一致：物理动作裁到 ±1（±0.5 rad / ±10 rad/s）。
# 旧值 100 只防 PPO 数值爆炸，挡不住重质量下的奇异目标。
CLIP_ACTIONS = 1.0
CLIP_OBS = 100.0
OBS_SCALE_ANG_VEL = 0.25
OBS_SCALE_DOF_POS = 1.0
OBS_SCALE_DOF_VEL = 0.05
COMMAND_SCALE = np.array([2.0, 0.25, 5.0])  # = cfg.normalization.obs_scales 里的 lin/ang/height
OBS_DOF_IDX = [0, 1, 3, 4]
HISTORY_LEN = 5
# 高度指令范围与训练配置一致（第四版/平地版是 0.14~0.24），
# 键盘按 5/6 也只会在这个范围内调，避免按出训练分布外导致失稳。
CMD_RANGES = dict(lin_vel=(-2.5, 2.5), yaw_vel=(-2.5, 2.5), height=(0.14, 0.24))
POLICY_DT = 0.01     # decimation 2 × dt 0.005
SIM_DT = 0.005
WHEEL_R = 0.06
HIP_OFFSET_Z = 0.11508   # base_link 原点 -> 髋关节
TRACK_WIDTH = 0.374      # 左右轮心间距 [m]
POSE_HIP_KP = 120.0      # f 模式锁髋增益（单独设置，不影响策略 PD）
POSE_HIP_KD = 8.0

# ---- 键盘操控（和 fudan 的 sim2sim 一致）----------------------------------
CMD_LIN_MAX = 2.0
CMD_YAW_MAX = 2.0
CMD_RAMP_TIME = 0.5   # 指令斜坡时间 [s]
HEIGHT_STEP = 0.01


class CommandState:
    """键盘指令状态机：按住 1/2/3/4 前进/后退/左转/右转，5/6 抬高/降低，
    0 停，b 全部失能/恢复，f 切换“姿态保持 + 轮差速”/策略模式。"""

    def __init__(self, lin=0.0, yaw=0.0, height=0.15):
        self.keys = set()
        self.lin = lin
        self.yaw = yaw
        self.lin_target = lin
        self.yaw_target = yaw
        self.height = float(np.clip(height, *CMD_RANGES["height"]))
        self.running = True
        self.mode = "policy"   # policy / pose / off
        self.listener = None

    def refresh(self):
        fwd = "1" in self.keys
        back = "2" in self.keys
        left = "3" in self.keys
        right = "4" in self.keys
        self.lin_target = CMD_LIN_MAX if fwd and not back else (-CMD_LIN_MAX if back and not fwd else 0.0)
        self.yaw_target = CMD_YAW_MAX if left and not right else (-CMD_YAW_MAX if right and not left else 0.0)

    def on_press(self, key):
        try:
            name = key.char
        except AttributeError:
            name = None
        if key == __import__("pynput").keyboard.Key.esc:
            self.running = False
            return False
        if name in {"1", "2", "3", "4"}:
            self.keys.add(name)
            self.refresh()
        elif name == "0":
            self.keys.clear()
            self.refresh()
        elif name == "5":
            self.height = min(CMD_RANGES["height"][1], self.height + HEIGHT_STEP)
            print(f"[CMD] height={self.height:.3f}")
        elif name == "6":
            self.height = max(CMD_RANGES["height"][0], self.height - HEIGHT_STEP)
            print(f"[CMD] height={self.height:.3f}")
        elif name == "b":
            self.mode = "policy" if self.mode == "off" else "off"
            print(f"[MODE] {self.mode}（b = 全部失能/恢复）")
        elif name == "f":
            self.mode = "policy" if self.mode == "pose" else "pose"
            print(f"[MODE] {self.mode}（f = 姿态保持 + 轮差速/策略）")
        elif name == "q":
            self.running = False
            return False
        return None

    def on_release(self, key):
        try:
            name = key.char
        except AttributeError:
            return
        if name in {"1", "2", "3", "4"}:
            self.keys.discard(name)
            self.refresh()

    def start_listener(self):
        try:
            from pynput import keyboard
            self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
            self.listener.start()
        except Exception as exc:  # 没有可用键盘后端（例如纯 Wayland）也不该让仿真挂掉
            self.listener = None
            print(f"[KEY] 键盘监听不可用（{exc}），只能看不能操控")

    def step_ramp(self, dt):
        def ramp(cur, tgt, limit):
            max_step = limit / CMD_RAMP_TIME * dt
            return cur + float(np.clip(tgt - cur, -max_step, max_step))
        self.lin = ramp(self.lin, self.lin_target, CMD_LIN_MAX)
        self.yaw = ramp(self.yaw, self.yaw_target, CMD_YAW_MAX)


def quat_rotate_inverse(q_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
    """base 系下的向量（q 是 base 的姿态，wxyz）。"""
    w, x, y, z = q_wxyz
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    return R.T @ v


class LegCoupler:
    """一条腿：真机 (rear, front) <-> 虚拟 (rear, knee) 的状态/力矩耦合。"""

    def __init__(self, leg: V11Leg, side: str):
        self.leg = leg
        self.side = side
        self.seed = None
        self.reset_seed()

    def reset_seed(self):
        q, res, seed = self.leg.solve_loop(0.0, 0.0)
        self.seed = seed

    def virtual_state(self, q_front: float, q_rear: float, v_front: float, v_rear: float):
        """返回 (q_knee, dq_knee, dq_knee_dq_rear, dq_knee_dq_front)。"""
        q, res, self.seed = self.leg.solve_loop(q_front, q_rear, seed=self.seed)
        d_rear, d_front = self.leg.jacobian(q_front, q_rear, seed=self.seed)
        q_knee = q["shank"]
        dq_knee = d_rear * v_rear + d_front * v_front
        return q_knee, dq_knee, d_rear, d_front, res

    @staticmethod
    def torque_map(tau_rear_v: float, tau_knee_v: float, d_rear: float, d_front: float):
        """虚拟 (大腿, 膝) 力矩 -> 真机 (后髋, 前髋曲柄) 力矩（虚功原理）。"""
        return tau_rear_v + tau_knee_v * d_rear, tau_knee_v * d_front


class V11Sim2Sim:
    def __init__(self, args):
        self.args = args
        self.m = mujoco.MjModel.from_xml_path(args.xml)
        self.d = mujoco.MjData(self.m)
        self.m.opt.timestep = SIM_DT
        _, leg_l, leg_r = load_legs(args.xml)
        self.legs = {"L": leg_l, "R": leg_r}
        self.couplers = {s: LegCoupler(l, s) for s, l in self.legs.items()}
        # 驱动器
        self.act = {}
        for s, names in (("L", ("Left_front_joint_motor", "Left_rear_joint_motor", "Left_Wheel_joint_motor")),
                         ("R", ("Right_front_joint_motor", "Right_rear_joint_motor", "Right_Wheel_joint_motor"))):
            self.act[s] = {
                k: mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
                for k, n in zip(("front", "rear", "wheel"), names)
            }
        self.dof_adr = {
            k: (int(self.m.jnt_qposadr[v]), int(self.m.jnt_dofadr[v]))
            for k, v in self.couplers["L"].leg.ids.items()
        }
        self.dof_adr_R = {
            k: (int(self.m.jnt_qposadr[v]), int(self.m.jnt_dofadr[v]))
            for k, v in self.couplers["R"].leg.ids.items()
        }
        if args.onnx:
            import onnxruntime as ort
            prov = ["CUDAExecutionProvider", "CPUExecutionProvider"] if args.device == "cuda" else ["CPUExecutionProvider"]
            try:
                self.sess = ort.InferenceSession(args.onnx, providers=prov)
            except Exception:
                self.sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
            self.in_names = [i.name for i in self.sess.get_inputs()]
        else:
            self.sess = None
        self.history = np.zeros((HISTORY_LEN, 25), dtype=np.float32)
        self.last_actions = np.zeros(6, dtype=np.float32)
        self.last_dof_pos = None
        self.cmd = CommandState(args.cmd_lin_vel, args.cmd_yaw_vel, args.cmd_height)
        # 真机不那么快的两个建模：动作延迟（策略周期整数倍）+ 执行器一阶滞后
        # 延迟队列长度必须正好等于 delay_steps：追加当前动作后，若队首
        # 多出 delay_steps 个元素才弹出。旧实现初始化 delay_steps+1 个零，
        # 并且每步无条件 pop，实际延迟成了 (delay_steps+1) 个策略周期
        # （默认 1 变成 20 ms，而不是训练配置里的 10 ms）。
        self.n_delay = max(0, int(args.delay_steps))
        self.action_fifo = [np.zeros(6, dtype=np.float32) for _ in range(self.n_delay)]
        self.tau_applied = np.zeros(6)
        self.pose_hold_target = None
        self.reset()

    # ---------------- 初始化 ----------------
    def default_joint_pos(self):
        """默认角对应的真机全关节角（含曲柄），用来摆初始位姿。"""
        out = {}
        for i, s in enumerate(("L", "R")):
            q_rear, q_knee = DEFAULT_DOF[3 * i], DEFAULT_DOF[3 * i + 1]
            q, res, _ = self.couplers[s].leg.solve_virtual(q_rear, q_knee)
            out[s] = q
        return out

    def reset(self):
        d = self.d
        mujoco.mj_resetData(self.m, d)
        full = self.default_joint_pos()
        for s in ("L", "R"):
            for k, (adr, _) in (self.dof_adr if s == "L" else self.dof_adr_R).items():
                d.qpos[adr] = full[s][k]
        # 初始 base 高度：让轮心正好离地 1 mm
        L0, th0 = self.couplers["L"].leg.virtual_leg(DEFAULT_DOF[0], DEFAULT_DOF[1])
        base_z = WHEEL_R + L0 * math.cos(th0) - HIP_OFFSET_Z + 0.001
        d.qpos[0:3] = [0.0, 0.0, base_z]
        # 注意：自由关节的四元数是 (w, x, y, z)！按 (x,y,z,w) 写会变成 180° 偏航，
        # 机器人就面朝后了——表现是策略看着"倒着走"。
        d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        mujoco.mj_forward(self.m, d)
        self.history[:] = 0.0
        self.last_actions[:] = 0.0
        self.tau_applied[:] = 0.0
        self.pose_hold_target = None
        self.action_fifo = [np.zeros(6, dtype=np.float32) for _ in range(self.n_delay)]
        self.last_dof_pos = None
        for c in self.couplers.values():
            c.reset_seed()

    # ---------------- 读状态 ----------------
    def read(self):
        d = self.d
        cmd = self.command_now()
        q = np.zeros(6)
        qd = np.zeros(6)
        jac = {}
        for i, s in enumerate(("L", "R")):
            adr = self.dof_adr if s == "L" else self.dof_adr_R
            q_front = float(d.qpos[adr["front"][0]])
            q_rear = float(d.qpos[adr["rear"][0]])
            v_front = float(d.qvel[adr["front"][1]])
            v_rear = float(d.qvel[adr["rear"][1]])
            if self.args.use_sim_knee:
                q_knee = float(d.qpos[adr["shank"][0]])
                d_rear, d_front = self.couplers[s].leg.jacobian(q_front, q_rear,
                                                                seed=self.couplers[s].seed)
                dq_knee = d_rear * v_rear + d_front * v_front
                res = 0.0
            else:
                q_knee, dq_knee, d_rear, d_front, res = self.couplers[s].virtual_state(
                    q_front, q_rear, v_front, v_rear)
            jac[s] = (d_rear, d_front, res)
            q[3 * i:3 * i + 3] = [q_rear, q_knee, float(d.qpos[adr["wheel"][0]])]
            qd[3 * i:3 * i + 3] = [v_rear, dq_knee, float(d.qvel[adr["wheel"][1]])]
        # 与训练一致：关节速度用位置差分（dof_vel_use_pos_diff=True）
        if self.last_dof_pos is not None:
            qd = (q - self.last_dof_pos + np.pi) % (2 * np.pi) - np.pi
            qd = qd / POLICY_DT
        self.last_dof_pos = q.copy()
        base_quat = np.array(d.qpos[3:7], dtype=np.float64)
        # 注意：MuJoCo 自由关节的 qvel[0:3] 是世界系线速度，qvel[3:6] 已经是**机体系**角速度，
        # 不能再旋转一次（Isaac 那边 root_states[10:13] 是世界系角速度，才需要转）。
        body_ang = np.array(d.qvel[3:6], dtype=np.float64)
        projected_g = quat_rotate_inverse(base_quat, np.array([0.0, 0.0, -1.0]))
        obs = np.concatenate([
            body_ang * OBS_SCALE_ANG_VEL,
            projected_g,
            cmd * COMMAND_SCALE,
            (q[OBS_DOF_IDX] - DEFAULT_DOF[OBS_DOF_IDX]) * OBS_SCALE_DOF_POS,
            qd * OBS_SCALE_DOF_VEL,
            self.last_actions,
        ]).astype(np.float32)
        obs = np.clip(obs, -CLIP_OBS, CLIP_OBS)
        return obs, q, qd, jac

    def command_now(self):
        return np.array([self.cmd.lin, self.cmd.yaw, self.cmd.height], dtype=np.float64)

    # ---------------- 控制 ----------------
    def _apply_actuator_lag(self, tau):
        tau = np.clip(tau, -TAU_LIMIT, TAU_LIMIT)
        if self.args.actuator_tau > 0.0:
            alpha = min(SIM_DT / self.args.actuator_tau, 1.0)
            tau = self.tau_applied + (tau - self.tau_applied) * alpha
        self.tau_applied = tau.copy()
        return tau

    def read_hip_pose(self):
        """当前四个髋电机角度，顺序 [L_front, L_rear, R_front, R_rear]。"""
        d = self.d
        return np.array([
            d.qpos[self.dof_adr["front"][0]],
            d.qpos[self.dof_adr["rear"][0]],
            d.qpos[self.dof_adr_R["front"][0]],
            d.qpos[self.dof_adr_R["rear"][0]],
        ], dtype=np.float64)

    def wheel_diff_torques(self, qd):
        """轮电机差速速度控制，髋电机输出保持为零。

        轮关节轴：左轮 +y、右轮 -y，所以前进时左轮目标为 +v/R、右轮为 -v/R。
        """
        v_left = self.cmd.lin - 0.5 * TRACK_WIDTH * self.cmd.yaw
        v_right = self.cmd.lin + 0.5 * TRACK_WIDTH * self.cmd.yaw
        tau = np.zeros(6)
        tau[2] = KD[2] * (v_left / WHEEL_R - qd[2])
        tau[5] = KD[5] * (-v_right / WHEEL_R - qd[5])
        return self._apply_actuator_lag(tau)

    def pose_hold_torques(self, qd):
        """f 模式：锁定按下 f 时的四个髋角，轮电机做差速速度控制。"""
        if self.pose_hold_target is None:
            self.pose_hold_target = self.read_hip_pose()
            print("[POSE] 锁定髋角 "
                  f"L=({self.pose_hold_target[0]:+.3f},{self.pose_hold_target[1]:+.3f}) "
                  f"R=({self.pose_hold_target[2]:+.3f},{self.pose_hold_target[3]:+.3f})",
                  flush=True)
        t = self.pose_hold_target
        d = self.d
        tau = np.zeros(6)
        hip_kp, hip_kd = POSE_HIP_KP, POSE_HIP_KD
        # 输出顺序：rear, front, wheel（左右各一组）
        tau[0] = hip_kp * (t[1] - d.qpos[self.dof_adr["rear"][0]]) - hip_kd * d.qvel[self.dof_adr["rear"][1]]
        tau[1] = hip_kp * (t[0] - d.qpos[self.dof_adr["front"][0]]) - hip_kd * d.qvel[self.dof_adr["front"][1]]
        tau[3] = hip_kp * (t[3] - d.qpos[self.dof_adr_R["rear"][0]]) - hip_kd * d.qvel[self.dof_adr_R["rear"][1]]
        tau[4] = hip_kp * (t[2] - d.qpos[self.dof_adr_R["front"][0]]) - hip_kd * d.qvel[self.dof_adr_R["front"][1]]
        # 轮子仍按 1/2/3/4 指令差速
        v_left = self.cmd.lin - 0.5 * TRACK_WIDTH * self.cmd.yaw
        v_right = self.cmd.lin + 0.5 * TRACK_WIDTH * self.cmd.yaw
        tau[2] = KD[2] * (v_left / WHEEL_R - qd[2])
        tau[5] = KD[5] * (-v_right / WHEEL_R - qd[5])
        return self._apply_actuator_lag(tau)

    def compute_torques(self, actions, q, qd, jac):
        if self.cmd.mode == "off":
            self.pose_hold_target = None
            self.tau_applied[:] = 0.0
            return np.zeros(6)
        if self.cmd.mode == "pose":
            return self.pose_hold_torques(qd)
        self.pose_hold_target = None
        pos_ref = actions * POS_ACTION_SCALE
        pos_ref[[2, 5]] = 0.0
        vel_ref = actions * VEL_ACTION_SCALE
        vel_ref[[0, 1, 3, 4]] = 0.0
        tau_v = KP * (pos_ref + DEFAULT_DOF - q) + (KD * self.args.kd_scale) * (vel_ref - qd)
        tau_v = np.clip(tau_v, -TAU_LIMIT, TAU_LIMIT)
        out = np.zeros(6)
        for i, s in enumerate(("L", "R")):
            d_rear, d_front, _ = jac[s]
            tau_rear_real, tau_front_real = LegCoupler.torque_map(
                tau_v[3 * i], tau_v[3 * i + 1], d_rear, d_front)
            out[3 * i] = tau_rear_real
            out[3 * i + 1] = tau_front_real
            out[3 * i + 2] = tau_v[3 * i + 2]
        return self._apply_actuator_lag(out)

    def apply(self, tau):
        self.d.ctrl[:] = 0.0
        if self.cmd.mode == "off":
            return
        for i, s in enumerate(("L", "R")):
            self.d.ctrl[self.act[s]["rear"]] = tau[3 * i]
            self.d.ctrl[self.act[s]["front"]] = tau[3 * i + 1]
            self.d.ctrl[self.act[s]["wheel"]] = tau[3 * i + 2]

    # ---------------- 主循环 ----------------
    def obs_history(self, obs):
        self.history[:-1] = self.history[1:]
        self.history[-1] = obs
        return self.history.reshape(1, -1).astype(np.float32)

    def run(self):
        args = self.args
        viewer = None if args.no_viewer else __import__("mujoco.viewer", fromlist=["launch_passive"]).launch_passive(self.m, self.d)
        t0 = time.time()
        step = 0
        vxs, heights = [], []
        dump_rows = []
        if not args.no_viewer:
            import mujoco.viewer as _viewer_mod
            del _viewer_mod
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            viewer.cam.distance = 2.2
            viewer.cam.azimuth = 140.0
            viewer.cam.elevation = -18.0
            self.cmd.start_listener()
            print("[KEY] 按住 1 前进 / 2 后退 / 3 左转 / 4 右转，5 抬高 / 6 降低，0 停，"
                  "b 全部失能/恢复，f 姿态保持+轮差速/策略，q 或 Esc 退出",
                  flush=True)
        next_t = time.perf_counter()
        last_sync = 0.0
        while (args.duration <= 0 or self.d.time < args.duration) and self.cmd.running:
            self.cmd.step_ramp(POLICY_DT)
            obs, q, qd, jac = self.read()
            hist = self.obs_history(obs)
            if self.sess is None or args.zero_action:
                act = np.zeros(6, dtype=np.float32)
            else:
                feed = {self.in_names[0]: obs.reshape(1, -1).astype(np.float32)}
                # 旧策略是 25 + 125 双输入；Stage 1 的平地策略没有 history encoder。
                if len(self.in_names) > 1:
                    feed[self.in_names[1]] = hist
                act = np.clip(self.sess.run(None, feed)[0].reshape(-1),
                              -CLIP_ACTIONS, CLIP_ACTIONS).astype(np.float32)
            self.last_actions = act
            # 动作延迟：PD 用的是 N 个策略周期之前的动作（obs 里的"上一步动作"仍是最新的）
            self.action_fifo.append(act.astype(np.float32))
            if len(self.action_fifo) > self.n_delay:
                act_ctrl = self.action_fifo.pop(0)
            else:
                act_ctrl = np.zeros(6, dtype=np.float32)
            # 注意：训练里 PD 与执行器滞后都是**每个仿真子步**算一次（decimation=2），
            # 这里也必须按子步算，否则一阶滞后的等效时间常数会变成 2 倍（比训练更苛刻）。
            for k_sub in range(int(round(POLICY_DT / SIM_DT))):
                tau = self.compute_torques(act_ctrl, q, qd, jac)
                if args.dump and k_sub == 0:
                    # t, obs[0..24], act[0..5], q[0..5], qd[0..5], tau[0..5]
                    dump_rows.append(np.concatenate(([self.d.time], obs, act, q, qd, tau)))
                self.apply(tau)
                mujoco.mj_step(self.m, self.d)
            if viewer is not None:
                now = time.perf_counter()
                if now - last_sync >= 1.0 / 60.0:   # 显示限 60 Hz，别按物理步刷新
                    viewer.cam.lookat[:] = self.d.qpos[:3]   # 跟随机位
                    viewer.sync()
                    last_sync = now
                # 1x 墙钟节流；落后时重新对齐，不追帧
                next_t += POLICY_DT
                wait = next_t - time.perf_counter()
                if wait > 0:
                    time.sleep(wait)
                else:
                    next_t = time.perf_counter()
            step += 1
            if step % max(1, int(1.0 / POLICY_DT)) == 0:
                L0L, th0L = self.couplers["L"].leg.virtual_leg(q[0], q[1])
                L0R, th0R = self.couplers["R"].leg.virtual_leg(q[3], q[4])
                bh = float(self.d.qpos[2])
                vx = float(self.d.qvel[0])
                vxs.append(vx)
                heights.append(bh)
                w, x, y, z = self.d.qpos[3:7]
                tilt = math.degrees(math.acos(max(-1.0, min(1.0, 1 - 2 * (x * x + y * y)))))
                yaw = math.degrees(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))
                tail = self.floor_contacts()
                print(f"[{self.d.time:5.2f}s] x={self.d.qpos[0]:+.2f} y={self.d.qpos[1]:+.2f} yaw={yaw:+.1f}deg h={bh:.4f} m  vx={vx:+.3f} m/s  "
                      f"L0=({L0L:.4f},{L0R:.4f})  theta0=({math.degrees(th0L):+.1f},{math.degrees(th0R):+.1f})deg  "
                      f"cmd=({self.cmd.lin:+.2f},{self.cmd.yaw:+.2f},{self.cmd.height:.3f})  "
                      f"mode={self.cmd.mode}  "
                      f"倾斜={tilt:.1f}deg  接地={tail}", flush=True)
            if viewer is not None and not viewer.is_running():
                break
        wall = time.time() - t0
        print(f"\n== 结束（窗口关闭 / 按了 q / 到了 --duration）: 仿真 {self.d.time:.2f} s / "
              f"实际 {wall:.2f} s (实时倍率 {self.d.time / max(wall,1e-6):.2f}x)")
        if heights:
            yaw = float(self.d.qvel[5])
            print(f"   高度 均值 {np.mean(heights):.4f} m（目标 {args.cmd_height:.3f}）")
            print(f"   前进速度 均值 {np.mean(vxs):+.3f} m/s（指令 {args.cmd_lin_vel:+.2f}）")
            print(f"   末态偏航角速度 {yaw:+.3f} rad/s（指令 {args.cmd_yaw_vel:+.2f}）")
        if args.dump and dump_rows:
            arr = np.array(dump_rows)
            header = (["t"] + [f"obs{i}" for i in range(25)] + [f"act{i}" for i in range(6)]
                      + [f"q{i}" for i in range(6)] + [f"qd{i}" for i in range(6)]
                      + [f"tau{i}" for i in range(6)])
            np.savetxt(args.dump, arr, delimiter=",", header=",".join(header), comments="",
                       fmt="%.6f")
            print(f"   已导出 {arr.shape[0]} 行参考数据 -> {args.dump}（列：t, obs0..24, act0..5, q0..5, qd0..5, tau0..5）")
        return 0

    def floor_contacts(self):
        """统计接触：轮子踩在什么上面、以及非轮子零件是否蹭到东西。"""
        wheels, ground, other = {}, {}, {}
        for c in self.d.contact:
            n1 = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or f"g{c.geom1}"
            n2 = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or f"g{c.geom2}"
            if "Wheel" in n1 or "Wheel" in n2:
                sup = n2 if "Wheel" in n1 else n1
                key = sup.split("_")[0].rstrip("0123456789") or sup
                wheels[key] = wheels.get(key, 0) + 1
            elif "floor" in n1 or "floor" in n2:
                o = n2 if "floor" in n1 else n1
                other[o] = other.get(o, 0) + 1
            else:
                other[f"{n1}~{n2}"] = other.get(f"{n1}~{n2}", 0) + 1
        txt = "轮->" + ",".join(f"{k}×{v}" for k, v in sorted(wheels.items()))
        if other:
            txt += " | 其它:" + ",".join(f"{k}×{v}" for k, v in sorted(other.items()))
        return txt


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", default=DEFAULT_XML)
    ap.add_argument("--onnx", default=None, help="不指定则只做 PD 站立自检")
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    ap.add_argument("--duration", type=float, default=None,
                    help="仿真时长 [s]，0 = 一直跑到你退出；不指定时：开窗口=一直跑，--no-viewer=6 s")
    ap.add_argument("--cmd-lin-vel", type=float, default=0.0)
    ap.add_argument("--cmd-yaw-vel", type=float, default=0.0)
    ap.add_argument("--cmd-height", type=float, default=0.15)
    ap.add_argument("--zero-action", action="store_true", help="忽略策略，动作恒为 0")
    ap.add_argument("--use-sim-knee", action="store_true",
                    help="直接用仿真里的膝角（默认按真机口径用闭环逆解）")
    ap.add_argument("--no-viewer", action="store_true")
    ap.add_argument("--dump", default=None, help="把每步的 obs/动作/关节状态/力矩写成 CSV，供单片机逐维对照")
    # 默认值 = 训练时的"慢电机"条件（v11_terrain 起）：延迟 0~10 ms 随机、一阶滞后 15 ms。
    # 想看理想电机就显式传 --delay-steps 0 --actuator-tau 0。
    ap.add_argument("--delay-steps", type=int, default=1,
                    help="动作延迟（策略周期数，1 = 10 ms）；默认 1 与训练上限一致")
    ap.add_argument("--actuator-tau", type=float, default=0.015,
                    help="执行器一阶滞后时间常数 [s]；默认 0.015 = 训练值，0 = 理想力矩源")
    ap.add_argument("--kd-scale", type=float, default=1.0,
                    help="kd 缩放（旧策略是按 kd=2 训的，用 0.667 还原；第四版是 kd=3，用默认 1.0）")
    return ap


if __name__ == "__main__":
    _args = build_parser().parse_args()
    if _args.duration is None:
        _args.duration = 6.0 if _args.no_viewer else 0.0
    raise SystemExit(V11Sim2Sim(_args).run())
