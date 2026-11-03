#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""mjlab 跳跃策略 sim2sim：冻结平地策略 + 42 维残差 + 相位状态机 + 气弹簧。

与 rl/mjlab 里 ``WheelLegged-Jump-v12-Closed`` 任务同口径：
* 平地策略：``v12_bc2.onnx``（125 维 term-major 单输入，wxk 策略的 DAgger 蒸馏）；
* 残差：``--jump-onnx``（42→6，export_onnx.py --num-obs 42 导出的训练 checkpoint）；
* 状态机：``rl/tools/jump_manager.py``（训练/部署共用的相位逻辑）；
* 气弹簧：v12 gas MJCF 的 tendon 电机，FLAT 20 N / 跳跃窗口 350 N；
* 动作 = clip(a_flat + scale(phase)·a_residual, ±100) → PD/滞后/限幅。

用法：
    python onnx_mj_v12_jump_mjlab.py --jump-onnx ../mjlab/logs/.../model_450.onnx
    python onnx_mj_v12_jump_mjlab.py --jump-onnx ... --auto-jump --no-viewer --duration 6
"""

from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
V12_GAS_SCENE = os.path.abspath(
    os.path.join(HERE, "..", "..", "mjcf", "rl_chassis_v12_gas_scene.xml"))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "tools")))

from onnx_mj_v11 import (  # noqa: E402
    CLIP_ACTIONS, POLICY_DT, SIM_DT, V11Sim2Sim, build_parser, quat_rotate_inverse)
from jump_manager import JumpManager, JumpPhase  # noqa: E402

FLAT_ONNX = os.path.join(HERE, "v12_bc2.onnx")
DIMS = (3, 3, 3, 4, 6, 6)  # 平地策略的 term-major 历史分块


class MjlabJumpSim(V11Sim2Sim):
    def __init__(self, args):
        super().__init__(args)
        import torch
        self.torch = torch
        self.device_t = torch.device("cpu")
        # 残差 ONNX
        if args.jump_onnx:
            import onnxruntime as ort
            self.jsess = ort.InferenceSession(args.jump_onnx, providers=["CPUExecutionProvider"])
            self.jinput = self.jsess.get_inputs()[0].name
        else:
            self.jsess = None
        self.manager = JumpManager(1, self.device_t, target_height=args.target_height)
        # 部署：默认不自动跳；--auto-jump 时 1.5 s 后自动触发一次（循环由相位机驱动）
        self.manager.reset_all(warmup=(1.5, 1.5) if getattr(args, "auto_jump", False) else (1e9, 1e9))
        self.term_bufs = [np.zeros((5, d), dtype=np.float32) for d in DIMS]
        self.last_residual = np.zeros(6, dtype=np.float32)
        self.last_total = np.zeros(6, dtype=np.float32)
        self.residual_delay = [np.zeros(6, dtype=np.float32) for _ in range(int(args.residual_delay))]
        # 气弹簧电机 ctrl 索引
        import mujoco
        self.gas_ids = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
                        for n in ("Left_gas_motor", "Right_gas_motor")]
        self.peak_z = 0.0
        self.start_z = None

    def reset(self):
        super().reset()
        if not hasattr(self, "term_bufs"):  # 父类构造期会先调 reset()
            return
        for b in self.term_bufs:
            b[:] = 0.0
        self.last_residual[:] = 0.0
        self.last_total[:] = 0.0
        for d in self.residual_delay:
            d[:] = 0.0
        self.manager.reset_all(warmup=(0.5, 1.5))
        self.peak_z = 0.0
        self.start_z = None

    # ---------------- 42 维观测 ----------------
    def _contact(self):
        import mujoco
        found = False
        for c in self.d.contact:
            n1 = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or ""
            n2 = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or ""
            if ("Wheel_link_col" in n1 and "floor" in n2) or ("Wheel_link_col" in n2 and "floor" in n1):
                found = True
                break
        return found

    def _obs42(self, obs25, contact_conf, torque_fb):
        t = self.torch
        onehot = self.manager.phase_onehot()[0]
        prog = self.manager.phase_progress()[0].unsqueeze(0)
        last = t.from_numpy(self.last_residual)
        return np.concatenate([
            obs25.astype(np.float32),
            onehot.numpy().astype(np.float32),
            prog.numpy().astype(np.float32),
            contact_conf.astype(np.float32),
            torque_fb.astype(np.float32),
            last.numpy().astype(np.float32),
        ]).reshape(1, -1)

    def run(self):
        args = self.args
        import mujoco
        import mujoco.viewer
        viewer = None
        if not args.no_viewer:
            viewer = mujoco.viewer.launch_passive(self.m, self.d)
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            viewer.cam.distance = 2.2
            viewer.cam.azimuth = 140.0
            viewer.cam.elevation = -18.0
            self.cmd.start_listener()
            print("[KEY] 空格起跳；1/2/3/4 移动；5/6 调高；0 停；q 退出", flush=True)
        t0 = 0.0
        step = 0
        next_t = __import__("time").perf_counter()
        last_sync = 0.0
        while not args.no_viewer and viewer.is_running() or args.no_viewer and (
                args.duration <= 0 or self.d.time < args.duration):
            self.cmd.step_ramp(POLICY_DT)
            obs25, q, qd, jac = self.read()
            # 平地策略：term-major 125
            i = 0
            for buf, d in zip(self.term_bufs, DIMS):
                buf[:-1] = buf[1:]
                buf[-1] = obs25[i:i + d]
                i += d
            flat_in = np.concatenate([b.reshape(-1) for b in self.term_bufs]).reshape(1, -1)
            a_flat = np.clip(self.sess.run(None, {self.in_names[0]: flat_in.astype(np.float32)})[0]
                             .reshape(-1), -CLIP_ACTIONS, CLIP_ACTIONS)

            # 状态机推进（可部署信号）
            contact = self._contact()
            root_z = float(self.d.qpos[2])
            base_vz = float(self.d.qvel[2])
            grav_z = float(quat_rotate_inverse(self.d.qpos[3:7], np.array([0, 0, -1.0]))[2])
            ang_xy = float(np.linalg.norm(self.d.qvel[3:5]))
            self.manager.advance(
                POLICY_DT, self.torch.tensor([contact]),
                self.torch.tensor([root_z]), self.torch.tensor([base_vz]),
                self.torch.tensor([grav_z]), self.torch.tensor([ang_xy]))
            # 空跳触发：空格由 cmd.mode 反射为 jump_request（见下）
            if getattr(self.cmd, "jump_request", False):
                self.manager.warmup[0] = 0.0
                self.cmd.jump_request = False

            # 接触置信度 / 力矩反馈（可部署近似）
            qd_arr = np.array(qd)
            tau_w = np.array(self.tau_applied[[2, 5]]) if hasattr(self, "tau_applied") else np.zeros(2)
            acc_w = np.abs(qd_arr[[2, 5]]) / POLICY_DT
            conf = np.clip(0.5 * np.clip(acc_w / 50.0, 0, 1) + 0.3 * np.clip(ang_xy * 0.4, 0, 1)
                           + 0.2 * np.clip(np.abs(tau_w) / 20.0, 0, 1), 0, 1)
            # 残差推理
            if self.jsess is not None:
                o42 = self._obs42(obs25, conf, tau_w / 20.0)
                res = np.clip(self.jsess.run(None, {self.jinput: o42.astype(np.float32)})[0]
                              .reshape(-1), -CLIP_ACTIONS, CLIP_ACTIONS)
            else:
                res = np.zeros(6, dtype=np.float32)
            # 延迟 + 相位缩放
            if self.residual_delay:
                self.residual_delay.append(res)
                res_d = self.residual_delay.pop(0)
            else:
                res_d = res
            scale = float(self.manager.residual_scale()[0])
            self.last_residual = res
            act = np.clip(a_flat + scale * res_d, -CLIP_ACTIONS, CLIP_ACTIONS).astype(np.float32)
            self.last_total = act

            # 气弹簧 ctrl（FLAT 20N / 跳跃窗口 350N）
            gas = float(args.gas_force_flat) if int(self.manager.phase[0]) == int(JumpPhase.FLAT) \
                else float(args.gas_force_jump)
            for gid in self.gas_ids:
                self.d.ctrl[gid] = gas

            self.last_actions = act
            self.action_fifo.append(act.astype(np.float32))
            act_ctrl = self.action_fifo.pop(0) if len(self.action_fifo) > self.n_delay else np.zeros(6, np.float32)
            for _ in range(int(round(POLICY_DT / SIM_DT))):
                tau = self.compute_torques(act_ctrl, q, qd, jac)
                self.apply(tau)
                mujoco.mj_step(self.m, self.d)

            # 记录/打印
            if self.start_z is None and self.manager.phase[0] != int(JumpPhase.FLAT):
                self.start_z = root_z
            if not contact:  # 与训练一致：真实离地 = 轮地无接触
                self.peak_z = max(self.peak_z, root_z - float(self.start_z or root_z))
            if viewer is not None:
                now = __import__("time").perf_counter()
                if now - last_sync >= 1.0 / 60.0:
                    viewer.cam.lookat[:] = self.d.qpos[:3]
                    viewer.sync()
                    last_sync = now
                next_t += POLICY_DT
                wait = next_t - __import__("time").perf_counter()
                if wait > 0:
                    __import__("time").sleep(wait)
                else:
                    next_t = __import__("time").perf_counter()
            step += 1
            if step % max(1, int(1.0 / POLICY_DT)) == 0:
                names = {0: "FLAT", 1: "CROUCH", 2: "THRUST", 3: "FLIGHT", 4: "EXTEND", 5: "LAND", 6: "RECOVER"}
                print(f"[{self.d.time:5.2f}s] {names[int(self.manager.phase[0])]:7s} "
                      f"z={root_z:.3f} vz={base_vz:+.2f} gas={gas:3.0f}N "
                      f"peak={self.peak_z:.3f} contact={contact}", flush=True)
        return 0


def main():
    parser = build_parser()
    parser.set_defaults(xml=V12_GAS_SCENE)
    parser.add_argument("--jump-onnx", default=None, help="42->6 残差 ONNX（不传=纯平地策略+状态机）")
    parser.add_argument("--target-height", type=float, default=0.10)
    parser.add_argument("--residual-delay", type=int, default=1)
    parser.add_argument("--auto-jump", action="store_true")
    parser.add_argument("--gas-force-flat", type=float, default=20.0)
    parser.add_argument("--gas-force-jump", type=float, default=350.0)
    args = parser.parse_args()
    if args.onnx is None:
        args.onnx = FLAT_ONNX
    if args.duration is None:
        args.duration = 6.0 if args.no_viewer else 0.0
    # 空格触发：给 CommandState 打个 jump_request 标记
    sim = MjlabJumpSim(args)
    return sim.run()


if __name__ == "__main__":
    raise SystemExit(main())
