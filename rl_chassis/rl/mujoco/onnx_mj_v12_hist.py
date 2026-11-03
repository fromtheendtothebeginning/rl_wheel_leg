#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v12 sim2sim 启动入口：**mjlab 历史口径**（125 维单输入策略）。

mjlab 训练的 flat_hist 策略：输入 = 25 维本体观测的 5 帧历史，按
**term 分块、帧旧→新** 展开成 125 维：

    [ang_vel×5帧(15)] [gravity×5(15)] [cmd×5(15)] [hip_knee×5(20)] [dof_vel×5(30)] [action×5(30)]

物理/控制器完全复用 onnx_mj_v11.V11Sim2Sim（闭链映射、PD、滞后、延迟都一致），
只把策略推理的喂入从 25+125 双输入改成 125 单输入。

用法：
    python onnx_mj_v12_hist.py --onnx v12_mjlab_hist.onnx
    python onnx_mj_v12_hist.py --onnx v12_mjlab_hist.onnx --cmd-lin-vel 1.0 --duration 8 --no-viewer
"""

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
V12_SCENE = os.path.abspath(os.path.join(HERE, "..", "..", "mjcf", "rl_chassis_v12_scene.xml"))
sys.path.insert(0, HERE)
from onnx_mj_v11 import V11Sim2Sim, build_parser, CLIP_ACTIONS, POLICY_DT, SIM_DT  # noqa: E402

# 25 维观测的 term 边界（与 mjlab actor 组的 term 顺序一致）
TERM_DIMS = (3, 3, 3, 4, 6, 6)  # ang_vel, gravity, cmd, hip_knee, dof_vel, action
HIST = 5


class V12HistSim2Sim(V11Sim2Sim):
    """125 维 term-major 历史输入版。"""

    def __init__(self, args):
        # 父类 __init__ 会调 reset()，缓冲必须先建好
        self.term_bufs = [np.zeros((HIST, d), dtype=np.float32) for d in TERM_DIMS]
        super().__init__(args)

    def reset(self):
        super().reset()
        for buf in self.term_bufs:
            buf[:] = 0.0

    def push_obs(self, obs: np.ndarray) -> np.ndarray:
        """把 25 维观测按 term 推入各 5 帧环形缓冲，返回 125 维展开。"""
        i = 0
        for buf, d in zip(self.term_bufs, TERM_DIMS):
            buf[:-1] = buf[1:]      # 旧帧左移
            buf[-1] = obs[i:i + d]  # 最新帧放尾部
            i += d
        return np.concatenate([b.reshape(-1) for b in self.term_bufs])

    def run(self):
        args = self.args
        viewer = None
        if not args.no_viewer:
            import mujoco
            import mujoco.viewer
            viewer = mujoco.viewer.launch_passive(self.m, self.d)
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            viewer.cam.distance = 2.2
            viewer.cam.azimuth = 140.0
            viewer.cam.elevation = -18.0
            self.cmd.start_listener()
            print("[KEY] 按住 1 前进 / 2 后退 / 3 左转 / 4 右转，5 抬高 / 6 降低，0 停，"
                  "b 全部失能/恢复，f 姿态保持+轮差速/策略，q 或 Esc 退出", flush=True)
        t0 = time.time()
        step = 0
        vxs, heights = [], []
        next_t = time.perf_counter()
        last_sync = 0.0
        while (args.duration <= 0 or self.d.time < args.duration) and (viewer is None or viewer.is_running()):
            if viewer is not None:
                self.cmd.step_ramp(POLICY_DT)
            obs, q, qd, jac = self.read()
            flat = self.push_obs(obs)
            if self.sess is None or args.zero_action:
                act = np.zeros(6, dtype=np.float32)
            else:
                feed = {self.in_names[0]: flat.reshape(1, -1).astype(np.float32)}
                act = np.clip(self.sess.run(None, feed)[0].reshape(-1),
                              -CLIP_ACTIONS, CLIP_ACTIONS).astype(np.float32)
            self.last_actions = act
            self.action_fifo.append(act.astype(np.float32))
            if len(self.action_fifo) > self.n_delay:
                act_ctrl = self.action_fifo.pop(0)
            else:
                act_ctrl = np.zeros(6, dtype=np.float32)
            for _ in range(int(round(POLICY_DT / SIM_DT))):
                tau = self.compute_torques(act_ctrl, q, qd, jac)
                self.apply(tau)
                import mujoco
                mujoco.mj_step(self.m, self.d)
            if viewer is not None:
                now = time.perf_counter()
                if now - last_sync >= 1.0 / 60.0:
                    viewer.cam.lookat[:] = self.d.qpos[:3]
                    viewer.sync()
                    last_sync = now
                next_t += POLICY_DT
                wait = next_t - time.perf_counter()
                if wait > 0:
                    time.sleep(wait)
                else:
                    next_t = time.perf_counter()
            step += 1
            if step % max(1, int(1.0 / POLICY_DT)) == 0:
                self.print_state(q, vxs, heights)
        wall = time.time() - t0
        print(f"\n== 结束: 仿真 {self.d.time:.2f} s / 实际 {wall:.2f} s "
              f"(实时倍率 {self.d.time / max(wall, 1e-6):.2f}x)")
        if heights:
            yaw = float(self.d.qvel[5])
            print(f"   高度 均值 {np.mean(heights):.4f} m（目标 {args.cmd_height:.3f}）")
            print(f"   前进速度 均值 {np.mean(vxs):+.3f} m/s（指令 {args.cmd_lin_vel:+.2f}）")
            print(f"   末态偏航角速度 {yaw:+.3f} rad/s（指令 {args.cmd_yaw_vel:+.2f}）")
        return 0

    def print_state(self, q, vxs, heights):
        import math
        import mujoco
        L0L, th0L = self.couplers["L"].leg.virtual_leg(q[0], q[1])
        L0R, th0R = self.couplers["R"].leg.virtual_leg(q[3], q[4])
        bh = float(self.d.qpos[2])
        vx = float(self.d.qvel[0])
        vxs.append(vx)
        heights.append(bh)
        w, x, y, z = self.d.qpos[3:7]
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, 1 - 2 * (x * x + y * y)))))
        yaw = math.degrees(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))
        print(f"[{self.d.time:5.2f}s] x={self.d.qpos[0]:+.2f} y={self.d.qpos[1]:+.2f} yaw={yaw:+.1f}deg h={bh:.4f} m  vx={vx:+.3f} m/s  "
              f"L0=({L0L:.4f},{L0R:.4f})  theta0=({math.degrees(th0L):+.1f},{math.degrees(th0R):+.1f})deg  "
              f"cmd=({self.cmd.lin:+.2f},{self.cmd.yaw:+.2f},{self.cmd.height:.3f})  "
              f"mode={self.cmd.mode}  倾斜={tilt:.1f}deg  接地={self.floor_contacts()}", flush=True)


def main():
    parser = build_parser()
    parser.set_defaults(xml=V12_SCENE)
    args = parser.parse_args()
    if args.duration is None:
        args.duration = 6.0 if args.no_viewer else 0.0
    return V12HistSim2Sim(args).run()


if __name__ == "__main__":
    raise SystemExit(main())
