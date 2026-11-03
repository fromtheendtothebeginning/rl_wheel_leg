"""Wheel-legged flat-ground environment (DirectRLEnv workflow).

移植自 ``plane_v11/wheel_legged_gym``：25 维本体观测、6 维动作、逐物理步 PD、
10 ms 动作延迟 + 15 ms 一阶执行器滞后，奖励项按 v4 配置的量级。

与 Isaac Gym 版本的对应关系：

* 控制周期 10 ms（sim dt 0.005 s × decimation 2）；
* 髋/膝位置 PD（kp=40, kd=3），轮速度 PD（kp=0, kd=0.3）；
* 观测里只有机体系角速度、投影重力、指令、髋/膝角偏差、6 个关节速度、上一步动作。
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.envs import DirectRLEnv
from isaaclab.utils import index_fill_
from isaaclab.utils.math import sample_uniform

from .wheel_env_cfg import DEFAULT_JOINT_POS, JOINT_ORDER, WheelLeggedFlatEnvCfg

# 虚拟腿几何（rl/tools/calibrate_leg.py 标定值，与 plane_v11 配置一致）。
LEG_L1 = 0.215000066
LEG_L2 = 0.253999977
LEG_A1 = 3.139319059
LEG_A2 = -3.701985976
LEG_SIGN = (-1.0, 1.0)  # 左、右

HIP_KNEE = [0, 1, 3, 4]
WHEELS = [2, 5]


class WheelLeggedEnv(DirectRLEnv):
    cfg: WheelLeggedFlatEnvCfg

    def __init__(self, cfg: WheelLeggedFlatEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._robot = self.scene["robot"]
        self._terrain = self.scene["terrain"]
        self._contact_sensor = self.scene["contact_sensor"]
        # 轮子以外任何 part 撞地都判失败：训练 URDF 的凸包碰撞体会让底盘"坐"在地上，
        # 不禁止的话策略会学成躺着/跪着，换到闭链模型（真实碰撞）立刻塌。
        self._bad_contact_ids, self._bad_contact_names = self._contact_sensor.find_sensors(
            ".*(base_link|rear_link|child1_link)"
        )

        # 动作/观测按 JOINT_ORDER 排列；articulation 内部顺序可能不同，按名字建立映射。
        articulation_names = list(self._robot.joint_names)
        missing = [name for name in JOINT_ORDER if name not in articulation_names]
        if missing:
            raise RuntimeError(f"训练 USD 缺关节 {missing}，实际关节：{articulation_names}")
        self._joint_ids = torch.tensor(
            [articulation_names.index(name) for name in JOINT_ORDER], dtype=torch.long, device=self.device
        )
        self._default_q = torch.tensor(
            [DEFAULT_JOINT_POS[name] for name in JOINT_ORDER], dtype=torch.float, device=self.device
        ).repeat(self.num_envs, 1)
        self._orthogonal_default = torch.tensor(
            [DEFAULT_JOINT_POS[name] for name in articulation_names], dtype=torch.float, device=self.device
        ).repeat(self.num_envs, 1)

        n = self.num_envs
        self._actions = torch.zeros(n, 6, device=self.device)
        self._applied_actions = torch.zeros(n, 6, device=self.device)
        self._prev2_actions = torch.zeros(n, 6, device=self.device)
        self._obs_actions = torch.zeros(n, 6, device=self.device)
        self._tau_applied = torch.zeros(n, 6, device=self.device)

        self._commands = torch.zeros(n, 3, device=self.device)
        self._cmd_timer = torch.zeros(n, device=self.device)

        self._fail_time = torch.zeros(n, device=self.device)

        self._cmd_scale = torch.tensor([2.0, 0.25, 5.0], device=self.device)
        self._noise_scale = torch.tensor(
            [self.cfg.obs_noise_ang_vel] * 3
            + [self.cfg.obs_noise_gravity] * 3
            + [0.0] * 3
            + [self.cfg.obs_noise_dof_pos] * 4
            + [self.cfg.obs_noise_dof_vel] * 6
            + [0.0] * 6,
            device=self.device,
        )

        # 训练时的控制器随机化：kp/kd、力矩上限按 env 采样一次（与 gym 版 startup 随机化一致）。
        self._kp = self.cfg.kp * sample_uniform(0.8, 1.2, (n, 1), self.device)
        self._kd = self.cfg.kd * sample_uniform(0.8, 1.2, (n, 1), self.device)
        self._wheel_kd = self.cfg.wheel_kd * sample_uniform(0.8, 1.2, (n, 1), self.device)
        self._effort_limit = torch.tensor(self.cfg.effort_limit, device=self.device).repeat(n, 1)
        self._effort_limit *= sample_uniform(0.8, 1.1, (n, 6), self.device)

        self._leg_sign = torch.tensor(LEG_SIGN, device=self.device)

        # 诊断指标（reset 时写入 extras["log"]）
        self._height_err_sum = torch.zeros(n, device=self.device)
        self._speed_abs_sum = torch.zeros(n, device=self.device)
        self._diag_steps = torch.zeros(n, device=self.device)

    # ------------------------------------------------------------------ helpers
    def _read_joints(self) -> tuple[torch.Tensor, torch.Tensor]:
        data = self._robot.data
        return data.joint_pos.torch[:, self._joint_ids], data.joint_vel.torch[:, self._joint_ids]

    def _virtual_leg_theta0(self, q: torch.Tensor) -> torch.Tensor:
        theta1 = LEG_A1 + self._leg_sign * q[:, [0, 3]]
        theta2 = LEG_A2 + self._leg_sign * q[:, [1, 4]]
        end_x = LEG_L1 * torch.cos(theta1) + LEG_L2 * torch.cos(theta1 + theta2)
        end_z = LEG_L1 * torch.sin(theta1) + LEG_L2 * torch.sin(theta1 + theta2)
        return torch.atan2(end_x, -end_z)

    def _resample_commands(self, env_ids: torch.Tensor, *, random_timer: bool = True) -> None:
        count = len(env_ids)
        if count == 0:
            return
        stand = torch.rand(count, device=self.device) < self.cfg.standing_command_prob
        vx = sample_uniform(*self.cfg.cmd_lin_vel_range, (count,), self.device)
        wz = sample_uniform(*self.cfg.cmd_yaw_vel_range, (count,), self.device)
        height = sample_uniform(*self.cfg.cmd_height_range, (count,), self.device)
        vx = torch.where(stand, torch.zeros_like(vx), vx)
        wz = torch.where(stand, torch.zeros_like(wz), wz)
        self._commands[env_ids] = torch.stack((vx, wz, height), dim=-1)
        if random_timer:
            self._cmd_timer[env_ids] = sample_uniform(
                0.0, self.cfg.command_resample_time_s, (count,), self.device
            )
        else:
            self._cmd_timer[env_ids] = self.cfg.command_resample_time_s

    # --------------------------------------------------------------- env hooks
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # 动作延迟 1 个控制周期：本步用上一步的策略输出。
        self._prev2_actions[:] = self._applied_actions
        self._applied_actions[:] = self._actions
        self._obs_actions[:] = self._actions
        self._actions = torch.clamp(actions.to(self.device), -100.0, 100.0)

    def _apply_action(self) -> None:
        q, qd = self._read_joints()
        a = self._applied_actions

        target = self._default_q + self.cfg.pos_action_scale * a
        tau = torch.zeros_like(q)
        tau[:, HIP_KNEE] = self._kp * (target[:, HIP_KNEE] - q[:, HIP_KNEE]) - self._kd * qd[:, HIP_KNEE]
        tau[:, WHEELS] = self._wheel_kd * (self.cfg.vel_action_scale * a[:, WHEELS] - qd[:, WHEELS])

        limit = self._effort_limit
        tau = torch.max(torch.min(tau, limit), -limit)

        # 一阶执行器滞后：tau <- tau + (tau_cmd - tau) * dt / actuator_tau
        alpha = min(1.0, self.physics_dt / max(self.cfg.actuator_tau, 1e-6))
        self._tau_applied += (tau - self._tau_applied) * alpha
        self._tau_applied = torch.max(torch.min(self._tau_applied, limit), -limit)
        self._robot.set_joint_effort_target_index(target=self._tau_applied, joint_ids=self._joint_ids)

    def _get_observations(self) -> dict:
        # 指令按 5 s 重采样
        self._cmd_timer -= self.step_dt
        expired = (self._cmd_timer <= 0.0).nonzero(as_tuple=False).squeeze(-1)
        self._resample_commands(expired, random_timer=True)

        q, qd = self._read_joints()
        data = self._robot.data
        obs = torch.cat(
            (
                data.root_ang_vel_b.torch * 0.25,
                -data.projected_gravity_b.torch,  # Isaac 3.0 的 projected_gravity_b 直立为 +1，转成策略口径的 -1
                self._commands * self._cmd_scale,
                q[:, HIP_KNEE] - self._default_q[:, HIP_KNEE],
                qd * 0.05,
                self._obs_actions,
            ),
            dim=-1,
        )
        if self.cfg.obs_noise_dof_pos > 0.0:
            obs = obs + (2.0 * torch.rand_like(obs) - 1.0) * self._noise_scale
        obs = torch.clamp(obs, -100.0, 100.0)

        # 诊断：高度误差与零指令下的残余速度
        height_err = (data.root_pos_w.torch[:, 2] - self._commands[:, 2]).abs()
        self._height_err_sum += height_err
        self._speed_abs_sum += data.root_lin_vel_b.torch[:, 0].abs()
        self._diag_steps += 1.0
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        q, qd = self._read_joints()
        data = self._robot.data
        lin = data.root_lin_vel_b.torch
        ang = data.root_ang_vel_b.torch
        grav = -data.projected_gravity_b.torch  # 策略口径：直立时 g_z = -1
        qdd = data.joint_acc.torch[:, self._joint_ids]

        lin_err = torch.square(self._commands[:, 0] - lin[:, 0])
        yaw_err = torch.square(self._commands[:, 1] - ang[:, 2])
        height_err = torch.square(data.root_pos_w.torch[:, 2] - self._commands[:, 2])
        theta0 = self._virtual_leg_theta0(q)
        contact_force = self._contact_sensor.data.net_normal_forces_w_history.torch
        bad_contact_reward = torch.linalg.norm(
            contact_force[:, :, self._bad_contact_ids], dim=-1
        ).amax(dim=(1, 2))

        rew = (
            1.0 * torch.exp(-lin_err / 0.25)
            + 1.0 * (torch.exp(-lin_err / 2.5) - 1.0)
            + 1.0 * torch.exp(-yaw_err / 0.25)
            + 1.0 * (torch.exp(-yaw_err / 2.5) - 1.0)
            + torch.exp(-height_err / 0.001)
            - 2.0 * torch.square(theta0[:, 0] - theta0[:, 1])
            - 2.0 * torch.square(lin[:, 2])
            - 0.6 * torch.sum(torch.square(ang[:, :2]), dim=1)
            - 100.0 * torch.sum(torch.square(grav[:, :2]), dim=1)
            - 4.0e-4 * torch.sum(torch.square(qd[:, HIP_KNEE]), dim=1)
            - 3.0e-6 * torch.sum(torch.square(qdd), dim=1)
            - 8.0e-4 * torch.sum(torch.square(self._tau_applied), dim=1)
            - 1.0 * (bad_contact_reward > 5.0).float()
            - 0.12 * torch.sum(torch.square(self._actions - self._applied_actions), dim=1)
            - 0.12
            * torch.sum(
                torch.square(
                    self._actions[:, HIP_KNEE]
                    - 2.0 * self._applied_actions[:, HIP_KNEE]
                    + self._prev2_actions[:, HIP_KNEE]
                ),
                dim=1,
            )
        )
        return rew

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        grav_z = -self._robot.data.projected_gravity_b.torch[:, 2]
        failed = grav_z > self.cfg.fail_tilt_cos
        self._fail_time = torch.where(failed, self._fail_time + self.step_dt, torch.zeros_like(self._fail_time))
        contact_force = self._contact_sensor.data.net_normal_forces_w_history.torch
        bad_contact = torch.linalg.norm(contact_force[:, :, self._bad_contact_ids], dim=-1).amax(dim=(1, 2))
        terminated = (self._fail_time > self.cfg.fail_to_terminal_time_s) | (bad_contact > 10.0)
        time_out = self.episode_length_buf >= self.max_episode_length
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = torch.arange(self.num_envs, dtype=torch.int32, device=self.device)
        else:
            env_ids = torch.as_tensor(env_ids, dtype=torch.int32, device=self.device)

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        if len(env_ids) == self.num_envs:
            # 打散初始相位，避免所有 env 同时超时。
            self.episode_length_buf[:] = torch.randint_like(
                self.episode_length_buf, high=int(self.max_episode_length)
            )

        n = len(env_ids)
        # 记录即将结束的这一段 episode 的诊断量
        if bool(self._diag_steps[env_ids].gt(0).any()):
            steps = self._diag_steps[env_ids].clamp_min(1.0)
            self.extras.setdefault("log", {})
            self.extras["log"]["Metrics/height_err"] = float((self._height_err_sum[env_ids] / steps).mean())
            self.extras["log"]["Metrics/speed_abs"] = float((self._speed_abs_sum[env_ids] / steps).mean())

        root_pose = self._robot.data.default_root_pose.torch[env_ids].clone()
        root_pose[:, :3] += self._terrain.env_origins[env_ids]
        yaw = sample_uniform(-0.3, 0.3, (n,), self.device)
        root_pose[:, 3] = torch.cos(yaw * 0.5)
        root_pose[:, 4] = 0.0
        root_pose[:, 5] = 0.0
        root_pose[:, 6] = torch.sin(yaw * 0.5)

        root_vel = torch.zeros(n, 6, device=self.device)
        root_vel[:, :2] = sample_uniform(-0.2, 0.2, (n, 2), self.device)
        root_vel[:, 5] = sample_uniform(-0.2, 0.2, (n,), self.device)

        joint_pos = self._orthogonal_default[env_ids].clone()
        joint_pos[:, :] += sample_uniform(-0.02, 0.02, joint_pos.shape, self.device)
        joint_vel = torch.zeros_like(joint_pos)

        self._robot.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
        self._robot.write_root_velocity_to_sim_index(root_velocity=root_vel, env_ids=env_ids)
        self._robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
        self._robot.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=env_ids)

        index_fill_(self._actions, env_ids, 0.0)
        index_fill_(self._applied_actions, env_ids, 0.0)
        index_fill_(self._prev2_actions, env_ids, 0.0)
        index_fill_(self._obs_actions, env_ids, 0.0)
        index_fill_(self._tau_applied, env_ids, 0.0)
        index_fill_(self._fail_time, env_ids, 0.0)
        index_fill_(self._height_err_sum, env_ids, 0.0)
        index_fill_(self._speed_abs_sum, env_ids, 0.0)
        index_fill_(self._diag_steps, env_ids, 0.0)

        self._resample_commands(env_ids, random_timer=True)
