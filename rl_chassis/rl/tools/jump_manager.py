#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""跳跃相位状态机（训练和部署共用）。

输入是可部署信号：轮地接触估计、IMU 角速度/投影重力、关节编码器和轮电流/力矩。
高度只用于训练奖励和成功判定，不进入 actor 观测。
"""

from __future__ import annotations

import math
from enum import IntEnum

import torch


class JumpPhase(IntEnum):
    FLAT = 0
    CROUCH = 1
    THRUST = 2
    FLIGHT = 3
    EXTEND = 4
    LAND = 5
    RECOVER = 6


PHASE_NAMES = {
    JumpPhase.FLAT: "FLAT",
    JumpPhase.CROUCH: "CROUCH",
    JumpPhase.THRUST: "THRUST",
    JumpPhase.FLIGHT: "FLIGHT",
    JumpPhase.EXTEND: "EXTEND",
    JumpPhase.LAND: "LAND",
    JumpPhase.RECOVER: "RECOVER",
}

PHASE_DUR = {
    JumpPhase.CROUCH: 0.12,
    JumpPhase.THRUST: 0.15,
    JumpPhase.EXTEND: 0.10,
    JumpPhase.LAND: 0.25,
    JumpPhase.RECOVER: 0.25,
}

# 跳跃时平地策略的高度指令（米）
HEIGHT_CROUCH = 0.14
HEIGHT_THRUST = 0.24
HEIGHT_FLIGHT = 0.14
HEIGHT_EXTEND = 0.24
HEIGHT_LAND = 0.15


class JumpManager:
    """向量化相位管理器，所有状态都在 GPU/CPU tensor 上。"""

    def __init__(self, num_envs: int, device, target_height: float = 0.10):
        self.n = int(num_envs)
        self.device = device
        self.target_height = float(target_height)
        self.flight_dur = 2.0 * math.sqrt(
            max(2.0 * self.target_height / 9.81, 1e-6)
        ) + 0.04

        self.phase = torch.full(
            (self.n,), int(JumpPhase.FLAT), dtype=torch.long, device=device
        )
        self.timer = torch.zeros(self.n, device=device)
        self.warmup = torch.zeros(self.n, device=device)
        self.start_z = torch.zeros(self.n, device=device)
        self.peak_z = torch.zeros(self.n, device=device)
        self.landing_impact = torch.zeros(self.n, device=device)
        self.landed = torch.zeros(self.n, dtype=torch.bool, device=device)
        self.recovered = torch.zeros(self.n, dtype=torch.bool, device=device)
        self.success = torch.zeros(self.n, dtype=torch.bool, device=device)

    # ------------------------------------------------------------------
    def reset_all(self, warmup=(0.5, 1.5)):
        self._reset_ids(torch.arange(self.n, device=self.device), warmup)

    def reset_envs(self, env_ids, warmup=(0.5, 1.5)):
        self._reset_ids(env_ids, warmup)

    def _reset_ids(self, env_ids, warmup):
        if len(env_ids) == 0:
            return
        lo, hi = float(warmup[0]), float(warmup[1])
        self.phase[env_ids] = int(JumpPhase.FLAT)
        self.timer[env_ids] = 0.0
        self.warmup[env_ids] = torch.rand(len(env_ids), device=self.device) * (hi - lo) + lo
        self.start_z[env_ids] = 0.0
        self.peak_z[env_ids] = 0.0
        self.landing_impact[env_ids] = 0.0
        self.landed[env_ids] = False
        self.recovered[env_ids] = False
        self.success[env_ids] = False

    # ------------------------------------------------------------------
    def advance(self, dt, contact, root_z=None, base_vz=None,
                gravity_z=None, ang_xy=None):
        """推进一个策略周期。contact 是 deployable detector 的布尔结果。"""
        dt = float(dt)
        self.timer += dt
        self.landed.zero_()
        self.recovered.zero_()
        self.success.zero_()

        flat = self.phase == int(JumpPhase.FLAT)
        self.warmup[flat] = torch.clamp(self.warmup[flat] - dt, min=0.0)

        if root_z is not None:
            ready = flat & (self.warmup <= 0.0) & contact
            self.start_z[ready] = root_z[ready]
            self.peak_z[ready] = 0.0
            self.landing_impact[ready] = 0.0
            self.phase[ready] = int(JumpPhase.CROUCH)
            self.timer[ready] = 0.0
            flat = self.phase == int(JumpPhase.FLAT)
            self.start_z = torch.where(flat, root_z, self.start_z)
            active = self.phase != int(JumpPhase.FLAT)
            self.peak_z = torch.maximum(
                self.peak_z,
                torch.where(active, root_z - self.start_z, self.peak_z),
            )

        # CROUCH -> THRUST
        mask = (self.phase == int(JumpPhase.CROUCH)) & (
            self.timer >= PHASE_DUR[JumpPhase.CROUCH]
        )
        self._to(mask, JumpPhase.THRUST)

        # THRUST -> FLIGHT：离地或超时
        mask = (self.phase == int(JumpPhase.THRUST)) & (
            (~contact) | (self.timer >= 0.20)
        )
        self._to(mask, JumpPhase.FLIGHT)

        # FLIGHT -> EXTEND
        mask = (self.phase == int(JumpPhase.FLIGHT)) & (
            self.timer >= max(self.flight_dur - 0.10, 0.15)
        )
        self._to(mask, JumpPhase.EXTEND)

        # EXTEND -> LAND
        mask = (self.phase == int(JumpPhase.EXTEND)) & (
            contact | (self.timer >= 0.12)
        )
        if base_vz is not None:
            self.landing_impact[mask] = torch.clamp(-base_vz[mask], min=0.0)
        self.landed[mask] = True
        self._to(mask, JumpPhase.LAND)

        # LAND -> RECOVER
        mask = (self.phase == int(JumpPhase.LAND)) & (
            self.timer >= PHASE_DUR[JumpPhase.LAND]
        )
        self._to(mask, JumpPhase.RECOVER)

        # RECOVER -> FLAT
        stable = torch.ones_like(contact, dtype=torch.bool)
        if gravity_z is not None:
            stable &= gravity_z < -0.98
        if ang_xy is not None:
            stable &= ang_xy < 0.5
        mask = (
            (self.phase == int(JumpPhase.RECOVER))
            & (self.timer >= PHASE_DUR[JumpPhase.RECOVER])
            & stable
        )
        self.recovered[mask] = True
        self.success[mask] = self.peak_z[mask] >= 0.8 * self.target_height
        self._to(mask, JumpPhase.FLAT, reset_warmup=True)

    def _to(self, mask, phase, reset_warmup=False):
        if not bool(mask.any()):
            return
        self.phase[mask] = int(phase)
        self.timer[mask] = 0.0
        if reset_warmup:
            self.warmup[mask] = torch.rand(
                int(mask.sum()), device=self.device
            ) * 1.0 + 0.5

    # ------------------------------------------------------------------
    def phase_onehot(self):
        out = torch.zeros(self.n, 6, device=self.device)
        active = self.phase != int(JumpPhase.FLAT)
        idx = torch.clamp(self.phase - 1, min=0, max=5)
        out[active, idx[active]] = 1.0
        return out

    def phase_progress(self):
        dur = torch.full((self.n,), 0.25, device=self.device)
        for phase, value in PHASE_DUR.items():
            dur[self.phase == int(phase)] = value
        dur[self.phase == int(JumpPhase.FLIGHT)] = self.flight_dur
        return torch.clamp(self.timer / dur, 0.0, 1.0)

    def height_command(self, flat_height):
        out = flat_height.clone()
        out[self.phase == int(JumpPhase.CROUCH)] = HEIGHT_CROUCH
        out[self.phase == int(JumpPhase.THRUST)] = HEIGHT_THRUST
        out[self.phase == int(JumpPhase.FLIGHT)] = HEIGHT_FLIGHT
        out[self.phase == int(JumpPhase.EXTEND)] = HEIGHT_EXTEND
        out[self.phase == int(JumpPhase.LAND)] = HEIGHT_LAND
        out[self.phase == int(JumpPhase.RECOVER)] = HEIGHT_LAND
        return out

    def residual_scale(self):
        scale = torch.ones(self.n, device=self.device)
        scale[self.phase == int(JumpPhase.FLAT)] = 0.0
        crouch = self.phase == int(JumpPhase.CROUCH)
        scale[crouch] = torch.clamp(self.timer[crouch] / 0.05, 0.0, 1.0)
        rec = self.phase == int(JumpPhase.RECOVER)
        scale[rec] = torch.clamp(
            1.0 - self.timer[rec] / PHASE_DUR[JumpPhase.RECOVER], 0.0, 1.0
        )
        return scale

    def contact_features(self, env):
        """只用可部署信号估计左右轮接触置信度。"""
        wheel_acc = torch.abs(env.dof_acc[:, [2, 5]])
        imu = torch.norm(env.base_ang_vel[:, :2], dim=1, keepdim=True).expand(-1, 2)
        current = torch.abs(env.torques[:, [2, 5]]) / 20.0
        conf = 0.5 * torch.clamp(wheel_acc / 50.0, 0.0, 1.0)
        conf += 0.3 * torch.clamp(imu * 2.0, 0.0, 1.0)
        conf += 0.2 * torch.clamp(current, 0.0, 1.0)
        return torch.clamp(conf, 0.0, 1.0)

    def phase_histogram(self):
        return {
            name: int((self.phase == int(phase)).sum().item())
            for phase, name in PHASE_NAMES.items()
        }
