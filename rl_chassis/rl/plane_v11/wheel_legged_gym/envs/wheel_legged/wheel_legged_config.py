# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 ETH Zurich, Nikita Rudin
# 针对 rl_chassis_v11 改写：机器人换成我们闭链轮腿的「单腿串联训练模型」，
# 虚拟腿参数由 rl/tools/calibrate_leg.py 从 v11 MJCF 标定得到。

import os

from wheel_legged_gym import WHEEL_LEGGED_GYM_ROOT_DIR
from wheel_legged_gym.envs.base.legged_robot_config import (
    LeggedRobotCfg,
    LeggedRobotCfgPPO,
)

# plane_v11/wheel_legged_gym/... -> rl_chassis/urdf/rl_chassis_train.urdf
_RL_DIR = os.path.normpath(os.path.join(WHEEL_LEGGED_GYM_ROOT_DIR, ".."))
_ROBOT_DIR = os.path.normpath(os.path.join(_RL_DIR, ".."))
V11_TRAIN_URDF = os.path.join(_ROBOT_DIR, "urdf", "rl_chassis_train.urdf")


class WheelLeggedCfg(LeggedRobotCfg):

    class init_state(LeggedRobotCfg.init_state):
        # 目标站姿 L0 = 0.205 m、theta0 = 0（腿竖直向下），base_link 高度 = 0.1499 m。
        # 默认角由 rl/tools/tune_default_pose.py 算成「重力静平衡点」：
        #   q_default = q_des + τ_gravity/kp（v11 的腿比较折，kp=40 时膝上要 ~7.5 N·m，
        #   不补偿的话零动作会静差下沉 0.19 rad、一路折到轮舱里）
        # 第四版（平地）：出生高度对齐高度指令范围 [0.14, 0.24] 的中位
        pos = [0.0, 0.0, 0.18]  # x,y,z [m]
        default_joint_angles = {
            "Left_rear_joint": -0.274884,        # 髋（大腿）
            "Left_rear_child1_joint": 0.517721,  # 虚拟膝（真机由前髋曲柄驱动）
            "Left_Wheel_joint": 0.0,
            "Right_rear_joint": 0.274883,
            "Right_rear_child1_joint": -0.517722,
            "Right_Wheel_joint": 0.0,
        }

    class control(LeggedRobotCfg.control):
        # 前两个关节走位置 PD，轮子走速度 PD（和复旦一致）
        pos_action_scale = 0.5
        vel_action_scale = 10.0
        # PD Drive parameters（按关节名子串匹配，键之间不能互相包含）
        # kp 从复旦的 20 提到 40：v11 的腿更折，保持 0.205 m 需要 ~7.5 N·m 的膝力矩，
        # kp=20 时静差 0.38 rad（会把腿折进轮舱），40 时 0.19 rad。kd/kp 保持 0.05。
        stiffness = {"rear_joint": 40.0, "rear_child1": 40.0, "Wheel": 0.0}
        # 第四版起 kd 提到 3.0：动作延迟 10 ms + 执行器滞后 15 ms 相当于 25 ms 的等效延迟，
        # kp=40 配 kd=2 在这种延迟下容易起极限环（腿抖），加阻尼是最直接的解法。
        damping = {"rear_joint": 3.0, "rear_child1": 3.0, "Wheel": 0.3}

    class asset(LeggedRobotCfg.asset):
        file = V11_TRAIN_URDF
        name = "rl_chassis"
        offset = 0.0
        # 虚拟腿参数（来自 v11 MJCF 闭链几何的标定）：
        #   theta1 = leg_a1 + leg_sign * q_rear
        #   theta2 = leg_a2 + leg_sign * q_shank
        #   end_x  = offset + l1*cos(theta1) + l2*cos(theta1+theta2)   (+x 前, +z 上)
        #   end_z  = l1*sin(theta1) + l2*sin(theta1+theta2)
        #   L0 = hypot(end_x, end_z)，theta0 = atan2(end_x, -end_z)（腿竖直向下时 theta0 = 0）
        l1 = 0.215000066
        l2 = 0.253999977
        leg_a1 = 3.139319059
        leg_a2 = -3.701985976
        leg_sign = [-1.0, 1.0]  # [左腿, 右腿]
        foot_name = "Wheel"
        dof_armature = {
            "rear_joint": 0.0,
            "rear_child1": 0.0,
            "Wheel": 0.0,
        }
        penalize_contacts_on = []
        terminate_after_contacts_on = []
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False


class WheelLeggedCfgPPO(LeggedRobotCfgPPO):
    class runner(LeggedRobotCfgPPO.runner):
        # logging
        experiment_name = "wheel_legged_v11"
        max_iterations = 2000
