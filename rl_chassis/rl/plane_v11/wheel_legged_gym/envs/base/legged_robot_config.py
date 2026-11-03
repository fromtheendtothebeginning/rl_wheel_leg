# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from .base_config import BaseConfig


class LeggedRobotCfg(BaseConfig):
    class env:
        # 复旦用 8192（云 GPU）；本地 4060 8G，先降到 2048
        num_envs = 2048
        num_observations = 25
        num_privileged_obs = (
            num_observations + 7 * 11 + 3 + 6 * 5 + 3 + 3
        )  # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise
        obs_history_length = 5  # number of observations stacked together
        obs_history_dec = 1
        num_actions = 6
        env_spacing = 3.0  # not used with heightfields/trimeshes
        send_timeouts = True  # send time out information to the algorithm
        episode_length_s = 20  # episode length in seconds
        dof_vel_use_pos_diff = True
        fail_to_terminal_time_s = 1

    class terrain:
        # 第四版：回到平地（第二/三版的地形在"高站姿+前进"下 MuJoCo 不稳，先只做平地；
        # 但保留第三版的慢电机建模（动作延迟 + 执行器滞后）与加大的惩罚）
        # mesh_type = "trimesh"  # "heightfield" # none, plane, heightfield or trimesh
        mesh_type = "plane"
        horizontal_scale = 0.1  # [m]
        vertical_scale = 0.005  # [m]
        border_size = 25  # [m]
        curriculum = True
        static_friction = 0.5
        dynamic_friction = 0.5
        restitution = 0.5
        # rough terrain only:
        measure_heights = True
        measured_points_x = [
            -0.5,
            -0.4,
            -0.3,
            -0.2,
            -0.1,
            0.0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
        ]  # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]
        selected = False  # select a unique terrain type and pass all arguments
        terrain_kwargs = None  # Dict of arguments for selected terrain
        # 难度 = 行号/num_rows，台阶高度 = 0.05 + 0.18*difficulty（行 0~9 → 5~21 cm）
        # 从平地模型续训，先压在低难度行上，靠课程自己往上爬
        # 第三版（抬高站姿）：起点接住当前水平（0~5 级 = 台阶 5~14 cm），再自己往上爬
        max_init_terrain_level = 6  # starting curriculum state
        terrain_length = 8.0
        terrain_width = 8.0
        num_rows = 10  # number of terrain rows (levels)#原本是10
        num_cols = 20  # number of terrain cols (types)
        # 列决定类型: [平地, 斜坡, 粗糙斜坡, 台阶上, 台阶下, 离散障碍]
        terrain_proportions = [0.2, 0.2, 0.2, 0.2, 0.1, 0.1]
        # trimesh only:
        slope_treshold = (
            0.75  # slopes above this threshold will be corrected to vertical surfaces
        )

    class commands:
        curriculum = True
        basic_max_curriculum = 2.5
        advanced_max_curriculum = 1.5
        curriculum_threshold = 0.7
        num_commands = 3  # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 5.0  # time before command are changed[s]
        heading_command = False  # if true: compute ang vel command from heading error

        class ranges:
            lin_vel_x = [-2.0, 2.0]  # min max [m/s]
            ang_vel_yaw = [-2, 2]  # min max [rad/s]
            # 高度 = base_link 离地高度 = L0 + R_wheel - 0.11508
            # 第四版（平地）：0.14 ~ 0.24 m（对应 L0 = 0.195 ~ 0.295 m），
            # 常用骑行档 0.15~0.18，要过东西时抬到 0.22+
            height = [0.14, 0.24]
            heading = [-3.14, 3.14]

    class init_state:
        pos = [0.0, 0.0, 1.0]  # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        default_joint_angles = {  # target angles when action = 0.0
            "joint_a": 0.0,
            "joint_b": 0.0,
        }

    class control:
        control_type = "P"  # P: position, V: velocity, T: torques
        # PD Drive parameters:
        stiffness = {"joint_a": 10.0, "joint_b": 15.0}  # [N*m/rad]
        damping = {"joint_a": 1.0, "joint_b": 1.5}  # [N*m*s/rad]
        # 执行器一阶滞后（真实电机"响应没那么快"）：τ_applied 向 τ_cmd 以时间常数
        # actuator_tau 逼近；0 = 关掉（理想力矩源）。训练/部署要一致。
        actuator_tau = 0.015  # [s]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 2

    class asset:
        dof_armature = {
            "hip": 0.00,
            "thigh": 0.00,
            "calf": 0.00,
            "wheel": 0.000,
        }
        file = ""
        name = "legged_robot"  # actor name
        foot_name = "None"  # name of the feet bodies, used to index body state and contact force tensors
        offset = 0
        l1 = 0
        l2 = 0
        penalize_contacts_on = []
        terminate_after_contacts_on = []
        disable_gravity = False
        collapse_fixed_joints = True  # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        fix_base_link = False  # fixe the base of the robot
        default_dof_drive_mode = 3  # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = True  # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = (
            True  # Some .obj meshes must be flipped from y-up to z-up
        )

        density = 0.001
        angular_damping = 0.0
        linear_damping = 0.0
        max_angular_velocity = 1000.0
        max_linear_velocity = 1000.0
        armature = 0.0
        thickness = 0.01

    class domain_rand:
        randomize_friction = True
        friction_range = [0.6, 1.4]
        randomize_restitution = True
        restitution_range = [0.6, 1.0]
        randomize_base_mass = True
        added_mass_range = [-1.0, 2.0]
        randomize_inertia = True
        randomize_inertia_range = [0.9, 1.1]
        randomize_base_com = True
        rand_com_vec = [0.02, 0.02, 0.02]
        push_robots = False
        push_interval_s = 7
        max_push_vel_xy = 2.0
        # 增益/力矩都往「真机可能达不到」的方向放宽，
        # 部署时用比训练更软的 kp 或更弱的电机也能站住
        randomize_Kp = True
        randomize_Kp_range = [0.8, 1.2]
        randomize_Kd = True
        # 部署端可能要往上调 kd 压抖，所以上限放宽
        randomize_Kd_range = [0.7, 1.4]
        randomize_motor_torque = True
        randomize_motor_torque_range = [0.8, 1.1]
        randomize_default_dof_pos = True
        randomize_default_dof_pos_range = [-0.03, 0.03]
        randomize_action_delay = True
        delay_ms_range = [0, 10]

    class rewards:
        class scales:
            tracking_lin_vel = 1.0
            tracking_lin_vel_enhance = 1.0
            tracking_ang_vel = 1.0
            tracking_ang_vel_enhance = 1.0

            # theta0_equ_0 = 0.4
            base_height = 1.0
            nominal_state = -2.0   # 惩罚左右腿 theta0 不对称（防单腿甩）
            lin_vel_z = -2.0
            ang_vel_xy = -0.60 #-0.05
            orientation = -100.0 #-10

            # 第四版：稳定性惩罚再加大一档
            # （腿抖的根因是 kp=40 配 ~25 ms 等效执行延迟下的高增益极限环，
            #   所以同时把 kd 提到 3.0，并在这里压速度/加速度/动作变化）
            dof_vel = -4.0e-4
            dof_acc = -3.0e-6
            torques = -0.0008
            action_rate = -0.12
            action_smooth = -0.12

            collision = -1.0
            dof_pos_limits = -5.0

        only_positive_rewards = False  # if true negative total rewards are clipped at zero (avoids early termination problems)
        clip_single_reward = 1
        tracking_sigma = 0.25  # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = (
            0.97  # percentage of urdf limits, values above this limit are penalized
        )
        soft_dof_vel_limit = 1.0
        soft_torque_limit = 1.0
        base_height_target = 0.15
        max_contact_force = 100.0  # forces above this value are penalized

    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            dof_acc = 0.0025
            height_measurements = 5.0
            torque = 0.05

        clip_observations = 100.0
        clip_actions = 100.0

    class noise:
        add_noise = True
        noise_level = 1.0  # scales other values

        class noise_scales:
            dof_pos = 0.01*2
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

    # viewer camera:
    class viewer:
        ref_env = 0
        pos = [0, -2, 1]  # [m]
        lookat = [0, 0, 0]  # [m]

    class sim:
        dt = 0.005
        substeps = 1
        gravity = [0.0, 0.0, -9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0  # [m]
            bounce_threshold_velocity = 0.5  # 0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23  # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = (
                2  # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
            )


class LeggedRobotCfgPPO(BaseConfig):
    seed = 1
    runner_class_name = "OnPolicyRunner"

    class policy:
        init_noise_std = 0.5
        actor_hidden_dims = [128, 64, 32]
        critic_hidden_dims = [256, 128, 64]
        activation = "elu"  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

        # only for ActorCriticSequence
        num_encoder_obs = (
            LeggedRobotCfg.env.obs_history_length * LeggedRobotCfg.env.num_observations
        )
        latent_dim = 3  # at least 3 to estimate base linear velocity
        encoder_hidden_dims = [128, 64]

    class algorithm:
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4  # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.0e-3  # 5.e-4
        schedule = "adaptive"  # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.005
        max_grad_norm = 1.0

        extra_learning_rate = 1e-3

    class runner:
        policy_class_name = (
            "ActorCriticSequence"  # could be ActorCritic, ActorCriticSequence
        )
        algorithm_class_name = "PPO"
        num_steps_per_env = 48  # per iteration
        max_iterations = 5000  # number of policy updates

        # logging
        save_interval = 100  # check for potential saves every this many iterations
        experiment_name = "test"
        run_name = ""
        # load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # updated from load_run and chkpt
