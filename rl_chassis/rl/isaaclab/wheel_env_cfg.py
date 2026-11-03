"""Flat-ground wheel-legged task configuration for Isaac Lab 3.0.

数值全部对齐 ``plane_v11/wheel_legged_gym`` 的第四版（v11_flat_slow）训练配置：
100 Hz 控制（sim dt 0.005 s，decimation 2）、kp/kd 40/3（轮 0/0.3）、
动作尺度 0.5（髋/膝）与 10（轮）、25 维本体观测、6 维动作。

和原 Isaac Gym 配置的差异（这一轮为了先站稳）：

* 速度指令范围收窄到 ±1.2 m/s、±1.0 rad/s（原 ±2.0），并有 50% 概率直接给零指令；
* 训练资产用串联腿 ``out/train_urdf/rl_chassis_v11_free.usda``（去掉转换器补的
  固定基座关节），闭链映射仍在部署侧由 sim2sim 完成。
"""

from __future__ import annotations

from pathlib import Path

from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.sim.schemas import PhysxArticulationCfg, PhysxRigidBodyCfg

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

_REPO_ROOT = Path(__file__).resolve().parents[3]
TRAIN_USD = str(_REPO_ROOT / "out" / "train_urdf" / "rl_chassis_v11_free.usda")

# 动作/观测里的关节顺序：左髋、左膝、左轮、右髋、右膝、右轮。
JOINT_ORDER = (
    "Left_rear_joint",
    "Left_rear_child1_joint",
    "Left_Wheel_joint",
    "Right_rear_joint",
    "Right_rear_child1_joint",
    "Right_Wheel_joint",
)
DEFAULT_JOINT_POS = {
    "Left_rear_joint": -0.274884,
    "Left_rear_child1_joint": 0.517721,
    "Left_Wheel_joint": 0.0,
    "Right_rear_joint": 0.274883,
    "Right_rear_child1_joint": -0.517722,
    "Right_Wheel_joint": 0.0,
}


@configclass
class WheelLeggedEventCfg:
    """训练时的域随机化；只保留对 sim2sim 迁移影响最大的两项。"""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.6, 1.4),
            "dynamic_friction_range": (0.45, 1.1),
            "restitution_range": (0.0, 0.2),
            "num_buckets": 64,
        },
    )
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (-0.5, 1.5),
            "operation": "add",
        },
    )


@configclass
class WheelLeggedSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.PhysxRigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=TRAIN_USD,
            activate_contact_sensors=True,
            rigid_props=PhysxRigidBodyCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=[
                PhysxArticulationCfg(
                    enabled_self_collisions=False,
                    solver_position_iteration_count=4,
                    solver_velocity_iteration_count=0,
                ),
            ],
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.18),
            joint_pos=DEFAULT_JOINT_POS,
        ),
        actuators={
            # 力矩由 env 里的 PD 逐物理步算好，驱动本身不再加刚度/阻尼。
            "all": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=0.0,
                damping=0.0,
                joint_effort_limit=1000.0,
            ),
        },
        soft_joint_pos_limit_factor=0.98,
    )
    contact_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/.*",
        history_length=3,
        update_period=0.005,
        track_air_time=False,
    )
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)),
    )


@configclass
class WheelLeggedFlatEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 20.0
    decimation = 2
    observation_space = 25
    action_space = 6
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=0.005,
        render_interval=2,
        physics=PhysxCfg(gpu_max_rigid_patch_count=2**20),
        physics_material=sim_utils.PhysxRigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # scene
    scene: WheelLeggedSceneCfg = WheelLeggedSceneCfg(num_envs=1024, env_spacing=1.5)

    # events
    events: WheelLeggedEventCfg = WheelLeggedEventCfg()

    # command ranges
    cmd_lin_vel_range = (-1.2, 1.2)
    cmd_yaw_vel_range = (-1.0, 1.0)
    cmd_height_range = (0.14, 0.24)
    command_resample_time_s = 5.0
    standing_command_prob = 0.5

    # PD / actuator model（与 plane_v11 一致）
    kp = 40.0
    kd = 3.0
    wheel_kd = 0.3
    pos_action_scale = 0.5
    vel_action_scale = 10.0
    effort_limit = (40.0, 40.0, 20.0, 40.0, 40.0, 20.0)
    actuator_tau = 0.015
    action_delay_steps = 1

    # observation noise（加在缩放后的观测上，与 README_sim2sim 表格一致）
    obs_noise_ang_vel = 0.05
    obs_noise_gravity = 0.05
    obs_noise_dof_pos = 0.02
    obs_noise_dof_vel = 0.075

    # termination
    fail_tilt_cos = -0.1
    fail_to_terminal_time_s = 1.0
