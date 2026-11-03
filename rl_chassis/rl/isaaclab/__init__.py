"""Isaac Lab port of the v11/v12 wheel-legged locomotion task.

用法（仓库根目录）::

    PYTHONPATH=. ~/IsaacLab-3.0/.venv/bin/isaaclab train --rl_library rsl_rl \\
        --external_callback rl_chassis.rl.isaaclab.register \\
        --task Isaac-WheelLegged-Flat-Direct --num_envs 1024

`--external_callback` 会在 CLI 解析任务前导入本模块，从而把任务注册进 gym。
"""

import gymnasium as gym

from . import agents  # noqa: F401

_TASK_ID = "Isaac-WheelLegged-Flat-Direct"


def register() -> None:
    """注册轮腿平地任务；重复调用是安全的。"""
    if _TASK_ID in gym.registry:
        return
    gym.register(
        id=_TASK_ID,
        entry_point=f"{__name__}.wheel_env:WheelLeggedEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.wheel_env_cfg:WheelLeggedFlatEnvCfg",
            "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_cfg:WheelLeggedPPORunnerCfg",
        },
    )


register()
