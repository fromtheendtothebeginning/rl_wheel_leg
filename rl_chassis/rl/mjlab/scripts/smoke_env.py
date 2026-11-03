#!/usr/bin/env python3
"""环境冒烟：构建 + reset + 若干步零动作，检查维度与闭链一致性。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch


def main() -> int:
  import mjlab_wheelleg  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg
  from mjlab_wheelleg import TASK_ID

  env_cfg = load_env_cfg(TASK_ID)
  env_cfg.scene.num_envs = 8
  env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0")
  print(env)

  obs, _ = env.reset()
  print("obs keys:", {k: tuple(v.shape) for k, v in obs.items()})

  actor = obs["actor"]
  assert actor.shape == (8, 25), actor.shape

  term = env.action_manager.get_term("wheelleg")
  for step in range(20):
    obs, rew, term_, trunc, infos = env.step(torch.zeros(8, 6, device="cuda:0"))
    if step % 10 == 0:
      q = env.scene["robot"].data.joint_pos[:, term._leg_idx["left"]]
      d_front, d_rear = term._d["left"]
      print(
        f"step {step}: rew={rew.mean().item():+.3f} "
        f"tau_real_max={term.tau_real.abs().max().item():.2f} "
        f"L_front={q[0,0].item():+.3f} L_rear={q[0,4].item():+.3f} "
        f"d_front={d_front[0].item():+.3f} d_rear={d_rear[0].item():+.3f} "
        f"obs[0,:3]={actor[0,:3].tolist() if False else obs['actor'][0,:3].tolist()}"
      )
  print("z =", env.scene["robot"].data.root_link_pos_w[:, 2].tolist())
  print("gravity_b[0] =", env.scene["robot"].data.projected_gravity_b[0].tolist())
  print("SMOKE OK")
  env.close()
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
