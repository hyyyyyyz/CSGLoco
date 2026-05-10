"""Unified policy evaluator.

Refactors the common evaluation pipeline previously duplicated across
eval_vanilla_cat_2000.py, eval_anymal_multiseed.py, eval_recovery_1500.py,
and the run_exp*.py family into a single function.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional, Sequence

# isaacgym must be imported before torch
import isaacgym  # noqa: F401
import torch

from legged_gym.envs import *  # noqa: F401,F403  registers tasks
from legged_gym.utils import get_args, task_registry


# ---------------------------------------------------------------------------
# Helpers


def find_latest_run(task_name: str, log_root: str) -> Optional[str]:
    """Return the most recently modified run directory under log_root/task_name, or None."""
    task_dir = Path(log_root) / task_name
    if not task_dir.exists():
        return None
    runs = sorted(
        (p for p in task_dir.iterdir() if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return runs[0].name if runs else None


def _build_args(task_name: str, num_envs: int) -> object:
    """Construct legged_gym args without leaking sys.argv to other code."""
    saved_argv = sys.argv
    sys.argv = ["eval", f"--task={task_name}", f"--num_envs={num_envs}", "--headless"]
    try:
        return get_args()
    finally:
        sys.argv = saved_argv


# ---------------------------------------------------------------------------
# Setting presets


SETTINGS = {
    # name: (force_range, fallen_start, native_termination, extra_overrides)
    "standard": ([80, 150], False, False, {}),
    "fallen":   ([80, 150], True,  False, {}),
    "native":   ([80, 150], False, True,  {}),
    "rough":    ([100, 250], False, False, {"rough": True}),
}


def _apply_rough(env_cfg) -> None:
    """Apply rough-condition domain randomization (used by --setting rough)."""
    env_cfg.terrain.mesh_type = "plane"
    env_cfg.terrain.measure_heights = False
    env_cfg.domain_rand.randomize_friction = True
    env_cfg.domain_rand.friction_range = [0.2, 1.5]
    env_cfg.domain_rand.randomize_base_mass = True
    env_cfg.domain_rand.added_mass_range = [-1.0, 2.0]
    env_cfg.noise.add_noise = True
    env_cfg.noise.noise_scales.dof_pos = 0.05
    env_cfg.noise.noise_scales.dof_vel = 0.15
    env_cfg.noise.noise_scales.ang_vel = 0.15
    env_cfg.noise.noise_scales.lin_vel = 0.1
    env_cfg.noise.noise_scales.gravity = 0.05
    env_cfg.noise.noise_scales.imu = 0.15
    env_cfg.perturbation.interval_range = [1.5, 4.0]


# ---------------------------------------------------------------------------
# Main entry


def evaluate_policy(
    task_name: str,
    load_run: str,
    checkpoint_iter: int,
    *,
    setting: str = "standard",
    num_envs: int = 128,
    num_eval_steps: int = 10000,
) -> dict:
    """Run a SafeRecovery evaluation episode and return the safety_logger summary.

    Parameters
    ----------
    task_name        legged_gym task name (e.g. "safe_recovery_a1").
    load_run         run timestamp directory under logs/<task>/.
    checkpoint_iter  iteration index of the checkpoint to load.
    setting          one of "standard", "fallen", "native", "rough".
    """
    if setting not in SETTINGS:
        raise ValueError(f"unknown setting {setting!r}; valid: {list(SETTINGS)}")
    force_range, fallen_start, native_termination, extra = SETTINGS[setting]

    args = _build_args(task_name, num_envs)
    env_cfg, train_cfg = task_registry.get_cfgs(name=task_name)
    env_cfg.env.num_envs = num_envs
    env_cfg.terrain.curriculum = False

    if extra.get("rough"):
        _apply_rough(env_cfg)
    else:
        env_cfg.noise.add_noise = False
        env_cfg.domain_rand.randomize_friction = False

    env_cfg.perturbation.enabled = True
    env_cfg.perturbation.force_range = force_range
    env_cfg.perturbation.interval_range = getattr(env_cfg.perturbation, "interval_range", [2.0, 5.0])

    env_cfg.safety.enable_constraint_termination = bool(native_termination)

    if fallen_start:
        if not hasattr(env_cfg, "fallen_start"):
            class _FallenStart:
                enabled = True
                fraction = 1.0
                roll_range = [-2.5, 2.5]
                pitch_range = [-1.5, 1.5]
                height_range = [0.08, 0.15]
            env_cfg.fallen_start = _FallenStart()
        else:
            env_cfg.fallen_start.enabled = True
            env_cfg.fallen_start.fraction = 1.0

    env, _ = task_registry.make_env(name=task_name, args=args, env_cfg=env_cfg)
    obs = env.get_observations()

    train_cfg.runner.resume = True
    train_cfg.runner.load_run = load_run
    train_cfg.runner.checkpoint = checkpoint_iter
    ppo_runner, _ = task_registry.make_alg_runner(
        env=env, name=task_name, args=args, train_cfg=train_cfg
    )
    policy = ppo_runner.get_inference_policy(device=env.device)

    env.safety_logger.reset()
    completed = 0
    episode_lengths = []
    cur_len = torch.zeros(num_envs, dtype=torch.int64, device=env.device)

    for _ in range(num_eval_steps):
        actions = policy(obs.detach())
        obs, _r, _rew, dones, _info = env.step(actions.detach())
        cur_len += 1
        done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
        if len(done_ids) > 0:
            episode_lengths.extend(cur_len[done_ids].cpu().tolist())
            cur_len[done_ids] = 0
            completed += int(done_ids.numel())

    summary = env.safety_logger.summary()
    summary["_meta"] = {
        "task": task_name,
        "load_run": load_run,
        "checkpoint": checkpoint_iter,
        "setting": setting,
        "num_envs": num_envs,
        "num_eval_steps": num_eval_steps,
        "completed_episodes": completed,
        "mean_episode_length": (sum(episode_lengths) / len(episode_lengths)) if episode_lengths else 0.0,
    }
    return summary
