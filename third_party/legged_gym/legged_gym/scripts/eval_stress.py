"""SafeRecovery Benchmark — Recovery Stress Test + Perturbation Sweep.

Two evaluation modes:
1. Recovery stress test: forced high-frequency perturbations to generate >=100 recovery events
2. Perturbation sweep: evaluate at 3 force bands (low/medium/high)

All methods evaluated under IDENTICAL benchmark wrapper (same reset, fall, timeout logic).
"""

import os
import json
import sys
from datetime import datetime

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch


def evaluate_single(task_name, load_run, num_eval_steps=5000, num_envs=64,
                    enable_perturbation=True, force_range=None,
                    interval_range=None, disable_cat_termination=True):
    """Run evaluation with identical benchmark wrapper for all methods."""
    argv_backup = sys.argv
    sys.argv = ["eval", "--task=%s" % task_name, "--num_envs=%d" % num_envs, "--headless"]
    args = get_args()
    sys.argv = argv_backup

    env_cfg, train_cfg = task_registry.get_cfgs(name=task_name)
    env_cfg.env.num_envs = num_envs
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.perturbation.enabled = enable_perturbation
    if force_range is not None:
        env_cfg.perturbation.force_range = force_range
    if interval_range is not None:
        env_cfg.perturbation.interval_range = interval_range

    # CRITICAL: Disable CaT termination during evaluation for fairness
    # All methods evaluated under same conditions — training differs, eval does not
    if disable_cat_termination:
        env_cfg.safety.enable_constraint_termination = False

    env, _ = task_registry.make_env(name=task_name, args=args, env_cfg=env_cfg)
    obs = env.get_observations()

    train_cfg.runner.resume = True
    if load_run is not None:
        train_cfg.runner.load_run = load_run
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=task_name, args=args, train_cfg=train_cfg
    )
    policy = ppo_runner.get_inference_policy(device=env.device)

    env.safety_logger.reset()
    completed_episodes = 0
    episode_lengths = []
    cur_ep_len = torch.zeros(num_envs, dtype=torch.int64, device=env.device)

    for step in range(num_eval_steps):
        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())
        cur_ep_len += 1
        done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
        if len(done_ids) > 0:
            for idx in done_ids:
                episode_lengths.append(cur_ep_len[idx].item())
            cur_ep_len[done_ids] = 0
            completed_episodes += len(done_ids)

    summary = env.get_safety_summary()
    dt = env.dt
    total_sim_time = num_eval_steps * dt

    summary["locomotion/completed_episodes"] = completed_episodes
    summary["locomotion/mean_episode_length_steps"] = (
        sum(episode_lengths) / len(episode_lengths) if episode_lengths else float("nan")
    )
    summary["locomotion/mean_episode_length_sec"] = (
        summary["locomotion/mean_episode_length_steps"] * dt
        if episode_lengths else float("nan")
    )
    summary["safety/violations_per_sec"] = summary["safety/total_violations"] / (total_sim_time * num_envs) if total_sim_time > 0 else 0
    summary["recovery/falls_per_min"] = summary["recovery/fall_count"] / (total_sim_time * num_envs / 60.0) if total_sim_time > 0 else 0
    summary["meta/task"] = task_name
    summary["meta/load_run"] = str(load_run)
    summary["meta/force_range"] = str(force_range)
    summary["meta/interval_range"] = str(interval_range)
    summary["meta/cat_termination_disabled"] = disable_cat_termination

    env.gym.destroy_sim(env.sim)
    return summary


def print_table(results, title):
    methods = list(results.keys())
    key_metrics = [
        ("vel_tracking_err", "locomotion/mean_vel_tracking_error"),
        ("violations/sec", "safety/violations_per_sec"),
        ("falls/min", "recovery/falls_per_min"),
        ("fall_count", "recovery/fall_count"),
        ("recovery_success", "recovery/success_rate"),
        ("recovery_attempts", "recovery/total_attempts"),
        ("time_to_upright", "recovery/mean_time_to_upright"),
        ("viol/fall", "coupling/violations_per_fall"),
        ("viol/rec_sec", "coupling/violations_per_recovery_sec"),
        ("pct_rec_w_viol", "coupling/pct_recovery_with_violation"),
    ]

    col_w = 18
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    header = "  %-25s" % "Metric"
    for m in methods:
        header += ("  %-" + str(col_w) + "s") % m
    print(header)
    print("-" * (27 + (col_w + 2) * len(methods)))
    for label, k in key_metrics:
        row = "  %-25s" % label
        for m in methods:
            v = results[m].get(k, float("nan"))
            if isinstance(v, float):
                row += ("  %-" + str(col_w) + "s") % ("%.4f" % v)
            else:
                row += ("  %-" + str(col_w) + "s") % str(v)
        print(row)
    print("=" * 80)


def main():
    log_base = "/home/hurricane/RL/CSGLoco/legged_gym/logs"

    def latest_run(exp):
        d = os.path.join(log_base, exp)
        if not os.path.exists(d):
            return None
        runs = sorted(os.listdir(d))
        return runs[-1] if runs else None

    tasks = {
        "Vanilla": ("safe_recovery_a1", latest_run("safe_recovery_a1")),
        "CaT": ("safe_recovery_a1_cat", latest_run("safe_recovery_a1_cat")),
        "Recovery": ("safe_recovery_a1_fallen", latest_run("safe_recovery_a1_fallen")),
    }

    all_results = {}

    # ========== Part 1: Recovery Stress Test ==========
    # High-frequency, high-force perturbations to generate many recovery events
    print("\n" + "#" * 80)
    print("# PART 1: Recovery Stress Test")
    print("# Aggressive perturbations (100-200N, interval 1-3s) for 10000 steps")
    print("#" * 80)

    stress_results = {}
    stress_kwargs = dict(
        num_eval_steps=10000,
        num_envs=64,
        force_range=[100, 200],
        interval_range=[1.0, 3.0],  # more frequent perturbations
        disable_cat_termination=True,  # FAIR: same eval for all
    )

    for method, (task, run) in tasks.items():
        if run is None:
            print("Skipping %s — no checkpoint" % method)
            continue
        print("\n>>> Stress test: %s" % method)
        stress_results[method] = evaluate_single(task, run, **stress_kwargs)

    if stress_results:
        print_table(stress_results, "Recovery Stress Test (100-200N, 1-3s interval, 10k steps)")
        all_results["stress_test"] = stress_results

    # ========== Part 2: Perturbation Sweep ==========
    print("\n" + "#" * 80)
    print("# PART 2: Perturbation Sweep (3 force bands)")
    print("#" * 80)

    bands = {
        "low": [30, 60],
        "medium": [80, 150],
        "high": [150, 250],
    }

    sweep_results = {}
    for band_name, force_range in bands.items():
        print("\n--- Force band: %s (%s N) ---" % (band_name, force_range))
        band_results = {}
        sweep_kwargs = dict(
            num_eval_steps=5000,
            num_envs=64,
            force_range=force_range,
            interval_range=[2.0, 5.0],
            disable_cat_termination=True,
        )
        for method, (task, run) in tasks.items():
            if run is None:
                continue
            print(">>> %s @ %s" % (method, band_name))
            band_results[method] = evaluate_single(task, run, **sweep_kwargs)
        if band_results:
            print_table(band_results, "Perturbation Sweep: %s (%s N)" % (band_name, force_range))
            sweep_results[band_name] = band_results

    all_results["perturbation_sweep"] = sweep_results

    # ========== Save ==========
    out_dir = os.path.join(log_base, "safe_recovery_eval")
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, "stress_sweep_%s.json" % ts)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print("\n\nAll results saved to: %s" % out_path)


if __name__ == "__main__":
    main()
