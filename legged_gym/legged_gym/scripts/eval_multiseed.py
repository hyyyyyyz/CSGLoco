"""SafeRecovery Benchmark — multi-seed evaluation with statistics."""

import os
import json
import sys
import math
from datetime import datetime
from collections import defaultdict

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch


def evaluate_single(task_name, load_run, num_eval_steps=5000, num_envs=64,
                    enable_perturbation=True, force_range=None):
    """Run evaluation for a single task/checkpoint."""
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
    env_cfg.perturbation.interval_range = [2.0, 5.0]

    # FAIR EVALUATION: disable CaT constraint termination for all methods
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
    total_env_steps = num_eval_steps * num_envs
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

    env.gym.destroy_sim(env.sim)
    return summary


def mean_std_ci(values):
    """Compute mean, std, and 95% CI for a list of values."""
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    m = sum(values) / n
    if n == 1:
        return m, 0.0, m, m
    var = sum((x - m) ** 2 for x in values) / (n - 1)
    s = math.sqrt(var)
    t_crit = 2.776 if n == 5 else 2.571 if n == 6 else 2.262 if n == 10 else 1.96  # t-dist approx
    ci_half = t_crit * s / math.sqrt(n)
    return m, s, m - ci_half, m + ci_half


def main():
    log_base = "/home/hurricane/RL/CSGLoco/legged_gym/logs"

    tasks = {
        "Vanilla": "safe_recovery_a1",
        "CaT": "safe_recovery_a1_cat",
        "Recovery": "safe_recovery_a1_fallen",
    }

    eval_kwargs = dict(num_eval_steps=5000, num_envs=64, force_range=[80, 150])

    key_metrics = [
        "locomotion/mean_vel_tracking_error",
        "locomotion/mean_episode_length_sec",
        "safety/total_violations",
        "safety/violations_per_sec",
        "safety/torque_violation_rate",
        "safety/contact_force_violation_rate",
        "safety/orientation_violation_rate",
        "recovery/fall_count",
        "recovery/falls_per_min",
        "recovery/success_rate",
        "recovery/mean_time_to_upright",
        "coupling/violations_during_recovery",
        "coupling/violations_per_fall",
        "coupling/viol_per_fallen_sec",
        "coupling/viol_per_active_rec_sec",
        "coupling/pct_attempts_with_violation",
        "coupling/pct_success_with_violation",
    ]

    all_results = {}

    for method, task in tasks.items():
        exp_dir = os.path.join(log_base, task)
        if not os.path.exists(exp_dir):
            print("Skipping %s — no logs" % method)
            continue

        runs = sorted(os.listdir(exp_dir))
        if not runs:
            print("Skipping %s — no runs" % method)
            continue

        print("\n%s: found %d runs: %s" % (method, len(runs), runs))
        seed_results = []

        for run in runs:
            # Check if model exists
            run_dir = os.path.join(exp_dir, run)
            models = [f for f in os.listdir(run_dir) if f.startswith("model_") and f.endswith(".pt")]
            if not models:
                print("  Skipping %s — no model" % run)
                continue
            # Use highest iteration model
            best_model_iter = max(int(f.replace("model_", "").replace(".pt", "")) for f in models)
            if best_model_iter < 500:
                print("  Skipping %s — model_%d too early" % (run, best_model_iter))
                continue

            print("  Evaluating %s (model_%d)..." % (run, best_model_iter))
            try:
                res = evaluate_single(task, run, **eval_kwargs)
                seed_results.append(res)
            except Exception as e:
                print("  ERROR evaluating %s: %s" % (run, e))

        if not seed_results:
            print("  No valid results for %s" % method)
            continue

        # Aggregate statistics
        stats = {"n_seeds": len(seed_results)}
        for k in key_metrics:
            vals = [r.get(k, float("nan")) for r in seed_results]
            vals = [v for v in vals if not (isinstance(v, float) and math.isnan(v))]
            if vals:
                m, s, ci_lo, ci_hi = mean_std_ci(vals)
                stats[k + "/mean"] = m
                stats[k + "/std"] = s
                stats[k + "/ci95_lo"] = ci_lo
                stats[k + "/ci95_hi"] = ci_hi
            else:
                stats[k + "/mean"] = float("nan")

        all_results[method] = stats

    # Print comparison table
    methods = list(all_results.keys())
    if len(methods) < 2:
        print("ERROR: Need at least 2 methods")
        sys.exit(1)

    col_w = 22
    print("\n" + "=" * 100)
    print("SafeRecovery Benchmark — Multi-Seed Comparison (mean +/- std, 95%% CI)")
    print("=" * 100)

    header = "  %-35s" % "Metric"
    for m in methods:
        n = all_results[m]["n_seeds"]
        header += ("  %-" + str(col_w) + "s") % ("%s (n=%d)" % (m, n))
    print(header)
    print("-" * (37 + (col_w + 2) * len(methods)))

    for k in key_metrics:
        short = k.split("/")[-1]
        row = "  %-35s" % short
        for m in methods:
            s = all_results[m]
            mean_val = s.get(k + "/mean", float("nan"))
            std_val = s.get(k + "/std", 0)
            if isinstance(mean_val, float) and math.isnan(mean_val):
                row += ("  %-" + str(col_w) + "s") % "N/A"
            else:
                row += ("  %-" + str(col_w) + "s") % ("%.3f +/- %.3f" % (mean_val, std_val))
        print(row)

    print("=" * 100)

    # Save
    out_dir = os.path.join(log_base, "safe_recovery_eval")
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, "multiseed_%s.json" % ts)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print("\nResults saved to: %s" % out_path)


if __name__ == "__main__":
    main()
