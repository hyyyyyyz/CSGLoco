"""SafeRecovery Benchmark — evaluation script with fair conditions.

CRITICAL: All methods evaluated under IDENTICAL benchmark wrapper.
CaT termination is DISABLED during evaluation. Training differs; eval does not.
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
                    interval_range=None):
    """Run evaluation for a single task/checkpoint under fair conditions."""
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

    # FAIR EVALUATION: disable CaT termination for ALL methods
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
    summary["meta/num_eval_steps"] = num_eval_steps
    summary["meta/total_sim_time_per_env"] = total_sim_time
    summary["meta/force_range"] = str(env_cfg.perturbation.force_range)
    summary["meta/cat_termination_disabled"] = True

    env.gym.destroy_sim(env.sim)
    return summary


def print_comparison(results):
    methods = list(results.keys())
    sections = {
        "Axis 1 (Locomotion)": [
            ("vel_tracking_error", "locomotion/mean_vel_tracking_error"),
            ("episode_length (sec)", "locomotion/mean_episode_length_sec"),
        ],
        "Axis 2 (Safety)": [
            ("torque_viol_rate", "safety/torque_violation_rate"),
            ("contact_viol_rate", "safety/contact_force_violation_rate"),
            ("orient_viol_rate", "safety/orientation_violation_rate"),
            ("total_violations", "safety/total_violations"),
            ("violations/sec", "safety/violations_per_sec"),
        ],
        "Axis 3 (Recovery)": [
            ("fall_count", "recovery/fall_count"),
            ("falls/min", "recovery/falls_per_min"),
            ("recovery_success", "recovery/success_rate"),
            ("recovery_attempts", "recovery/total_attempts"),
            ("time_to_upright", "recovery/mean_time_to_upright"),
            ("timeout_count", "recovery/timeout_count"),
        ],
        "Axis 4 (Coupling)": [
            ("viol_during_recovery", "coupling/violations_during_recovery"),
            ("viol_per_fall", "coupling/violations_per_fall"),
            ("viol_per_rec_sec", "coupling/violations_per_recovery_sec"),
            ("pct_rec_w_viol", "coupling/pct_recovery_with_violation"),
        ],
    }

    col_w = max(15, max(len(m) for m in methods) + 2)
    header = "  %-30s" % "Metric"
    for m in methods:
        header += (" %" + str(col_w) + "s") % m
    sep = "-" * (32 + (col_w + 1) * len(methods))

    print("\n" + "=" * len(sep))
    print("SafeRecovery Benchmark — 4-Axis Comparison (FAIR EVAL)")
    print("NOTE: CaT termination DISABLED for all methods during evaluation")
    print("=" * len(sep))

    for section, keys in sections.items():
        print("\n  [%s]" % section)
        print(sep)
        for label, k in keys:
            row = "  %-30s" % label
            for m in methods:
                v = results[m].get(k, float("nan"))
                if isinstance(v, float):
                    row += (" %" + str(col_w) + ".4f") % v
                else:
                    row += (" %" + str(col_w) + "s") % str(v)
            print(row)

    print("\n" + "=" * len(sep))


def main():
    log_base = "/home/hurricane/RL/CSGLoco/legged_gym/logs"

    def latest_run(exp):
        d = os.path.join(log_base, exp)
        if not os.path.exists(d):
            return None
        runs = sorted(os.listdir(d))
        return runs[-1] if runs else None

    vanilla_run = latest_run("safe_recovery_a1")
    cat_run = latest_run("safe_recovery_a1_cat")
    fallen_run = latest_run("safe_recovery_a1_fallen")

    print("Vanilla:  %s" % vanilla_run)
    print("CaT:      %s" % cat_run)
    print("Recovery: %s" % fallen_run)

    # Use first/original run for each (not multi-seed runs)
    # Hardcode the original run names to avoid picking multi-seed runs
    vanilla_run = "Mar16_23-37-58_"
    cat_run = "Mar17_00-12-03_"
    fallen_run = "Mar17_12-09-35_"

    results = {}
    eval_kwargs = dict(num_eval_steps=5000, num_envs=64, force_range=[80, 150])

    print("\n>>> Evaluating Vanilla PPO (fair eval)...")
    results["Vanilla"] = evaluate_single("safe_recovery_a1", vanilla_run, **eval_kwargs)

    print("\n>>> Evaluating CaT PPO (fair eval, termination OFF)...")
    results["CaT"] = evaluate_single("safe_recovery_a1_cat", cat_run, **eval_kwargs)

    print("\n>>> Evaluating Recovery PPO (fair eval)...")
    results["Recovery"] = evaluate_single("safe_recovery_a1_fallen", fallen_run, **eval_kwargs)

    print_comparison(results)

    out_dir = os.path.join(log_base, "safe_recovery_eval")
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, "fair_comparison_%s.json" % ts)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nResults saved to: %s" % out_path)


if __name__ == "__main__":
    main()
