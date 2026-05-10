"""SafeRecovery Benchmark — Dense Recovery Evaluation.

Evaluates all methods under standardized fallen-start conditions.
Each method gets 100+ recovery attempts for statistical robustness.

Protocol: Enable fallen-start for ALL methods during evaluation (50% fallen init).
Run for 20k steps with 128 envs. This guarantees many fall events even for robust policies.
"""

import os
import json
import sys
from datetime import datetime

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch


def evaluate_fallen_start(task_name, load_run, num_eval_steps=20000, num_envs=128):
    """Evaluate with forced fallen-start for dense recovery testing."""
    argv_backup = sys.argv
    sys.argv = ["eval", "--task=%s" % task_name, "--num_envs=%d" % num_envs, "--headless"]
    args = get_args()
    sys.argv = argv_backup

    env_cfg, train_cfg = task_registry.get_cfgs(name=task_name)
    env_cfg.env.num_envs = num_envs
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False

    # Force fallen-start for dense recovery evaluation
    env_cfg.fallen_start = type('fallen_start', (), {
        'enabled': True,
        'fraction': 0.5,
        'roll_range': [-2.5, 2.5],
        'pitch_range': [-1.5, 1.5],
        'height_range': [0.08, 0.15],
    })()

    # Enable perturbation to generate additional falls
    env_cfg.perturbation.enabled = True
    env_cfg.perturbation.force_range = [100, 200]
    env_cfg.perturbation.interval_range = [2.0, 5.0]

    # FAIR: disable CaT termination
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

    for step in range(num_eval_steps):
        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())

    summary = env.get_safety_summary()
    dt = env.dt
    total_sim_time = num_eval_steps * dt

    summary["recovery/falls_per_min"] = summary["recovery/fall_count"] / (total_sim_time * num_envs / 60.0) if total_sim_time > 0 else 0
    summary["safety/violations_per_sec"] = summary["safety/total_violations"] / (total_sim_time * num_envs) if total_sim_time > 0 else 0
    summary["meta/task"] = task_name
    summary["meta/load_run"] = str(load_run)
    summary["meta/eval_type"] = "dense_recovery"
    summary["meta/num_eval_steps"] = num_eval_steps
    summary["meta/num_envs"] = num_envs

    env.gym.destroy_sim(env.sim)
    return summary


def main():
    log_base = "/home/hurricane/RL/CSGLoco/legged_gym/logs"

    tasks = {
        "Vanilla": ("safe_recovery_a1", "Mar16_23-37-58_"),
        "CaT": ("safe_recovery_a1_cat", "Mar17_00-12-03_"),
        "Recovery": ("safe_recovery_a1_fallen", "Mar17_12-09-35_"),
    }

    results = {}
    for method, (task, run) in tasks.items():
        print("\n>>> Dense recovery eval: %s" % method)
        results[method] = evaluate_fallen_start(task, run)
        r = results[method]
        print("  Falls: %d, Attempts: %d, Success: %.1f%%, Time-to-upright: %.2fs" % (
            r["recovery/fall_count"],
            r["recovery/total_attempts"],
            r["recovery/success_rate"] * 100,
            r["recovery/mean_time_to_upright"] if r["recovery/mean_time_to_upright"] == r["recovery/mean_time_to_upright"] else 0,
        ))

    # Print comparison
    methods = list(results.keys())
    key_metrics = [
        ("fall_count", "recovery/fall_count"),
        ("recovery_attempts", "recovery/total_attempts"),
        ("success_rate", "recovery/success_rate"),
        ("time_to_upright", "recovery/mean_time_to_upright"),
        ("falls/min", "recovery/falls_per_min"),
        ("violations/sec", "safety/violations_per_sec"),
        ("total_events", "safety/total_events"),
        ("viol/fall", "coupling/violations_per_fall"),
        ("viol/rec_sec", "coupling/violations_per_recovery_sec"),
        ("pct_rec_w_viol", "coupling/pct_recovery_with_violation"),
        ("torque_viol_rec", "coupling/torque_viol_during_recovery"),
        ("orient_viol_rec", "coupling/orient_viol_during_recovery"),
    ]

    col_w = 18
    print("\n" + "=" * 80)
    print("Dense Recovery Evaluation (fallen-start + perturbation, 20k steps, 128 envs)")
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

    out_dir = os.path.join(log_base, "safe_recovery_eval")
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, "dense_recovery_%s.json" % ts)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nResults saved to: %s" % out_path)


if __name__ == "__main__":
    main()
