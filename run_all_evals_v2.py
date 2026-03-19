"""SafeRecovery Benchmark — Complete evaluation suite for paper v2.
Runs: (1) Multi-seed standard eval, (2) Controlled fallen-start, (3) CaT native vs fair.
"""
import os, json, sys, math
from datetime import datetime
from collections import defaultdict

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch

LOG_BASE = "/home/hurricane/RL/CSGLoco/legged_gym/logs"
OUTDIR = "/home/hurricane/RL/CSGLoco/safe_recovery_eval_v2"
os.makedirs(OUTDIR, exist_ok=True)

# Mar18 3-seed batch runs
RUNS = {
    "Vanilla": {
        "task": "safe_recovery_a1",
        "seeds": {1: "Mar18_16-16-27_", 2: "Mar18_16-53-02_", 3: "Mar18_17-29-18_"},
    },
    "CaT": {
        "task": "safe_recovery_a1_cat",
        "seeds": {1: "Mar18_18-05-33_", 2: "Mar18_18-42-10_", 3: "Mar18_19-18-36_"},
    },
    "Recovery": {
        "task": "safe_recovery_a1_fallen",
        "seeds": {1: "Mar18_19-54-41_", 2: "Mar18_20-44-07_", 3: "Mar18_21-33-50_"},
    },
}


def evaluate(task_name, load_run, num_eval_steps=10000, num_envs=128,
             force_range=None, fallen_start=False, native_termination=False):
    """Single evaluation run."""
    argv_backup = sys.argv
    sys.argv = ["eval", "--task=%s" % task_name, "--num_envs=%d" % num_envs, "--headless"]
    args = get_args()
    sys.argv = argv_backup

    env_cfg, train_cfg = task_registry.get_cfgs(name=task_name)
    env_cfg.env.num_envs = num_envs
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False

    if force_range is not None:
        env_cfg.perturbation.enabled = force_range[1] > 0
        env_cfg.perturbation.force_range = force_range
    else:
        env_cfg.perturbation.enabled = True
        env_cfg.perturbation.force_range = [80, 150]
    env_cfg.perturbation.interval_range = [2.0, 5.0]

    # Fair eval by default; native only if explicitly requested
    if native_termination:
        env_cfg.safety.enable_constraint_termination = True
    else:
        env_cfg.safety.enable_constraint_termination = False

    # Fallen start: initialize all envs in random fallen poses
    if fallen_start:
        # Ensure fallen_start config exists (may not in base config)
        if not hasattr(env_cfg, "fallen_start"):
            class FallenStart:
                enabled = True
                fraction = 1.0
                roll_range = [-2.5, 2.5]
                pitch_range = [-1.5, 1.5]
                height_range = [0.08, 0.15]
            env_cfg.fallen_start = FallenStart()
        else:
            env_cfg.fallen_start.enabled = True
            env_cfg.fallen_start.fraction = 1.0

    env, _ = task_registry.make_env(name=task_name, args=args, env_cfg=env_cfg)
    obs = env.get_observations()

    train_cfg.runner.resume = True
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
    summary["safety/violations_per_sec"] = summary["safety/total_violations"] / (total_sim_time * num_envs) if total_sim_time > 0 else 0
    summary["recovery/falls_per_min"] = summary["recovery/fall_count"] / (total_sim_time * num_envs / 60.0) if total_sim_time > 0 else 0

    summary["meta/task"] = task_name
    summary["meta/load_run"] = str(load_run)
    summary["meta/num_eval_steps"] = num_eval_steps
    summary["meta/num_envs"] = num_envs
    summary["meta/fallen_start"] = fallen_start
    summary["meta/native_termination"] = native_termination

    env.gym.destroy_sim(env.sim)
    return summary


def save_result(result, filename):
    path = os.path.join(OUTDIR, filename)
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print("  Saved: %s" % path)


def mean_std(values):
    n = len(values)
    if n == 0: return float("nan"), float("nan")
    m = sum(values) / n
    if n == 1: return m, 0.0
    var = sum((x - m) ** 2 for x in values) / (n - 1)
    return m, math.sqrt(var)


# ====================================================================
# EXPERIMENT 1: Multi-seed standard evaluation
# ====================================================================
def run_experiment_1():
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Multi-seed standard evaluation (80-150N, 10k x 128)")
    print("=" * 70)

    all_results = {}
    for method, info in RUNS.items():
        print("\n--- %s ---" % method)
        seed_results = []
        for seed_id, run_name in sorted(info["seeds"].items()):
            print("  Seed %d (%s)..." % (seed_id, run_name))
            try:
                r = evaluate(info["task"], run_name, num_eval_steps=10000, num_envs=128,
                             force_range=[80, 150])
                save_result(r, "%s_seed%d_standard.json" % (method.lower(), seed_id))
                seed_results.append(r)
            except Exception as e:
                print("  ERROR: %s" % e)
                import traceback; traceback.print_exc()

        all_results[method] = seed_results

    return all_results


# ====================================================================
# EXPERIMENT 2: Controlled fallen-start evaluation
# ====================================================================
def run_experiment_2():
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Controlled fallen-start (100% fallen init, no perturbation)")
    print("=" * 70)

    all_results = {}
    for method, info in RUNS.items():
        print("\n--- %s ---" % method)
        seed_results = []
        for seed_id, run_name in sorted(info["seeds"].items()):
            print("  Fallen-start seed %d (%s)..." % (seed_id, run_name))
            try:
                r = evaluate(info["task"], run_name, num_eval_steps=5000, num_envs=128,
                             force_range=[0, 0], fallen_start=True)
                save_result(r, "%s_seed%d_fallen.json" % (method.lower(), seed_id))
                seed_results.append(r)
            except Exception as e:
                print("  ERROR: %s" % e)
                import traceback; traceback.print_exc()

        all_results[method] = seed_results

    return all_results


# ====================================================================
# EXPERIMENT 3: CaT native vs fair dual-protocol
# ====================================================================
def run_experiment_3():
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: CaT native (termination ON) vs fair (termination OFF)")
    print("=" * 70)

    cat_info = RUNS["CaT"]
    native_results = []
    for seed_id, run_name in sorted(cat_info["seeds"].items()):
        print("  CaT native seed %d (%s)..." % (seed_id, run_name))
        try:
            r = evaluate(cat_info["task"], run_name, num_eval_steps=10000, num_envs=128,
                         force_range=[80, 150], native_termination=True)
            save_result(r, "cat_native_seed%d.json" % seed_id)
            native_results.append(r)
        except Exception as e:
            print("  ERROR: %s" % e)
            import traceback; traceback.print_exc()

    return native_results


# ====================================================================
# AGGREGATE AND PRINT
# ====================================================================
def aggregate_and_print(exp1_results, exp2_results, exp3_results):
    print("\n" + "=" * 80)
    print("AGGREGATED MULTI-SEED RESULTS")
    print("=" * 80)

    KEY_METRICS = [
        "locomotion/mean_vel_tracking_error",
        "safety/violations_per_sec",
        "safety/torque_violation_rate",
        "recovery/fall_count",
        "recovery/falls_per_min",
        "recovery/success_rate",
        "recovery/total_attempts",
        "recovery/total_successes",
        "recovery/mean_time_to_upright",
        "coupling/viol_per_fallen_sec",
        "coupling/viol_per_active_rec_sec",
        "coupling/pct_attempts_with_violation",
    ]

    print("\n--- Standard Evaluation (80-150N) ---")
    for method, seeds in exp1_results.items():
        print("\n  %s (n=%d seeds):" % (method, len(seeds)))
        for k in KEY_METRICS:
            vals = [s.get(k, float("nan")) for s in seeds]
            vals = [v for v in vals if not (isinstance(v, float) and math.isnan(v))]
            if vals:
                m, s = mean_std(vals)
                print("    %-40s: %.4f +/- %.4f  [%s]" % (k.split("/")[-1], m, s, ", ".join("%.4f" % v for v in vals)))

    print("\n--- Controlled Fallen-Start ---")
    for method, seeds in exp2_results.items():
        print("\n  %s (n=%d seeds):" % (method, len(seeds)))
        for k in KEY_METRICS:
            vals = [s.get(k, float("nan")) for s in seeds]
            vals = [v for v in vals if not (isinstance(v, float) and math.isnan(v))]
            if vals:
                m, s = mean_std(vals)
                print("    %-40s: %.4f +/- %.4f  [%s]" % (k.split("/")[-1], m, s, ", ".join("%.4f" % v for v in vals)))

    print("\n--- CaT Native vs Fair ---")
    if exp3_results:
        print("  CaT native (termination ON) n=%d:" % len(exp3_results))
        for k in KEY_METRICS:
            vals = [s.get(k, float("nan")) for s in exp3_results]
            vals = [v for v in vals if not (isinstance(v, float) and math.isnan(v))]
            if vals:
                m, s = mean_std(vals)
                print("    %-40s: %.4f +/- %.4f" % (k.split("/")[-1], m, s))

    # Save aggregate
    aggregate = {
        "exp1_standard": {m: [s for s in seeds] for m, seeds in exp1_results.items()},
        "exp2_fallen": {m: [s for s in seeds] for m, seeds in exp2_results.items()},
        "exp3_native": exp3_results,
        "timestamp": datetime.now().isoformat(),
    }
    agg_path = os.path.join(OUTDIR, "aggregate_results.json")
    with open(agg_path, "w") as f:
        json.dump(aggregate, f, indent=2, default=str)
    print("\nAggregate saved to: %s" % agg_path)


if __name__ == "__main__":
    print("SafeRecovery Benchmark — Full Evaluation Suite v2")
    print("Started: %s" % datetime.now().isoformat())

    exp1 = run_experiment_1()
    exp2 = run_experiment_2()
    exp3 = run_experiment_3()
    aggregate_and_print(exp1, exp2, exp3)

    print("\nALL DONE: %s" % datetime.now().isoformat())
