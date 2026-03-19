"""SafeRecovery — Experiments 2 & 3 only (fallen-start + CaT native)."""
import os, json, sys, math, gc
from datetime import datetime

import isaacgym
import torch
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry

LOG_BASE = "/home/hurricane/RL/CSGLoco/legged_gym/logs"
OUTDIR = "/home/hurricane/RL/CSGLoco/safe_recovery_eval_v2"
os.makedirs(OUTDIR, exist_ok=True)

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

    if native_termination:
        env_cfg.safety.enable_constraint_termination = True
    else:
        env_cfg.safety.enable_constraint_termination = False

    if fallen_start:
        # Always set ALL attributes regardless of whether fallen_start exists
        fs = env_cfg.fallen_start if hasattr(env_cfg, "fallen_start") else type("FS", (), {})()
        fs.enabled = True
        fs.fraction = 1.0
        fs.roll_range = [-2.5, 2.5]
        fs.pitch_range = [-1.5, 1.5]
        fs.height_range = [0.08, 0.15]
        env_cfg.fallen_start = fs

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
    del env, ppo_runner, policy, obs
    gc.collect()
    torch.cuda.empty_cache()
    return summary


def save_result(result, filename):
    path = os.path.join(OUTDIR, filename)
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print("  Saved: %s" % path, flush=True)


if __name__ == "__main__":
    print("SafeRecovery — Experiments 2 & 3", flush=True)
    print("Started: %s" % datetime.now().isoformat(), flush=True)

    # ===== EXPERIMENT 2: Controlled fallen-start =====
    print("\n" + "=" * 70, flush=True)
    print("EXPERIMENT 2: Controlled fallen-start (100% fallen init, no perturbation, 5k x 128)", flush=True)
    print("=" * 70, flush=True)

    for method, info in RUNS.items():
        print("\n--- %s ---" % method, flush=True)
        for seed_id, run_name in sorted(info["seeds"].items()):
            outfile = "%s_seed%d_fallen.json" % (method.lower(), seed_id)
            if os.path.exists(os.path.join(OUTDIR, outfile)):
                print("  Skipping %s (already exists)" % outfile, flush=True)
                continue
            print("  Fallen-start seed %d (%s)..." % (seed_id, run_name), flush=True)
            try:
                r = evaluate(info["task"], run_name, num_eval_steps=5000, num_envs=128,
                             force_range=[0, 0], fallen_start=True)
                save_result(r, outfile)
            except Exception as e:
                print("  ERROR: %s" % e, flush=True)
                import traceback; traceback.print_exc()

    # ===== EXPERIMENT 3: CaT native =====
    print("\n" + "=" * 70, flush=True)
    print("EXPERIMENT 3: CaT native (termination ON, 10k x 128, 80-150N)", flush=True)
    print("=" * 70, flush=True)

    cat_info = RUNS["CaT"]
    for seed_id, run_name in sorted(cat_info["seeds"].items()):
        outfile = "cat_native_seed%d.json" % seed_id
        if os.path.exists(os.path.join(OUTDIR, outfile)):
            print("  Skipping %s (already exists)" % outfile, flush=True)
            continue
        print("  CaT native seed %d (%s)..." % (seed_id, run_name), flush=True)
        try:
            r = evaluate(cat_info["task"], run_name, num_eval_steps=10000, num_envs=128,
                         force_range=[80, 150], native_termination=True)
            save_result(r, outfile)
        except Exception as e:
            print("  ERROR: %s" % e, flush=True)
            import traceback; traceback.print_exc()

    print("\nALL DONE: %s" % datetime.now().isoformat(), flush=True)
