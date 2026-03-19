"""Evaluate Recovery PPO at 1500 iterations (matched-budget ablation)."""
import os, json, sys, math, gc
from datetime import datetime

import isaacgym
import torch
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry

OUTDIR = "/home/hurricane/RL/CSGLoco/safe_recovery_eval_v2"
os.makedirs(OUTDIR, exist_ok=True)

RECOVERY_RUNS = {
    1: "Mar18_19-54-41_",
    2: "Mar18_20-44-07_",
    3: "Mar18_21-33-50_",
}


def evaluate(task_name, load_run, checkpoint, num_eval_steps=10000, num_envs=128,
             force_range=None):
    argv_backup = sys.argv
    sys.argv = ["eval", "--task=%s" % task_name, "--num_envs=%d" % num_envs, "--headless"]
    args = get_args()
    sys.argv = argv_backup

    env_cfg, train_cfg = task_registry.get_cfgs(name=task_name)
    env_cfg.env.num_envs = num_envs
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.perturbation.enabled = True
    env_cfg.perturbation.force_range = force_range or [80, 150]
    env_cfg.perturbation.interval_range = [2.0, 5.0]
    env_cfg.safety.enable_constraint_termination = False

    env, _ = task_registry.make_env(name=task_name, args=args, env_cfg=env_cfg)
    obs = env.get_observations()

    train_cfg.runner.resume = True
    train_cfg.runner.load_run = load_run
    train_cfg.runner.checkpoint = checkpoint
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
    summary["meta/checkpoint"] = checkpoint
    summary["meta/num_eval_steps"] = num_eval_steps
    summary["meta/num_envs"] = num_envs

    env.gym.destroy_sim(env.sim)
    del env, ppo_runner, policy, obs
    gc.collect()
    torch.cuda.empty_cache()
    return summary


if __name__ == "__main__":
    print("Recovery@1500 matched-budget ablation evaluation", flush=True)
    print("Started: %s" % datetime.now().isoformat(), flush=True)

    for seed_id, run_name in sorted(RECOVERY_RUNS.items()):
        outfile = "recovery1500_seed%d_standard.json" % seed_id
        outpath = os.path.join(OUTDIR, outfile)
        if os.path.exists(outpath):
            print("Skipping %s (exists)" % outfile, flush=True)
            continue
        print("Recovery@1500 seed %d (%s)..." % (seed_id, run_name), flush=True)
        try:
            r = evaluate("safe_recovery_a1_fallen", run_name, checkpoint=1500,
                          num_eval_steps=10000, num_envs=128, force_range=[80, 150])
            with open(outpath, "w") as f:
                json.dump(r, f, indent=2, default=str)
            print("  Saved: %s" % outpath, flush=True)
        except Exception as e:
            print("  ERROR: %s" % e, flush=True)
            import traceback; traceback.print_exc()

    print("\nDONE: %s" % datetime.now().isoformat(), flush=True)
