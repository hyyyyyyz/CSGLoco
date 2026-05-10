"""Smoke test: verify the repo is wired up correctly.

Stages
------
1. pure-Python imports (no GPU required)
2. metric aggregation roundtrip on the bundled JSON results
3. (if --gpu) full Isaac Gym env construction with 4 envs / 16 steps using a random policy

The GPU stage requires `isaacgym` to be installed and a CUDA device available.
On a headless CPU machine, run without --gpu to verify packaging only.
"""
from __future__ import annotations

import sys

# Isaac Gym demands that "import isaacgym" precede "import torch", and several
# of our modules pull torch in transitively. If --gpu was requested, do that
# first; otherwise the GPU stage is skipped entirely so the early import is
# unnecessary.
if "--gpu" in sys.argv:
    import isaacgym  # noqa: F401   must precede torch

import argparse
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def stage_imports() -> None:
    print("[1/3] checking pure-Python imports...")
    import saferecovery
    # metrics is pure stdlib + numpy
    from saferecovery.eval.metrics import aggregate_seeds, load_results, format_mean_std  # noqa: F401
    # controllers depend on torch
    from saferecovery.controllers import QuadrupedMPC, ScriptedRecoveryController  # noqa: F401
    print(f"      saferecovery v{saferecovery.__version__} OK")


def stage_metrics() -> None:
    print("[2/3] aggregating bundled JSON results...")
    from saferecovery.eval.metrics import aggregate_seeds, load_results

    results_dir = REPO_ROOT / "results/eval_v2"
    pat = str(results_dir / "vanilla_seed*_standard.json")
    results = load_results(pat)
    if not results:
        print(f"      WARN: no results matched {pat} — skipping aggregation check")
        return
    agg = aggregate_seeds(results)
    rs = agg["recovery_success_pct"]
    print(f"      vanilla recovery success: {rs['mean']:.2f} ± {rs['std']:.2f}% over n={rs['n']} seeds")


def stage_gpu(num_envs: int = 4, num_steps: int = 16) -> None:
    print(f"[3/3] tiny GPU env smoke ({num_envs} envs, {num_steps} steps)...")
    import torch
    import legged_gym.envs  # noqa: F401   registers tasks via side effects on import
    from legged_gym.utils import get_args, task_registry

    if not torch.cuda.is_available():
        print("      no CUDA device — GPU stage skipped")
        return

    saved_argv = sys.argv
    sys.argv = ["smoke", "--task=safe_recovery_a1", f"--num_envs={num_envs}", "--headless"]
    try:
        args = get_args()
    finally:
        sys.argv = saved_argv

    env_cfg, _ = task_registry.get_cfgs(name="safe_recovery_a1")
    env_cfg.env.num_envs = num_envs
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.perturbation.enabled = False

    env, _ = task_registry.make_env(name="safe_recovery_a1", args=args, env_cfg=env_cfg)
    env.get_observations()
    for _ in range(num_steps):
        actions = torch.zeros((num_envs, env.num_actions), device=env.device)
        env.step(actions)
    print("      env stepped without crashing")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", action="store_true",
                   help="Also run a 4-env / 16-step Isaac Gym roundtrip (needs CUDA + isaacgym).")
    args = p.parse_args()

    stages = [stage_imports, stage_metrics]
    if args.gpu:
        stages.append(stage_gpu)

    failed = []
    for stage in stages:
        try:
            stage()
        except Exception:
            failed.append(stage.__name__)
            traceback.print_exc()

    print()
    if failed:
        print(f"FAIL: {len(failed)} stage(s) failed: {failed}")
        return 1
    print("OK: all stages passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
