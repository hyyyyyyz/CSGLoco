"""Unified training entry point.

Replaces train_multiseed.py, train_anymal_multiseed.py, train_vanilla_cat_2000.py,
train_sac_loco.py, batch_train*.sh, retrain_*.sh, run_*_training.sh, etc.

Examples
--------
# Train Vanilla PPO on A1 with seed 1
python scripts/train.py --baseline vanilla --robot a1 --seed 1

# Train Recovery PPO on ANYmal-C for 2000 iterations
python scripts/train.py --baseline recovery --robot anymal --seed 1 --max-iterations 2000

# Multi-seed sweep
for s in 1 2 3 4 5 6; do
    python scripts/train.py --baseline recovery --robot a1 --seed "$s"
done
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# (baseline, robot) -> task name registered by legged_gym
TASK_MAP = {
    ("vanilla",  "a1"):     "safe_recovery_a1",
    ("cat",      "a1"):     "safe_recovery_a1_cat",
    ("recovery", "a1"):     "safe_recovery_a1_fallen",
    ("vanilla",  "anymal"): "safe_recovery_anymal",
    ("cat",      "anymal"): "safe_recovery_anymal_cat",
    ("recovery", "anymal"): "safe_recovery_anymal_fallen",
}

DEFAULT_ITERS = {
    "vanilla":  1500,
    "cat":      1500,
    "recovery": 2000,
}

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--baseline", required=True, choices=["vanilla", "cat", "recovery"])
    p.add_argument("--robot", default="a1", choices=["a1", "anymal"])
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--max-iterations", type=int, default=None,
                   help="Overrides the per-baseline default if set.")
    p.add_argument("--no-headless", action="store_true",
                   help="Run with the simulator window visible (default: headless).")
    args = p.parse_args()

    task = TASK_MAP[(args.baseline, args.robot)]
    iters = args.max_iterations or DEFAULT_ITERS[args.baseline]

    cmd = [
        sys.executable,
        str(REPO_ROOT / "third_party/legged_gym/legged_gym/scripts/train.py"),
        f"--task={task}",
        f"--max_iterations={iters}",
        f"--seed={args.seed}",
    ]
    if not args.no_headless:
        cmd.append("--headless")

    print(f"[train] {args.baseline} on {args.robot}, seed={args.seed}, iters={iters}")
    print(f"[train] task={task}")
    return subprocess.run(cmd, check=False).returncode


if __name__ == "__main__":
    sys.exit(main())
