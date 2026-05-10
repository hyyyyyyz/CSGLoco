"""Unified evaluation entry point.

Replaces eval_vanilla_cat_2000.py, eval_anymal_multiseed.py, eval_recovery_1500.py,
eval_sac_loco.py, eval_mpc.py, run_exp1_*.py, run_exp2_*.py, run_exp3_*.py,
run_exp4_rough*.py, and the eval_*.sh family.

Examples
--------
# Standard A1 evaluation, seed 1, latest run
python scripts/evaluate.py --baseline vanilla --robot a1 --seed 1

# Rough-condition robustness for Recovery PPO seed 2
python scripts/evaluate.py --baseline recovery --seed 2 --setting rough

# Native (CaT termination preserved) — Table 5 row
python scripts/evaluate.py --baseline cat --seed 1 --setting native

# Override the run timestamp explicitly (instead of "latest")
python scripts/evaluate.py --baseline recovery --seed 1 --load-run Mar22_10-58-04_
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Task name lookup duplicated here so we don't import torch at parser time
TASK_MAP = {
    ("vanilla",  "a1"):     "safe_recovery_a1",
    ("cat",      "a1"):     "safe_recovery_a1_cat",
    ("recovery", "a1"):     "safe_recovery_a1_fallen",
    ("vanilla",  "anymal"): "safe_recovery_anymal",
    ("cat",      "anymal"): "safe_recovery_anymal_cat",
    ("recovery", "anymal"): "safe_recovery_anymal_fallen",
}

DEFAULT_CKPT = {"vanilla": 1500, "cat": 1500, "recovery": 2000}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--baseline", required=True, choices=["vanilla", "cat", "recovery"])
    p.add_argument("--robot", default="a1", choices=["a1", "anymal"])
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--checkpoint", type=int, default=None,
                   help="Iteration of the checkpoint to load (defaults to baseline default).")
    p.add_argument("--load-run", default=None,
                   help="Run timestamp dir under logs/<task>/. If omitted, uses the latest.")
    p.add_argument("--setting", default="standard", choices=["standard", "fallen", "native", "rough"])
    p.add_argument("--num-envs", type=int, default=128)
    p.add_argument("--num-eval-steps", type=int, default=10000)
    p.add_argument("--logs-root", default=str(REPO_ROOT / "third_party/legged_gym/logs"))
    p.add_argument("--output-dir", default=str(REPO_ROOT / "results/eval_v2"))
    p.add_argument("--output-name", default=None,
                   help="Output JSON filename (default: <baseline>_seed<N>_<setting>.json).")
    args = p.parse_args()

    # Defer the heavy imports (isaacgym, torch) until after argparse
    from saferecovery.eval.runner import evaluate_policy, find_latest_run

    task = TASK_MAP[(args.baseline, args.robot)]
    load_run = args.load_run or find_latest_run(task, args.logs_root)
    if load_run is None:
        print(f"[eval] no runs found under {args.logs_root}/{task}", file=sys.stderr)
        return 2
    ckpt = args.checkpoint or DEFAULT_CKPT[args.baseline]

    print(f"[eval] task={task} load_run={load_run} ckpt={ckpt} setting={args.setting}")
    summary = evaluate_policy(
        task_name=task,
        load_run=load_run,
        checkpoint_iter=ckpt,
        setting=args.setting,
        num_envs=args.num_envs,
        num_eval_steps=args.num_eval_steps,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = args.output_name or f"{args.baseline}_seed{args.seed}_{args.setting}.json"
    out_path = out_dir / out_name
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[eval] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
