"""Reproduce the paper's main result tables from JSON eval outputs.

Reads results/eval_v2/*.json and prints Tables 4–7 in markdown.

Examples
--------
python scripts/aggregate_results.py
python scripts/aggregate_results.py --results-dir results/eval_v2 --output table4.md
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from saferecovery.eval.metrics import (
    aggregate_seeds,
    build_main_results_table,
    format_mean_std,
    load_results,
)


def fairness_table(results_dir: Path) -> str:
    rows = ["| Method | Recovery Success | n |", "|---|---|---|"]
    for label, pat in [
        ("Vanilla-1500",   "vanilla_seed*_standard.json"),
        ("Vanilla-2000",   "vanilla_2000_seed*_standard.json"),
        ("CaT-1500",       "cat_seed*_standard.json"),
        ("CaT-2000",       "cat_2000_seed*_standard.json"),
        ("Recovery@1500",  "recovery1500_seed*_standard.json"),
    ]:
        agg = aggregate_seeds(load_results(str(results_dir / pat)))
        if not agg:
            rows.append(f"| {label} | — | 0 |")
            continue
        rs = agg["recovery_success_pct"]
        rows.append(f"| {label} | {rs['mean']:.2f} ± {rs['std']:.2f}% | {rs['n']} |")
    return "\n".join(rows)


def rough_table(results_dir: Path) -> str:
    rows = ["| Method | Recovery Success | Falls/min | Violations/sec |", "|---|---|---|---|"]
    for label, pat in [
        ("Vanilla PPO",  "vanilla_seed*_rough.json"),
        ("CaT",          "cat_seed*_rough.json"),
        ("Recovery PPO", "recovery_seed*_rough.json"),
    ]:
        agg = aggregate_seeds(load_results(str(results_dir / pat)))
        if not agg:
            rows.append(f"| {label} | — | — | — |")
            continue
        rows.append(
            f"| {label} | {format_mean_std(agg['recovery_success_pct'], '{mean:.2f} ± {std:.2f}%')} | "
            f"{format_mean_std(agg['falls_per_min'])} | "
            f"{format_mean_std(agg['violations_per_sec'])} |"
        )
    return "\n".join(rows)


def anymal_table(results_dir: Path) -> str:
    rows = ["| Method | Recovery Success | n |", "|---|---|---|"]
    for label, pat in [
        ("Vanilla PPO",  "anymal_vanilla_seed*_standard.json"),
        ("CaT",          "anymal_cat_seed*_standard.json"),
        ("Recovery PPO", "anymal_recovery_seed*_standard.json"),
    ]:
        agg = aggregate_seeds(load_results(str(results_dir / pat)))
        if not agg:
            rows.append(f"| {label} | — | 0 |")
            continue
        rs = agg["recovery_success_pct"]
        rows.append(f"| {label} | {rs['mean']:.2f} ± {rs['std']:.2f}% | {rs['n']} |")
    return "\n".join(rows)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--results-dir", default=str(REPO_ROOT / "results/eval_v2"))
    p.add_argument("--output", default=None, help="Optional path to write markdown to.")
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"[error] results dir not found: {results_dir}", file=sys.stderr)
        return 1

    out = []
    out.append("# SafeRecovery Aggregate Results\n")
    out.append("## Table 4 — Main results on Unitree A1 (6 seeds, medium perturbation)\n")
    out.append(build_main_results_table(str(results_dir)))
    out.append("\n## Table 6 — Training-budget fairness\n")
    out.append(fairness_table(results_dir))
    out.append("\n## Table 7 — Rough-condition robustness (3 seeds, friction/noise DR)\n")
    out.append(rough_table(results_dir))
    out.append("\n## Table 8 — Cross-morphology evaluation on ANYmal-C\n")
    out.append(anymal_table(results_dir))

    text = "\n".join(out) + "\n"
    if args.output:
        Path(args.output).write_text(text)
        print(f"[aggregate] wrote {args.output}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
