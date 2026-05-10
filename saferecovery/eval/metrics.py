"""Metric aggregation across seeds. Reproduces Tables 4–7 of the paper."""
from __future__ import annotations

import glob
import json
import math
import os
from pathlib import Path
from typing import Iterable, Mapping


def _load_one(path: str) -> dict:
    with open(path) as f:
        data = json.load(f)
    data["_filename"] = os.path.basename(path)
    return data


def load_results(pattern: str) -> list[dict]:
    """Load all JSON files matching a glob, attaching their filename."""
    out = []
    for p in sorted(glob.glob(pattern)):
        try:
            out.append(_load_one(p))
        except Exception as exc:
            print(f"[warn] could not load {p}: {exc}")
    return out


def _extract(result: Mapping) -> dict[str, float]:
    """Pull the four metrics we report in the main tables."""
    rec = result.get("recovery/success_rate", 0.0)
    if rec < 1:  # stored as fraction
        rec *= 100
    return {
        "recovery_success_pct": rec,
        "falls_per_min": result.get("recovery/falls_per_min", 0.0),
        "violations_per_sec": result.get("safety/violations_per_sec", 0.0),
        "active_rec_haz_per_sec": result.get("coupling/viol_per_active_rec_sec", 0.0),
    }


def _mean_std(values: Iterable[float]) -> tuple[float, float]:
    vals = list(values)
    if not vals:
        return 0.0, 0.0
    m = sum(vals) / len(vals)
    s = math.sqrt(sum((x - m) ** 2 for x in vals) / max(1, len(vals) - 1))
    return m, s


def aggregate_seeds(results: list[Mapping]) -> dict[str, dict[str, float]]:
    """Aggregate a list of per-seed result dicts into mean/std for each metric."""
    if not results:
        return {}
    metrics = [_extract(r) for r in results]
    keys = metrics[0].keys()
    out = {}
    for k in keys:
        m, s = _mean_std(m_[k] for m_ in metrics)
        out[k] = {"mean": m, "std": s, "n": len(metrics)}
    return out


def format_mean_std(stat: dict, fmt: str = "{mean:.2f} ± {std:.2f}") -> str:
    if not stat:
        return "—"
    return fmt.format(**stat)


# ---------------------------------------------------------------------------
# Table-builder helpers


def build_main_results_table(results_dir: str) -> str:
    """Reproduce Table 4 (A1, 6 seeds, standard setting)."""
    rd = Path(results_dir)
    methods = [
        ("Vanilla PPO",  "vanilla_seed*_standard.json"),
        ("CaT",          "cat_seed*_standard.json"),
        ("SAC-Loco",     "sac_loco_seed*_standard.json"),
        ("MPC",          "mpc_standard.json"),
        ("Recovery PPO", "recovery_seed*_standard.json"),
    ]
    rows = ["| Method | Recovery Success | Falls/min | Violations/sec |",
            "|---|---|---|---|"]
    for name, pat in methods:
        agg = aggregate_seeds(load_results(str(rd / pat)))
        if not agg:
            rows.append(f"| {name} | — | — | — |")
            continue
        rows.append(
            f"| {name} | "
            f"{format_mean_std(agg['recovery_success_pct'], '{mean:.2f} ± {std:.2f}%')} | "
            f"{format_mean_std(agg['falls_per_min'])} | "
            f"{format_mean_std(agg['violations_per_sec'])} |"
        )
    return "\n".join(rows)
