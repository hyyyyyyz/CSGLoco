# Reproducing the SafeRecovery paper

This guide walks through the exact commands needed to regenerate every table in
the paper. **Total compute: ~32 GPU-hours on an RTX 3090 Ti** (training is the
bottleneck; evaluation alone takes ~30 minutes per method per seed).

If you only want to inspect the headline numbers, the JSON outputs from the
runs reported in the paper are bundled in `results/eval_v2/`, and
`scripts/aggregate_results.py` rebuilds the tables in seconds.

## Setup

```bash
conda create -n isaacgym python=3.8
conda activate isaacgym
# Install Isaac Gym Preview 4 manually from NVIDIA
cd <isaacgym_dir>/python && pip install -e .
# Then this repo
cd <repo_root>
pip install -e .
```

## Sanity check before launching long runs

```bash
python scripts/smoke_test.py --gpu
```

Expected: `OK: all stages passed`.

## Table 4 — main results on A1

Train all three baselines for 6 seeds:

```bash
for s in 1 2 3 4 5 6; do
  python scripts/train.py --baseline vanilla  --robot a1 --seed "$s"
  python scripts/train.py --baseline cat      --robot a1 --seed "$s"
  python scripts/train.py --baseline recovery --robot a1 --seed "$s"
done
```

Evaluate each seed under the standard setting:

```bash
for s in 1 2 3 4 5 6; do
  python scripts/evaluate.py --baseline vanilla  --seed "$s" --setting standard
  python scripts/evaluate.py --baseline cat      --seed "$s" --setting standard
  python scripts/evaluate.py --baseline recovery --seed "$s" --setting standard
done
python scripts/aggregate_results.py
```

The MPC and scripted controllers are non-RL; they are evaluated through the
same interface but skip training.

## Table 5 — Axis 4 coupling

The coupling metrics come from the same JSONs produced for Table 4 — no extra
runs needed.

## Table 6 — Training-budget fairness

Train Vanilla and CaT for an extended budget (2000 iterations):

```bash
for s in 1 2 3 4 5 6; do
  python scripts/train.py --baseline vanilla --robot a1 --seed "$s" --max-iterations 2000
  python scripts/train.py --baseline cat     --robot a1 --seed "$s" --max-iterations 2000
done
```

Evaluate the matched-stage Recovery@1500 checkpoints (only seeds 1–3 retained
the 1500-iteration weights, hence n=3 in the paper):

```bash
for s in 1 2 3; do
  python scripts/evaluate.py --baseline recovery --seed "$s" --checkpoint 1500 \
    --output-name "recovery1500_seed${s}_standard.json"
done
```

## Table 7 — Rough-condition robustness

```bash
for s in 1 2 3; do
  python scripts/evaluate.py --baseline vanilla  --seed "$s" --setting rough
  python scripts/evaluate.py --baseline cat      --seed "$s" --setting rough
  python scripts/evaluate.py --baseline recovery --seed "$s" --setting rough
done
```

## Table 8 — Cross-morphology evaluation on ANYmal-C

```bash
for s in 1 2 3 4 5 6; do
  python scripts/train.py --baseline vanilla  --robot anymal --seed "$s"
  python scripts/train.py --baseline cat      --robot anymal --seed "$s"
  python scripts/train.py --baseline recovery --robot anymal --seed "$s"
  python scripts/evaluate.py --baseline vanilla  --robot anymal --seed "$s"
  python scripts/evaluate.py --baseline cat      --robot anymal --seed "$s"
  python scripts/evaluate.py --baseline recovery --robot anymal --seed "$s"
done
```

## Final aggregation

```bash
python scripts/aggregate_results.py --output paper_tables.md
```

This rebuilds Tables 4, 6, 7, and the ANYmal-C cross-morphology table from the
JSONs and writes them to `paper_tables.md`.
