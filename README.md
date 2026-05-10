# SafeRecovery

A four-axis evaluation benchmark for quadruped locomotion that jointly measures
**locomotion**, **safety**, **recovery**, and the **coupling** between safety and
recovery. Implemented on Unitree A1 and ANYmal-C in Isaac Gym.

> **Paper.** "SafeRecovery: A Safety-Recovery Benchmark with Four-Axis Evaluation
> for Quadruped Locomotion", ICIC 2026.

---

## Quick start

```bash
# 1. Install in a conda env that already has isaacgym set up
pip install -e .

# 2. Smoke test (CPU only — ~10 s)
python scripts/smoke_test.py

# 3. Smoke test with GPU (verifies an Isaac Gym env can be built)
python scripts/smoke_test.py --gpu

# 4. Reproduce the main results from the bundled JSON eval outputs
python scripts/aggregate_results.py
```

Train + evaluate one baseline end-to-end:

```bash
python scripts/train.py    --baseline recovery --robot a1 --seed 1
python scripts/evaluate.py --baseline recovery --robot a1 --seed 1 --setting standard
python scripts/aggregate_results.py
```

See **[`docs/REPRODUCE.md`](docs/REPRODUCE.md)** for the full multi-seed protocol that
reproduces every table in the paper.

---

## Repository layout

```
saferecovery/        Core Python package
├── controllers/     MPC + scripted-recovery baselines
├── eval/            Unified evaluator (runner + metric aggregation)
└── utils/

scripts/             Command-line entry points (Python only — no shell scripts)
├── train.py             unified training driver
├── evaluate.py          unified evaluation driver
├── aggregate_results.py rebuild paper Tables 4-8 from JSON outputs
└── smoke_test.py        sanity check

configs/             YAML configs mirrored from the canonical Python config classes
third_party/         Vendored, lightly modified copies of legged_gym and rsl_rl
results/eval_v2/     JSON outputs from every seed in the paper (130 files)
docs/REPRODUCE.md    Step-by-step reproduction guide
```

The canonical training environment lives in
`third_party/legged_gym/legged_gym/envs/saferecovery/`; the configs in `configs/`
are human-readable references, not the source of truth.

---

## Four evaluation axes

| Axis | What it measures | Headline metric |
|---|---|---|
| 1. Locomotion | Velocity tracking under nominal conditions | `velocity_error_mps` |
| 2. Safety | Constraint violations across torque / contact / orientation | `violations_per_sec` |
| 3. Recovery | Fall detection + two-phase self-righting | `recovery_success_pct` |
| 4. **Coupling** | Safety violations conditioned on recovery state | `viol_per_active_rec_sec` |

Axis 4 is the contribution: it reveals whether safety cost is concentrated in
*deliberate recovery* or in *nominal locomotion*.

---

## Main results (A1, 6 seeds, medium perturbation 80–150 N)

| Method | Recovery Success | Falls/min | Violations/sec |
|---|---|---|---|
| Vanilla PPO     | 0.07 ± 0.05% | 8.07 ± 1.88 | 0.21 ± 0.05 |
| CaT             | 3.08 ± 2.18% | 5.75 ± 0.89 | 0.16 ± 0.02 |
| SAC-Loco        | 0.03 ± 0.05% | 10.74 ± 1.23 | **0.04 ± 0.01** |
| MPC             | 0.00 ± 0.00% | 24.84 ± 2.15 | 31.53 ± 3.21 |
| **Recovery PPO** | **57.71 ± 17.70%** | **0.14 ± 0.03** | 0.44 ± 0.13 |

Run `python scripts/aggregate_results.py` to regenerate this table from the
JSONs in `results/eval_v2/`.

---

## Requirements

- NVIDIA GPU (RTX 3090 Ti or equivalent recommended for training)
- Python 3.8+
- [Isaac Gym Preview 4](https://developer.nvidia.com/isaac-gym) (manual install)
- PyTorch 1.10+
- See `requirements.txt` for the rest

---

## Citation

```bibtex
@inproceedings{huang2026saferecovery,
  title     = {SafeRecovery: A Safety-Recovery Benchmark with Four-Axis Evaluation
               for Quadruped Locomotion},
  author    = {Huang, Yaozeng and Zhang, Ziyi and Jiang, Shuhao and Li, Jingjie
               and Wang, Yifan and Zhou, Yanyun and Shi, Fei},
  booktitle = {International Conference on Intelligent Computing (ICIC)},
  year      = {2026}
}
```

## License

Released under the MIT License — see [`LICENSE`](LICENSE). The vendored
`third_party/legged_gym` and `third_party/rsl_rl` retain their upstream licenses.
