"""Evaluation utilities.

The ``runner`` submodule pulls in Isaac Gym (which requires ``import isaacgym``
to happen *before* ``import torch``); to keep the rest of the package importable
without a GPU we do not re-export it here. Use ``from saferecovery.eval.runner
import evaluate_policy`` when you actually want to run an episode.
"""
from .metrics import aggregate_seeds, load_results, format_mean_std

__all__ = ["aggregate_seeds", "load_results", "format_mean_std"]
