"""Run-time settings shared by every stage."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    """Evaluation protocol parameters (defaults = the values used in the paper).

    Attributes
    ----------
    seed:
        Global model / sampler / bootstrap seed (paper: 42).
    repeats, k:
        Outer repeated stratified *group* k-fold: ``repeats`` x ``k`` (paper: 5 x 5).
    inner_k, n_iter:
        Nested tuning of the 14 baselines: inner folds and random-search draws.
    threads:
        Threads per model (paper: 2).
    """

    seed: int = 42
    repeats: int = 5
    k: int = 5
    inner_k: int = 3
    n_iter: int = 10
    threads: int = 2
