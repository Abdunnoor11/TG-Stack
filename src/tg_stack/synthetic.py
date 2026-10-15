"""Synthetic stand-in with the *schema* of the real dataset (for tests and smoke runs only).

The data are random draws from a latent-factor model: they have the same columns,
scales and rough correlation structure (correlated Likert constructs, straight-liners,
duplicated rows) as the survey, but they are NOT the real data and must never be used
to report scientific results.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import EXPECTED_ITEM_COUNTS


def make_synthetic_dataset(
    n: int = 757, seed: int = 0, straightliner_frac: float = 0.08, duplicate_frac: float = 0.08
) -> pd.DataFrame:
    """Return a DataFrame with the columns of S-COMMERCE_GenZ_UniversityStudent_757_DIB.csv."""
    rng = np.random.default_rng(seed)
    base = int(round(n * (1 - duplicate_frac)))
    theta = rng.normal(size=base)  # general attitude towards s-commerce (drives common-method overlap)
    cols: dict = {}
    cols["Gender"] = rng.choice([1, 2, 3], size=base, p=[0.22, 0.77, 0.01])
    cols["Job"] = np.ones(base, dtype=int)  # constant in the real data -> dropped
    cols["Income"] = rng.choice([1, 2, 3, 4, 5], size=base, p=[0.89, 0.09, 0.015, 0.003, 0.002])
    cols["Area"] = rng.choice([1, 2, 3], size=base, p=[0.61, 0.10, 0.29])
    cols["Frequently"] = rng.choice([1, 2, 3, 4], size=base, p=[0.96, 0.02, 0.01, 0.01])

    def likert(latent: np.ndarray, noise: float) -> np.ndarray:
        raw = 3.6 + 0.8 * latent + rng.normal(scale=noise, size=latent.shape)
        return np.clip(np.rint(raw), 1, 5).astype(int)

    latent = {g: 0.8 * theta + 0.6 * rng.normal(size=base) for g in ["PU", "PEU", "FSC", "SP", "TP"]}
    latent["IB"] = 0.35 * latent["PEU"] + 0.30 * latent["TP"] + 0.15 * latent["PU"] + 0.5 * rng.normal(size=base)
    latent["AUB"] = 0.45 * latent["FSC"] + 0.30 * latent["IB"] + 0.15 * latent["TP"] + 0.4 * rng.normal(size=base)
    for g, k in EXPECTED_ITEM_COUNTS.items():
        for i in range(1, k + 1):
            cols[f"{g}{i}"] = likert(latent[g], 0.45)
    df = pd.DataFrame(cols)
    # straight-liners: same answer for every Likert item
    likert_cols = [c for c in df.columns if c.rstrip("0123456789") in EXPECTED_ITEM_COUNTS]
    idx = rng.choice(base, size=int(straightliner_frac * base), replace=False)
    df.loc[idx, likert_cols] = rng.integers(2, 6, size=(len(idx), 1))
    # exact duplicate submissions
    dup = df.sample(n - base, random_state=seed, replace=True)
    return pd.concat([df, dup], ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
