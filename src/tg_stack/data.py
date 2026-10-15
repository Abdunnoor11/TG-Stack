"""Data loading, target construction and duplicate-aware grouping."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import Schema


def load_dataset(path) -> tuple[pd.DataFrame, Schema]:
    """Read the survey CSV (UTF-8 with optional BOM) and infer its schema."""
    raw = pd.read_csv(path, encoding="utf-8-sig")
    return raw, Schema.from_columns(list(raw.columns))


def make_xy(df: pd.DataFrame, schema: Schema, thr: float = 4.0):
    """Build predictors, binary target and duplicate groups.

    Target: ``y = 1`` if the mean of the AUB items is ``>= thr`` (paper: 4.0).
    Groups: identical response vectors (all 31 variables) share one group id, so a
    duplicated respondent can never sit on both sides of a train/test split.
    """
    y = (df[schema.outcome_items].mean(axis=1) >= thr).astype(int).values
    X = df[schema.features].reset_index(drop=True)
    groups = pd.factorize(df.astype(str).agg("|".join, axis=1))[0]
    return X, y, groups


def data_quality_audit(raw: pd.DataFrame, schema: Schema, y: np.ndarray, groups: np.ndarray) -> dict:
    """Quality audit reported in Table 3 of the paper."""
    return {
        "n": int(len(y)),
        "n_features": len(schema.features),
        "positive_rate": float(y.mean()),
        "straightliners": int((raw[schema.all_items].nunique(axis=1) == 1).sum()),
        "exact_duplicates": int(raw.duplicated().sum()),
        "unique_groups": int(groups.max() + 1),
    }
