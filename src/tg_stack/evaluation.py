"""Repeated stratified *group* cross-validation with optional nested tuning."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold

from .metrics import metrics, nb_ttest
from .settings import Settings

logger = logging.getLogger(__name__)

METRIC_COLUMNS = ["Accuracy", "Precision", "Recall", "F1", "AUC", "MCC", "Kappa", "Brier"]


def outer_folds(X, y, groups, repeat: int, k: int):
    """Outer split of one repetition (seed = repetition index, as in the paper)."""
    cv = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=repeat)
    return cv.split(X, y, groups)


def cv_eval(
    models: dict,
    X,
    y,
    groups,
    settings: Settings,
    out: Path,
    *,
    tune: bool = False,
    tag: str = "",
    save_oof: bool = False,
) -> pd.DataFrame:
    """Evaluate models on repeated stratified group k-fold; returns a long fold-metric table.

    ``models`` maps a name to either a zero-argument factory (fixed model) or an
    ``(estimator, search_space)`` tuple. With ``tune=True`` tuples are tuned by
    randomised search *inside every outer training fold* (nested CV, AUC criterion,
    inner stratified group k-fold).
    """
    R, K = settings.repeats, settings.k
    rec, oof = [], {m: np.zeros((R, len(y))) for m in models}
    params = {m: [] for m in models}
    t0 = time.time()
    for r in range(R):
        for f, (tr, te) in enumerate(outer_folds(X, y, groups, r, K)):
            for name, spec in models.items():
                if isinstance(spec, tuple):
                    est, grid = spec
                    if tune:
                        inner = StratifiedGroupKFold(n_splits=settings.inner_k, shuffle=True, random_state=100 + r)
                        rs = RandomizedSearchCV(
                            clone(est),
                            grid,
                            n_iter=settings.n_iter,
                            scoring="roc_auc",
                            cv=inner,
                            random_state=settings.seed,
                            n_jobs=1,
                            refit=True,
                        )
                        rs.fit(X.iloc[tr], y[tr], groups=groups[tr])
                        m = rs.best_estimator_
                        params[name].append(
                            {
                                key: (v if not isinstance(v, (np.floating, np.integer)) else v.item())
                                for key, v in rs.best_params_.items()
                            }
                        )
                    else:
                        m = clone(est).fit(X.iloc[tr], y[tr])
                else:
                    m = spec().fit(X.iloc[tr].values, y[tr])
                p = m.predict_proba(X.iloc[te] if isinstance(spec, tuple) else X.iloc[te].values)[:, 1]
                oof[name][r, te] = p
                rec.append({"model": name, "repeat": r, "fold": f, **metrics(y[te], p)})
            logger.info("[%s] repeat %d fold %d done  (%.0fs)", tag, r + 1, f + 1, time.time() - t0)
    df = pd.DataFrame(rec)
    if save_oof:
        np.savez_compressed(out / f"oof_{tag}.npz", y=y, **{k.replace(" ", "_"): v for k, v in oof.items()})
    if tune:
        with open(out / f"best_params_{tag}.json", "w") as fh:
            json.dump(params, fh, indent=1, default=str)
    return df


def summarise(df: pd.DataFrame, ref: str, n: int, k: int = 5, order_by: str = "AUC"):
    """Mean +/- SD table per model and Nadeau-Bengio tests of ``ref`` against every other model."""
    g = df.groupby("model")[METRIC_COLUMNS]
    s = g.mean().round(4).astype(str) + " ± " + g.std().round(4).astype(str)
    s["_k"] = g.mean()[order_by]
    s = s.sort_values("_k", ascending=False).drop(columns="_k")
    tests = []
    R = df[df.model == ref].sort_values(["repeat", "fold"])
    for m in df.model.unique():
        if m == ref:
            continue
        B = df[df.model == m].sort_values(["repeat", "fold"])
        row = {"model": m}
        for met in ["AUC", "Accuracy", "F1", "MCC"]:
            d, t, p = nb_ttest(R[met].values, B[met].values, n, k)
            row[f"d{met}"] = round(d, 4)
            row[f"p_{met}"] = round(p, 4)
        tests.append(row)
    return s, pd.DataFrame(tests)
