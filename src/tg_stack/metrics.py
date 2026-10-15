"""Performance metrics and statistical inference used in the paper."""

from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

#: Studentised-range critical values q_0.05 for the Nemenyi test (Demsar, 2006), k models.
NEMENYI_Q05 = {15: 3.391, 16: 3.426, 10: 3.164, 9: 3.102, 8: 3.031, 7: 2.949, 6: 2.850}


def metrics(y, p, thr: float = 0.5) -> dict:
    """The eight reported metrics for predicted probabilities ``p``."""
    yh = (p >= thr).astype(int)
    return {
        "Accuracy": accuracy_score(y, yh),
        "Precision": precision_score(y, yh, zero_division=0),
        "Recall": recall_score(y, yh),
        "F1": f1_score(y, yh),
        "AUC": roc_auc_score(y, p),
        "MCC": matthews_corrcoef(y, yh),
        "Kappa": cohen_kappa_score(y, yh),
        "Brier": brier_score_loss(y, p),
    }


def nb_ttest(a, b, n: int, k: int = 5):
    """Nadeau & Bengio (2003) corrected resampled t-test.

    Returns ``(mean difference, t, two-sided p)`` for paired fold-wise scores
    ``a`` and ``b`` from a k-fold scheme on ``n`` samples.
    """
    d = np.asarray(a) - np.asarray(b)
    J = len(d)
    n_te = n / k
    n_tr = n - n_te
    v = d.var(ddof=1)
    if v == 0:
        return d.mean(), 0.0, 1.0
    t = d.mean() / np.sqrt((1 / J + n_te / n_tr) * v)
    return d.mean(), t, 2 * stats.t.sf(abs(t), J - 1)


def friedman(df, met: str = "AUC"):
    """Friedman test + mean ranks + Nemenyi critical difference over fold-wise scores."""
    piv = df.pivot_table(index=["repeat", "fold"], columns="model", values=met)
    stat, p = stats.friedmanchisquare(*[piv[c].values for c in piv.columns])
    ranks = piv.rank(axis=1, ascending=False).mean().sort_values()
    k, N = piv.shape[1], piv.shape[0]
    q = NEMENYI_Q05.get(k, 3.4)  # fall back to 3.4 for unlisted k, as in the paper script
    cd = q * np.sqrt(k * (k + 1) / (6 * N))
    return stat, p, ranks, cd
