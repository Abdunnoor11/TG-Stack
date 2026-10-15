import numpy as np
import pandas as pd
import pytest
from scipy import stats

from tg_stack.evaluation import METRIC_COLUMNS, outer_folds, summarise
from tg_stack.metrics import friedman, metrics, nb_ttest
from tg_stack.reproducibility import PAPER_FOLD_MD5, fold_fingerprint


def test_metrics_perfect_and_keys():
    y = np.array([0, 0, 1, 1])
    m = metrics(y, np.array([0.1, 0.2, 0.8, 0.9]))
    assert list(m) == METRIC_COLUMNS
    assert m["Accuracy"] == m["F1"] == m["AUC"] == m["MCC"] == m["Kappa"] == 1.0
    assert m["Brier"] == pytest.approx(np.mean([0.01, 0.04, 0.04, 0.01]))


def test_nb_ttest_matches_formula():
    rng = np.random.default_rng(0)
    a, b = rng.normal(0.85, 0.02, 25), rng.normal(0.84, 0.02, 25)
    d, t, p = nb_ttest(a, b, n=757, k=5)
    diff = a - b
    n_te = 757 / 5
    t_ref = diff.mean() / np.sqrt((1 / 25 + n_te / (757 - n_te)) * diff.var(ddof=1))
    assert d == pytest.approx(diff.mean())
    assert t == pytest.approx(t_ref)
    assert p == pytest.approx(2 * stats.t.sf(abs(t_ref), 24))
    # the correction makes the test more conservative than the naive paired t-test
    assert p > stats.ttest_rel(a, b).pvalue


def test_nb_ttest_zero_variance():
    assert nb_ttest([0.5] * 25, [0.5] * 25, 757) == (0.0, 0.0, 1.0)


def test_friedman_ranks_and_cd():
    rng = np.random.default_rng(1)
    rows = []
    for r in range(5):
        for f in range(5):
            for m, mu in {"A": 0.9, "B": 0.85, "C": 0.8}.items():
                rows.append({"model": m, "repeat": r, "fold": f, "AUC": mu + rng.normal(0, 0.01)})
    stat, p, ranks, cd = friedman(pd.DataFrame(rows))
    assert list(ranks.index) == ["A", "B", "C"]
    assert p < 1e-6
    assert cd == pytest.approx(3.4 * np.sqrt(3 * 4 / (6 * 25)))  # fallback q for k not in the table


def test_summarise_reference_excluded():
    rng = np.random.default_rng(2)
    rows = [
        {"model": m, "repeat": r, "fold": f, **{c: rng.uniform(0.6, 0.9) for c in METRIC_COLUMNS}}
        for m in ["ref", "x", "y"]
        for r in range(2)
        for f in range(5)
    ]
    s, t = summarise(pd.DataFrame(rows), "ref", n=300)
    assert set(t.model) == {"x", "y"} and len(s) == 3


def test_outer_folds_are_group_safe_and_stratified(xy):
    X, y, groups = xy
    for r in range(3):
        seen = []
        for tr, te in outer_folds(X, y, groups, r, 5):
            assert not set(groups[tr]) & set(groups[te]), "a duplicate group leaked across a split"
            assert abs(y[te].mean() - y.mean()) < 0.12
            seen += list(te)
        assert sorted(seen) == list(range(len(y)))  # every row is tested exactly once per repetition


def test_fold_fingerprint_is_deterministic(xy):
    X, y, groups = xy
    a = fold_fingerprint(X, y, groups)
    assert a == fold_fingerprint(X, y, groups)
    assert a != PAPER_FOLD_MD5  # synthetic data must not match the real-data fingerprint
