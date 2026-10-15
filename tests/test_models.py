import numpy as np
import pytest
from sklearn.ensemble import StackingClassifier, VotingClassifier

from tg_stack.features import TheoryFeatures
from tg_stack.models import base_learners, baselines, build_tg_stack


def test_baseline_roster():
    bl = baselines()
    assert len(bl) == 14
    assert set(bl) == {
        "LR",
        "NB",
        "KNN",
        "DT",
        "SVM",
        "MLP",
        "RF",
        "ET",
        "AdaBoost",
        "GBM",
        "HistGB",
        "XGBoost",
        "LightGBM",
        "CatBoost",
    }
    assert all(isinstance(v, tuple) and len(v) == 2 and v[1] for v in bl.values())


def test_base_learner_roster():
    assert [n for n, _ in base_learners()] == ["lr", "svm", "et", "cat", "lgb", "mlp"]


def test_tg_stack_structure(schema):
    m = build_tg_stack(schema)
    assert isinstance(m.steps[0][1], TheoryFeatures)
    stk = m.steps[-1][1]
    assert isinstance(stk, StackingClassifier) and len(stk.estimators) == 6 and stk.cv == 5
    assert len(build_tg_stack(schema, drop="mlp").steps[-1][1].estimators) == 5
    assert isinstance(build_tg_stack(schema, meta="vote").steps[-1][1], VotingClassifier)
    assert len(build_tg_stack(schema, features=False).steps) == 1


@pytest.mark.slow
def test_tg_stack_fit_predict(schema, xy):
    X, y, _ = xy
    m = build_tg_stack(schema, threads=1).fit(X.values[:180], y[:180])
    p = m.predict_proba(X.values[180:])[:, 1]
    assert p.shape == (len(X) - 180,) and ((p >= 0) & (p <= 1)).all()
    w = m.steps[-1][1].final_estimator_.coef_[0]
    assert np.isfinite(w).all() and len(w) == 6
