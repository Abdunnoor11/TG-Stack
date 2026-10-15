"""TG-Stack and the 14 baseline learners (hyper-parameters as in the paper)."""

from __future__ import annotations

import numpy as np
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from .features import TheoryFeatures
from .schema import Schema


def _scaled(model):
    return make_pipeline(StandardScaler(), model)


def base_learners(threads: int = 2, seed: int = 42) -> list:
    """The six heterogeneous level-0 learners of TG-Stack (fixed hyper-parameters)."""
    return [
        ("lr", _scaled(LogisticRegression(C=0.5, max_iter=3000))),
        ("svm", _scaled(SVC(C=1.0, probability=True, random_state=seed))),
        ("et", ExtraTreesClassifier(n_estimators=500, min_samples_leaf=2, random_state=seed, n_jobs=threads)),
        (
            "cat",
            CatBoostClassifier(
                iterations=600, learning_rate=0.03, depth=4, random_seed=seed, verbose=0, thread_count=threads
            ),
        ),
        (
            "lgb",
            LGBMClassifier(
                n_estimators=400,
                learning_rate=0.03,
                num_leaves=15,
                subsample=0.8,
                subsample_freq=1,
                colsample_bytree=0.8,
                random_state=seed,
                verbose=-1,
                n_jobs=threads,
            ),
        ),
        ("mlp", _scaled(MLPClassifier((64, 32), max_iter=1500, alpha=1e-2, early_stopping=True, random_state=seed))),
    ]


def build_tg_stack(
    schema: Schema,
    *,
    features: bool = True,
    drop: str | None = None,
    meta: str = "lr",
    interactions: bool = True,
    dispersion: bool = True,
    threads: int = 2,
    seed: int = 42,
):
    """Build the (untrained) TG-Stack pipeline.

    Parameters
    ----------
    features, interactions, dispersion:
        Switches for the theory-guided feature layer (used by the ablation study).
    drop:
        Name of one level-0 learner to leave out (leave-one-learner-out ablation).
    meta:
        ``"lr"`` = logistic-regression meta-learner on 5-fold out-of-fold
        probabilities (the proposed model); anything else = unweighted soft voting.

    Notes
    -----
    ``StackingClassifier(cv=5)`` uses scikit-learn's stratified (not group-aware)
    internal split to build the out-of-fold probabilities. This is how the paper's
    results were produced; see ``docs/LIMITATIONS.md``.
    """
    base = [b for b in base_learners(threads, seed) if b[0] != drop]
    if meta == "lr":
        ens = StackingClassifier(
            base, final_estimator=LogisticRegression(C=1.0, max_iter=3000), cv=5, stack_method="predict_proba", n_jobs=1
        )
    else:  # soft voting (unweighted average)
        ens = VotingClassifier(base, voting="soft", n_jobs=1)
    steps = [TheoryFeatures(schema, interactions, dispersion)] if features else []
    return make_pipeline(*steps, ens)


def baselines(threads: int = 2, seed: int = 42) -> dict:
    """The 14 baselines with their nested random-search spaces: name -> (estimator, grid)."""
    return {
        "LR": (_scaled(LogisticRegression(max_iter=3000)), {"logisticregression__C": np.logspace(-3, 2, 20)}),
        "NB": (GaussianNB(), {"var_smoothing": np.logspace(-11, -5, 10)}),
        "KNN": (
            _scaled(KNeighborsClassifier()),
            {
                "kneighborsclassifier__n_neighbors": list(range(3, 41, 2)),
                "kneighborsclassifier__weights": ["uniform", "distance"],
            },
        ),
        "DT": (
            DecisionTreeClassifier(random_state=seed),
            {"max_depth": [2, 3, 4, 5, 6, 8, None], "min_samples_leaf": [1, 3, 5, 10, 20]},
        ),
        "SVM": (
            _scaled(SVC(probability=True, random_state=seed)),
            {"svc__C": np.logspace(-2, 2, 15), "svc__gamma": ["scale", 0.003, 0.01, 0.03, 0.1]},
        ),
        "MLP": (
            _scaled(MLPClassifier(max_iter=1500, early_stopping=True, random_state=seed)),
            {
                "mlpclassifier__hidden_layer_sizes": [(32,), (64,), (64, 32), (128, 64)],
                "mlpclassifier__alpha": np.logspace(-4, 0, 10),
            },
        ),
        "RF": (
            RandomForestClassifier(n_estimators=400, random_state=seed, n_jobs=threads),
            {"max_depth": [3, 5, 8, None], "min_samples_leaf": [1, 2, 5, 10], "max_features": ["sqrt", 0.3, 0.5]},
        ),
        "ET": (
            ExtraTreesClassifier(n_estimators=400, random_state=seed, n_jobs=threads),
            {"max_depth": [3, 5, 8, None], "min_samples_leaf": [1, 2, 5, 10], "max_features": ["sqrt", 0.3, 0.5]},
        ),
        "AdaBoost": (
            AdaBoostClassifier(random_state=seed),
            {"n_estimators": [100, 200, 400], "learning_rate": [0.01, 0.03, 0.1, 0.3, 1.0]},
        ),
        "GBM": (
            GradientBoostingClassifier(random_state=seed),
            {
                "n_estimators": [100, 200, 400],
                "learning_rate": [0.01, 0.03, 0.1],
                "max_depth": [2, 3, 4],
                "subsample": [0.7, 1.0],
            },
        ),
        "HistGB": (
            HistGradientBoostingClassifier(random_state=seed),
            {
                "learning_rate": [0.01, 0.03, 0.1],
                "max_depth": [2, 3, 5, None],
                "l2_regularization": [0, 0.1, 1.0],
                "max_iter": [100, 300],
            },
        ),
        "XGBoost": (
            XGBClassifier(random_state=seed, verbosity=0, n_jobs=threads),
            {
                "n_estimators": [200, 400, 600],
                "learning_rate": [0.01, 0.03, 0.1],
                "max_depth": [2, 3, 4, 6],
                "subsample": [0.7, 0.85, 1.0],
                "colsample_bytree": [0.6, 0.8, 1.0],
                "reg_lambda": [0.1, 1, 10],
            },
        ),
        "LightGBM": (
            LGBMClassifier(random_state=seed, verbose=-1, n_jobs=threads, subsample_freq=1),
            {
                "n_estimators": [200, 400, 600],
                "learning_rate": [0.01, 0.03, 0.1],
                "num_leaves": [7, 15, 31],
                "subsample": [0.7, 0.85, 1.0],
                "colsample_bytree": [0.6, 0.8, 1.0],
                "min_child_samples": [5, 10, 20],
            },
        ),
        "CatBoost": (
            CatBoostClassifier(random_seed=seed, verbose=0, thread_count=threads),
            {"iterations": [300, 600], "learning_rate": [0.03, 0.1], "depth": [3, 4, 6], "l2_leaf_reg": [1, 3, 10]},
        ),
    }
