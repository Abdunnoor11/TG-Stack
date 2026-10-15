"""Explainability: grouped permutation importance, SHAP and meta-weights (Figure 6)."""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from ..context import Context
from ..features import TheoryFeatures
from ..models import build_tg_stack

logger = logging.getLogger(__name__)


def run(ctx: Context) -> None:
    import shap  # imported lazily: heavy import, only needed here

    st, sch, X, y, groups = ctx.settings, ctx.schema, ctx.X, ctx.y, ctx.groups
    out = ctx.out
    mk = lambda: build_tg_stack(sch, threads=st.threads, seed=st.seed)  # noqa: E731
    # (a) construct-level grouped permutation importance of TG-Stack on held-out folds
    blocks = {g: sch.constructs[g] for g in sch.predictor_constructs}
    blocks["Demographics"] = sch.demographics
    rng = np.random.default_rng(st.seed)
    imp = []
    cv = StratifiedGroupKFold(n_splits=st.k, shuffle=True, random_state=0)
    for f, (tr, te) in enumerate(cv.split(X, y, groups)):
        m = mk().fit(X.iloc[tr].values, y[tr])
        Xt = X.iloc[te].copy()
        base = roc_auc_score(y[te], m.predict_proba(Xt.values)[:, 1])
        for g, cols in blocks.items():
            for _rep in range(10):
                Xp = Xt.copy()
                perm = rng.permutation(len(Xp))
                Xp[cols] = Xt[cols].values[perm]
                imp.append(
                    {"fold": f, "block": g, "drop_auc": base - roc_auc_score(y[te], m.predict_proba(Xp.values)[:, 1])}
                )
        logger.info("perm fold %d", f)
    imp = pd.DataFrame(imp)
    imp.to_csv(out / "perm_importance.csv", index=False)
    print(imp.groupby("block").drop_auc.agg(["mean", "std"]).sort_values("mean", ascending=False))
    # (b) SHAP on the CatBoost member of TG-Stack fitted on full data with theory features
    tf = TheoryFeatures(sch)
    Xf = pd.DataFrame(tf.transform(X.values), columns=tf.names())
    cb = CatBoostClassifier(
        iterations=600, learning_rate=0.03, depth=4, random_seed=st.seed, verbose=0, thread_count=st.threads
    ).fit(Xf, y)
    sv = shap.TreeExplainer(cb).shap_values(Xf)
    np.savez_compressed(out / "shap_cat.npz", sv=sv, X=Xf.values, names=np.array(Xf.columns))
    ms = pd.Series(np.abs(sv).mean(0), index=Xf.columns).sort_values(ascending=False)
    ms.to_csv(out / "shap_meanabs.csv")
    print(ms.head(15))
    # (c) meta-learner weights of the full-data TG-Stack
    full = mk().fit(X.values, y)
    stk = full.steps[-1][1]
    w = dict(zip([n for n, _ in stk.estimators], stk.final_estimator_.coef_[0].round(4)))
    with open(out / "meta_weights.json", "w") as fh:
        json.dump({k: float(v) for k, v in w.items()}, fh, indent=1)
    print("meta weights", w)
