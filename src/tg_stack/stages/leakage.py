"""Leakage experiment: naive vs. duplicate-aware cross-validation (Table 11)."""

from __future__ import annotations

import logging

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from ..context import Context
from ..metrics import metrics
from ..models import build_tg_stack

logger = logging.getLogger(__name__)


def run(ctx: Context) -> None:
    st, X, y, groups = ctx.settings, ctx.X, ctx.y, ctx.groups
    yc = ctx.raw[ctx.schema.outcome_items].mean(axis=1).values
    rec = []
    for scheme in ["naive", "group"]:
        for r in range(st.repeats):
            if scheme == "naive":
                splits = StratifiedKFold(st.k, shuffle=True, random_state=r).split(X, y)
            else:
                splits = StratifiedGroupKFold(st.k, shuffle=True, random_state=r).split(X, y, groups)
            for tr, te in splits:
                for name, mk in [
                    ("TG-Stack", lambda: build_tg_stack(ctx.schema, threads=st.threads, seed=st.seed)),
                    ("RF", lambda: RandomForestClassifier(n_estimators=400, random_state=st.seed, n_jobs=st.threads)),
                ]:
                    m = mk().fit(X.iloc[tr].values, y[tr])
                    p = m.predict_proba(X.iloc[te].values)[:, 1]
                    rec.append({"scheme": scheme, "model": name, **metrics(y[te], p)})
                rg = RandomForestRegressor(n_estimators=400, random_state=st.seed, n_jobs=st.threads).fit(
                    X.iloc[tr], yc[tr]
                )
                rec.append(
                    {
                        "scheme": scheme,
                        "model": "RF regression (R2 on AUB)",
                        "R2": r2_score(yc[te], rg.predict(X.iloc[te])),
                    }
                )
            logger.info("%s %d", scheme, r)
    df = pd.DataFrame(rec)
    df.to_csv(ctx.out / "folds_leak.csv", index=False)
    print(df.groupby(["model", "scheme"])[["Accuracy", "AUC", "MCC", "R2"]].agg(["mean", "std"]).round(4))
