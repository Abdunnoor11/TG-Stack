"""Robustness scenarios R1 (cleaned data) and R2 (alternative threshold) -- Table 10."""

from __future__ import annotations

import json

import pandas as pd

from ..context import Context
from ..data import make_xy
from ..evaluation import cv_eval, summarise
from ..models import baselines, build_tg_stack

REF = "TG-Stack (proposed)"


def run(ctx: Context) -> None:
    st, sch = ctx.settings, ctx.schema
    bl = baselines(st.threads, st.seed)
    sel = {k: bl[k] for k in ["LR", "SVM", "ET", "XGBoost", "CatBoost", "LightGBM"]}
    sel[REF] = lambda: build_tg_stack(sch, threads=st.threads, seed=st.seed)
    # R1: remove straight-liners and exact duplicates
    it = sch.all_items
    clean = ctx.raw[(ctx.raw[it].nunique(axis=1) > 1)].drop_duplicates().reset_index(drop=True)
    Xc, yc, gc = make_xy(clean, sch)
    d1 = cv_eval(sel, Xc, yc, gc, st, ctx.out, tag="robust_clean")
    d1["scenario"] = f"R1 cleaned (n={len(yc)})"
    # R2: alternative outcome threshold (AUB >= 3.75)
    X2, y2, g2 = make_xy(ctx.raw, sch, thr=3.75)
    d2 = cv_eval(sel, X2, y2, g2, st, ctx.out, tag="robust_thr")
    d2["scenario"] = "R2 threshold AUB>=3.75"
    d = pd.concat([d1, d2])
    d.to_csv(ctx.out / "folds_robust.csv", index=False)
    out = d.groupby(["scenario", "model"])[["Accuracy", "F1", "AUC", "MCC"]].agg(["mean", "std"]).round(4)
    out.to_csv(ctx.out / "table_robust.csv")
    print(out)
    tt = []
    for sc, dd in d.groupby("scenario"):
        nn = len(yc) if sc.startswith("R1") else ctx.n
        _, t = summarise(dd, REF, nn, st.k)
        t["scenario"] = sc
        tt.append(t)
    pd.concat(tt).to_csv(ctx.out / "tests_robust.csv", index=False)
    with open(ctx.out / "robust_meta.json", "w") as fh:
        json.dump({"R1_n": int(len(yc)), "R1_pos": float(yc.mean()), "R2_pos": float(y2.mean())}, fh)
