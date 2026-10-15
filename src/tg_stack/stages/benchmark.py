"""Main benchmark: TG-Stack vs. 14 nested-tuned baselines (Tables 7-8, Figures 2-4)."""

from __future__ import annotations

import json

from ..context import Context
from ..evaluation import cv_eval, summarise
from ..metrics import friedman
from ..models import baselines, build_tg_stack

REF = "TG-Stack (proposed)"


def run(ctx: Context) -> None:
    st = ctx.settings
    models = dict(baselines(st.threads, st.seed))
    models[REF] = lambda: build_tg_stack(ctx.schema, threads=st.threads, seed=st.seed)
    df = cv_eval(models, ctx.X, ctx.y, ctx.groups, st, ctx.out, tune=True, tag="main", save_oof=True)
    df.to_csv(ctx.out / "folds_main.csv", index=False)
    s, t = summarise(df, REF, ctx.n, st.k)
    s.to_csv(ctx.out / "table_main.csv")
    t.to_csv(ctx.out / "tests_main.csv", index=False)
    stat, p, ranks, cd = friedman(df)
    with open(ctx.out / "friedman_main.json", "w") as fh:
        json.dump({"friedman_chi2": stat, "p": p, "ranks": ranks.to_dict(), "CD": cd}, fh, indent=1)
    print(s.to_string())
    print(t.to_string())
    print("Friedman", stat, p, "CD", cd)
    print(ranks)
