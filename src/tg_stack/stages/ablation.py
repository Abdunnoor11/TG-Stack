"""Ablation study with 16 variants (Table 9, Figure 5)."""

from __future__ import annotations

from functools import partial

from sklearn.base import clone
from sklearn.pipeline import make_pipeline

from ..context import Context
from ..evaluation import cv_eval, summarise
from ..features import TheoryFeatures
from ..models import base_learners, build_tg_stack

REF = "TG-Stack (full)"


def _single(sch, est):
    return make_pipeline(TheoryFeatures(sch), clone(est))


def run(ctx: Context) -> None:
    st, sch = ctx.settings, ctx.schema
    kw = {"threads": st.threads, "seed": st.seed}
    ab = {
        REF: lambda: build_tg_stack(sch, **kw),
        "w/o theory features": lambda: build_tg_stack(sch, features=False, **kw),
        "w/o interaction terms": lambda: build_tg_stack(sch, interactions=False, **kw),
        "w/o dispersion features": lambda: build_tg_stack(sch, dispersion=False, **kw),
        "Soft-voting meta": lambda: build_tg_stack(sch, meta="vote", **kw),
    }
    for b in ["lr", "svm", "et", "cat", "lgb", "mlp"]:
        ab[f"w/o {b.upper()}"] = partial(build_tg_stack, sch, drop=b, **kw)
    for name, est in base_learners(st.threads, st.seed):
        ab[f"Single {name.upper()} + TFE"] = partial(_single, sch, est)
    df = cv_eval(ab, ctx.X, ctx.y, ctx.groups, st, ctx.out, tag="ablation")
    df.to_csv(ctx.out / "folds_ablation.csv", index=False)
    s, t = summarise(df, REF, ctx.n, st.k)
    s.to_csv(ctx.out / "table_ablation.csv")
    t.to_csv(ctx.out / "tests_ablation.csv", index=False)
    print(s.to_string())
    print(t.to_string())
