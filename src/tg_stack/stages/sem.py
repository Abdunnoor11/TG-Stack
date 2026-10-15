"""Composite-based measurement and structural model (Tables 4-6)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import FactorAnalysis
from sklearn.linear_model import LinearRegression

from ..context import Context


def run(ctx: Context) -> None:
    RAW, CONS, N, RS = ctx.raw, ctx.schema.constructs, ctx.n, ctx.settings.seed
    out = ctx.out
    # ---- measurement model (composite-based, PLS-SEM-style reporting)
    Z = (RAW - RAW.mean()) / RAW.std()
    rows, comp = [], {}
    for g, cols in CONS.items():
        fa = FactorAnalysis(1, random_state=RS).fit(Z[cols])
        lam = np.abs(fa.components_[0]) / np.sqrt(np.abs(fa.components_[0]) ** 2 + fa.noise_variance_)
        k = len(cols)
        Xg = RAW[cols]
        alpha = k / (k - 1) * (1 - Xg.var().sum() / Xg.sum(axis=1).var())
        cr = lam.sum() ** 2 / (lam.sum() ** 2 + (1 - lam**2).sum())
        ave = (lam**2).mean()
        rows.append(
            {
                "Construct": g,
                "Items": k,
                "Loading range": f"{lam.min():.3f}-{lam.max():.3f}",
                "Cronbach_alpha": round(alpha, 3),
                "CR": round(cr, 3),
                "AVE": round(ave, 3),
            }
        )
        comp[g] = Z[cols].mean(axis=1)
    pd.DataFrame(rows).to_csv(out / "sem_measurement.csv", index=False)
    # ---- HTMT discriminant validity
    C = RAW[sum(CONS.values(), [])].corr().abs()
    names = list(CONS)
    ht = pd.DataFrame(index=names, columns=names, dtype=float)
    for a in names:
        for b in names:
            if a == b:
                ht.loc[a, b] = np.nan
                continue
            het = C.loc[CONS[a], CONS[b]].values.mean()
            ma = C.loc[CONS[a], CONS[a]].values[np.triu_indices(len(CONS[a]), 1)].mean()
            mb = C.loc[CONS[b], CONS[b]].values[np.triu_indices(len(CONS[b]), 1)].mean()
            ht.loc[a, b] = het / np.sqrt(ma * mb)
    ht.round(3).to_csv(out / "sem_htmt.csv")
    # ---- structural model with 5,000 bootstrap resamples
    Cdf = pd.DataFrame(comp)
    Cdf = (Cdf - Cdf.mean()) / Cdf.std()
    eqs = {"IB": ["PU", "PEU", "FSC", "SP", "TP"], "AUB": ["IB", "FSC", "TP"]}
    rng = np.random.default_rng(RS)
    B = 5000
    res = []
    for dv, ivs in eqs.items():
        lr = LinearRegression().fit(Cdf[ivs], Cdf[dv])
        r2 = lr.score(Cdf[ivs], Cdf[dv])
        boots = np.empty((B, len(ivs)))
        for b in range(B):
            idx = rng.integers(0, N, N)
            boots[b] = LinearRegression().fit(Cdf[ivs].iloc[idx], Cdf[dv].iloc[idx]).coef_
        for j, iv in enumerate(ivs):
            se = boots[:, j].std()
            t = lr.coef_[j] / se
            res.append(
                {
                    "Path": f"{iv} -> {dv}",
                    "beta": round(lr.coef_[j], 3),
                    "SE": round(se, 3),
                    "t": round(t, 2),
                    "p": round(2 * stats.norm.sf(abs(t)), 4),
                    "CI95": f"[{np.percentile(boots[:, j], 2.5):.3f}, {np.percentile(boots[:, j], 97.5):.3f}]",
                    "R2_dv": round(r2, 3),
                }
            )
    pd.DataFrame(res).to_csv(out / "sem_structural.csv", index=False)
    print(pd.DataFrame(rows))
    print(pd.DataFrame(res))
