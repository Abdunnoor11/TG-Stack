"""All figures (Figures 1-6). Figures whose inputs are missing are skipped with a warning."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

from ..context import Context  # noqa: E402
from ..features import TheoryFeatures  # noqa: E402

logger = logging.getLogger(__name__)

HYB, GREY, INK, MUT = "#2a78d6", "#a9a8a2", "#1f1f1f", "#52514e"
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
HN = "TG-Stack (proposed)"


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#6b6a66",
            "axes.labelcolor": "#1f1f1f",
            "xtick.color": "#52514e",
            "ytick.color": "#52514e",
        }
    )


def _need(path: Path, what: str) -> bool:
    if not path.exists():
        logger.warning("Skipping %s: %s not found (run the producing stage first).", what, path.name)
        return False
    return True


# --------------------------------------------------------------------------- Figures 2-4
def _benchmark_figures(R: Path, F: Path) -> None:
    if not (
        _need(R / "folds_main.csv", "Figures 2-4")
        and _need(R / "oof_main.npz", "Figures 2-4")
        and _need(R / "friedman_main.json", "Figures 2-4")
    ):
        return
    d = pd.read_csv(R / "folds_main.csv")
    g = d.groupby("model").AUC.agg(["mean", "std", "count"]).sort_values("mean")
    # Fig 2: AUC and MCC with 95% CI across models
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), dpi=300)
    for ax, met, lim, lab in [
        (axes[0], "AUC", (0.80, 0.935), "(a) ROC-AUC"),
        (axes[1], "MCC", (0.50, 0.73), "(b) Matthews correlation coefficient"),
    ]:
        gm = d.groupby("model")[met].agg(["mean", "std", "count"]).sort_values("mean")
        ci = 1.96 * gm["std"] / np.sqrt(gm["count"])
        cols = [HYB if m == HN else GREY for m in gm.index]
        ax.barh(gm.index, gm["mean"], color=cols, height=0.62, xerr=ci, error_kw=dict(ecolor="#444", lw=0.9, capsize=2))
        for i, (m, v) in enumerate(gm["mean"].items()):
            ax.text(
                v + ci[m] + (lim[1] - lim[0]) * 0.01,
                i,
                f"{v:.3f}",
                va="center",
                fontsize=7.2,
                color=INK,
                fontweight="bold" if m == HN else "normal",
            )
        ax.set_xlim(*lim)
        ax.set_xlabel(f"Mean {met} over 25 outer folds (95% CI)")
        ax.set_title(lab, loc="left", fontsize=9.5)
        ax.grid(axis="x", color="#e6e5e0", lw=0.6)
        ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(F / "fig2_auc_models.png", facecolor="white")
    plt.close()

    # Fig 3: ROC curves (OOF probabilities averaged over repeats) for the top-4 baselines + TG-Stack
    z = np.load(R / "oof_main.npz")
    y = z["y"]
    top = [m for m in g.sort_values("mean", ascending=False).index if m != HN][:4]
    fig, ax = plt.subplots(figsize=(4.8, 4.4), dpi=300)
    ax.plot([0, 1], [0, 1], ls="--", color="#c3c2b7", lw=1)
    for i, m in enumerate(top):
        p = z[m.replace(" ", "_")].mean(0)
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, lw=1.4, color=PAL[i + 1], label=f"{m} (AUC={roc_auc_score(y, p):.3f})")
    p = z[HN.replace(" ", "_")].mean(0)
    fpr, tpr, _ = roc_curve(y, p)
    ax.plot(fpr, tpr, lw=2.2, color=HYB, label=f"TG-Stack (AUC={roc_auc_score(y, p):.3f})")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(frameon=False, fontsize=7.2, loc="lower right")
    plt.tight_layout()
    plt.savefig(F / "fig3_roc.png", facecolor="white")
    plt.close()

    # Fig 4: critical-difference style mean-rank plot
    with open(R / "friedman_main.json") as fh:
        fr = json.load(fh)
    rk = pd.Series(fr["ranks"]).sort_values()
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=300)
    cols = [HYB if m == HN else GREY for m in rk.index]
    ax.scatter(rk.values, range(len(rk)), color=cols, s=36, zorder=3)
    best = rk.iloc[0]
    ax.axvspan(best, best + fr["CD"], color="#e8f0fb", zorder=0)
    ax.set_yticks(range(len(rk)))
    ax.set_yticklabels(rk.index)
    ax.invert_yaxis()
    ax.set_xlabel("Mean rank by AUC across 25 outer folds (lower is better)")
    ax.set_title(
        f"Friedman chi2 = {fr['friedman_chi2']:.1f}, p = {fr['p']:.1e}; "
        f"shaded band = Nemenyi CD ({fr['CD']:.2f}) from best",
        fontsize=8.5,
        color=MUT,
        loc="left",
    )
    ax.grid(axis="x", color="#e6e5e0", lw=0.6)
    plt.tight_layout()
    plt.savefig(F / "fig4_ranks.png", facecolor="white")
    plt.close()


# --------------------------------------------------------------------------- Figure 5
def _ablation_figure(R: Path, F: Path) -> None:
    if not _need(R / "folds_ablation.csv", "Figure 5"):
        return
    a = pd.read_csv(R / "folds_ablation.csv")
    order = a.groupby("model").MCC.mean().sort_values().index
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), dpi=300, sharey=True)
    for ax, met, lim, lab in [
        (axes[0], "AUC", (0.86, 0.925), "(a) ROC-AUC"),
        (axes[1], "MCC", (0.58, 0.72), "(b) MCC"),
    ]:
        ga = a.groupby("model")[met].agg(["mean", "std", "count"]).loc[order]
        cia = 1.96 * ga["std"] / np.sqrt(ga["count"])
        cols = [HYB if m == "TG-Stack (full)" else ("#9085e9" if m.startswith("Single") else GREY) for m in ga.index]
        ax.barh(
            ga.index, ga["mean"], color=cols, height=0.62, xerr=cia, error_kw=dict(ecolor="#444", lw=0.9, capsize=2)
        )
        for i, (m, v) in enumerate(ga["mean"].items()):
            ax.text(v + cia[m] + (lim[1] - lim[0]) * 0.01, i, f"{v:.3f}", va="center", fontsize=7.2)
        ax.set_xlim(*lim)
        ax.set_xlabel(f"Mean {met} (95% CI)")
        ax.set_title(lab, loc="left", fontsize=9.5)
        ax.grid(axis="x", color="#e6e5e0", lw=0.6)
        ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(F / "fig5_ablation.png", facecolor="white")
    plt.close()


# --------------------------------------------------------------------------- Figure 6
def _explain_figure(R: Path, F: Path) -> None:
    if not (_need(R / "perm_importance.csv", "Figure 6") and _need(R / "shap_cat.npz", "Figure 6")):
        return
    import shap

    pi = pd.read_csv(R / "perm_importance.csv")
    gp = pi.groupby("block").drop_auc.agg(["mean", "std"]).sort_values("mean")
    s = np.load(R / "shap_cat.npz", allow_pickle=True)
    fig = plt.figure(figsize=(11, 4.6), dpi=300)
    ax = fig.add_subplot(1, 2, 1)
    ax.barh(
        gp.index, gp["mean"], xerr=gp["std"], color=HYB, height=0.6, error_kw=dict(ecolor="#444", lw=0.9, capsize=2)
    )
    ax.set_xlabel("Drop in held-out AUC when construct block is permuted")
    ax.set_title("(a) Construct-level importance of TG-Stack", loc="left", fontsize=9.5)
    ax.grid(axis="x", color="#e6e5e0", lw=0.6)
    ax.set_axisbelow(True)
    ax2 = fig.add_subplot(1, 2, 2)
    plt.sca(ax2)
    shap.summary_plot(
        s["sv"], s["X"], feature_names=list(s["names"]), max_display=15, show=False, plot_size=None, color_bar=True
    )
    ax2.set_title("(b) SHAP values, CatBoost member (top 15)", loc="left", fontsize=9.5)
    plt.tight_layout()
    plt.savefig(F / "fig6_explain.png", facecolor="white")
    plt.close()


# --------------------------------------------------------------------------- Figure 1
def _overview_figure(ctx: Context, F: Path) -> None:
    thumbs = ["fig2_auc_models.png", "fig4_ranks.png", "fig5_ablation.png", "fig6_explain.png"]
    if not all((F / t).exists() for t in thumbs):
        logger.warning(
            "Skipping Figure 1: it embeds Figures 2, 4, 5 and 6 as thumbnails; run stages "
            "main, ablation, explain and figures first."
        )
        return
    from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch
    from PIL import Image

    # ---- numbers drawn in the figure come from the data (not hard-coded)
    n = ctx.n
    pos = int(ctx.y.sum())
    sl = int((ctx.raw[ctx.schema.all_items].nunique(axis=1) == 1).sum())
    dup = int(ctx.raw.duplicated().sum())
    ngr = int(ctx.groups.max() + 1)
    n_feat = len(ctx.schema.features)
    n_tfe = len(TheoryFeatures(ctx.schema).names())
    n_items = len(ctx.schema.items)
    r2_aub = None
    sp = ctx.out / "sem_structural.csv"
    if sp.exists():
        st_df = pd.read_csv(sp)
        r2_aub = float(st_df[st_df.Path.str.endswith("-> AUB")].R2_dv.iloc[0])
    rate = pos / n
    rate_txt = (
        f"Balanced: {100 * rate:.1f}% positive" if abs(rate - 0.5) <= 0.05 else f"Positive rate: {100 * rate:.1f}%"
    )

    FD, FIG1 = F, F / "fig1_overview.png"
    plt.rcParams.update({"font.family": "DejaVu Serif"})
    W, H = 140, 101
    fig = plt.figure(figsize=(W / 10, H / 10), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    INKC = "#1d1d1b"
    BAND = [
        ("#eef7f4", "#167a66", ["#cfeee5", "#a9dfcf", "#e6f5f0"]),
        ("#eef2fb", "#2c55c4", ["#d6e1fa", "#b9cdf6", "#e9effc"]),
        ("#fbf4ea", "#b9700f", ["#f8e2c2", "#f2cf9b", "#fbeedb"]),
    ]

    def rbox(x, y, w, h, fc, ec, lw=1.6, ls="-", r=1.6, z=1):
        ax.add_patch(
            FancyBboxPatch(
                (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, ls=ls, zorder=z
            )
        )

    def chip(x, y, w, h, txt, fc, ec, fs=12.5, bold=False, color=INKC):
        rbox(x, y, w, h, fc, ec, lw=1.2, r=0.8, z=3)
        ax.text(
            x + w / 2,
            y + h / 2,
            txt,
            ha="center",
            va="center",
            fontsize=fs,
            color=color,
            fontweight="bold" if bold else "normal",
            zorder=4,
        )

    def label(x, y, txt, fs=13.5):
        ax.text(x, y, txt, ha="center", va="center", fontsize=fs, fontweight="bold", color=INKC, zorder=4)

    def arrow(x1, y1, x2, y2, c="#3a3a38", lw=2.2, ms=20, rad=0):
        ax.add_patch(
            FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                arrowstyle="-|>",
                mutation_scale=ms,
                lw=lw,
                color=c,
                connectionstyle=f"arc3,rad={rad}",
                zorder=5,
            )
        )

    def thumb(path, crop, x, y, w, h, ec="#9a9a96"):
        im = Image.open(path).convert("RGB")
        if crop:
            im = im.crop(crop)
        iw, ih = im.size
        s = min(w / iw, h / ih)
        tw, th = iw * s, ih * s
        x0, y0 = x + (w - tw) / 2, y + (h - th) / 2
        ax.imshow(np.asarray(im), extent=[x0, x0 + tw, y0, y0 + th], zorder=3, interpolation="lanczos")
        ax.add_patch(plt.Rectangle((x0, y0), tw, th, fill=False, ec=ec, lw=0.8, zorder=4))

    def band(i, y, h, side):
        fc, ec, _ = BAND[i]
        rbox(1.5, y, W - 3, h, fc, ec, lw=2.2, r=2.5, z=0)
        rbox(3, y + 1.5, 7.5, h - 3, ec, ec, lw=0, r=1.2, z=1)
        ax.text(
            6.75,
            y + h / 2,
            side,
            rotation=90,
            ha="center",
            va="center",
            fontsize=15.5,
            fontweight="bold",
            color="white",
            zorder=2,
            linespacing=1.25,
        )

    def sub(x, y, w, h, ec):
        rbox(x, y, w, h, "white", ec, lw=1.6, ls=(0, (4, 2.5)), r=1.4, z=1)

    # ===== BAND 1: data acquisition and quality audit
    y1, h1 = 69, 30
    band(0, y1, h1, "Data Acquisition\n& Quality Audit")
    _, e1, c1 = BAND[0]
    sub(13, y1 + 6, 27, h1 - 8, e1)
    d = pd.read_csv(ctx.data_path, encoding="utf-8-sig")
    cols = ["PU1", "PEU1", "FSC1", "SP1", "TP1", "IB1", "AUB1"]
    tab = d[cols].head(6).values
    tx, ty, cw, chh = 14.1, y1 + 11.4, 3.6, 2.1
    for j, c in enumerate(cols):
        ax.add_patch(plt.Rectangle((tx + j * cw, ty + 6 * chh), cw, chh, fc=e1, ec="white", lw=0.8, zorder=3))
        ax.text(
            tx + j * cw + cw / 2,
            ty + 6 * chh + chh / 2,
            c,
            ha="center",
            va="center",
            fontsize=7.6,
            color="white",
            fontweight="bold",
            zorder=4,
        )
        for r in range(6):
            ax.add_patch(
                plt.Rectangle(
                    (tx + j * cw, ty + (5 - r) * chh),
                    cw,
                    chh,
                    fc=c1[r % 2 + 1] if r % 2 else "white",
                    ec="#b9c9c3",
                    lw=0.6,
                    zorder=3,
                )
            )
            ax.text(
                tx + j * cw + cw / 2,
                ty + (5 - r) * chh + chh / 2,
                str(tab[r, j]),
                ha="center",
                va="center",
                fontsize=9.5,
                color=INKC,
                zorder=4,
            )
    ax.text(26.5, y1 + 8.9, f"{n} students · Likert 1–5", ha="center", va="center", fontsize=11.5, color=INKC, zorder=4)
    label(26.5, y1 + 3.2, "Data in Brief Dataset")
    arrow(40.6, y1 + 17, 44.2, y1 + 17)

    sub(44.5, y1 + 6, 28, h1 - 8, e1)
    cons = [
        ("PU", "Usefulness"),
        ("PEU", "Ease of use"),
        ("FSC", "Facilitating"),
        ("SP", "Social presence"),
        ("TP", "Trust"),
        ("IB", "Intention"),
    ]
    for k, (a, _b) in enumerate(cons):
        cx = 46.0 + (k % 2) * 13.0
        cy = y1 + 21.4 - (k // 2) * 4.3
        chip(cx, cy, 12.2, 3.4, a, c1[k % 2], e1, fs=13, bold=True)
    chip(46.0, y1 + 7.0, 25.2, 3.6, "AUB ≥ 4  →  High use", e1, e1, fs=13, bold=True, color="white")
    label(58.5, y1 + 3.2, "TAM / UTAUT / S-O-R Constructs")
    arrow(73.1, y1 + 17, 76.7, y1 + 17)

    sub(77, y1 + 6, 30, h1 - 8, e1)
    items = [("Straight-liners", sl, "#e8a33d"), ("Exact duplicates", dup, "#d9673b"), ("Unique groups", ngr, e1)]
    for k, (t, v, col) in enumerate(items):
        yy = y1 + 21.0 - k * 4.6
        ax.text(78.5, yy + 1.2, t, ha="left", va="center", fontsize=12, color=INKC, zorder=4)
        ax.add_patch(plt.Rectangle((92.5, yy), 8.5 * v / n, 2.4, fc=col, ec="none", zorder=3))
        ax.text(
            92.5 + 8.5 * v / n + 0.6,
            yy + 1.2,
            str(v),
            ha="left",
            va="center",
            fontsize=12,
            fontweight="bold",
            color=INKC,
            zorder=4,
        )
    chip(78.5, y1 + 7.0, 27, 3.6, "Duplicate-aware grouping", c1[0], e1, fs=12.5, bold=True)
    label(92, y1 + 3.2, "Data-Quality Audit")
    arrow(107.6, y1 + 17, 111.2, y1 + 17)

    sub(111.5, y1 + 6, 25.5, h1 - 8, e1)
    ax.add_patch(plt.Rectangle((113.5, y1 + 18.5), 10.6, 4.6, fc="#a9dfcf", ec=e1, lw=1, zorder=3))
    ax.add_patch(plt.Rectangle((124.1, y1 + 18.5), 10.9, 4.6, fc="#f2cf9b", ec="#b9700f", lw=1, zorder=3))
    ax.text(118.8, y1 + 20.8, f"High\n{pos}", ha="center", va="center", fontsize=11.5, fontweight="bold", zorder=4)
    ax.text(
        129.55, y1 + 20.8, f"Other\n{n - pos}", ha="center", va="center", fontsize=11.5, fontweight="bold", zorder=4
    )
    ax.text(124.25, y1 + 15.6, rate_txt, ha="center", va="center", fontsize=11, zorder=4)
    chip(113.5, y1 + 7.0, 21.5, 3.6, f"{n_feat} → {n_tfe} features", c1[0], e1, fs=12.5, bold=True)
    ax.text(
        124.25,
        y1 + 12.6,
        f"{n_items} items + {n_feat - n_items} demographics",
        ha="center",
        va="center",
        fontsize=10.5,
        zorder=4,
    )
    label(124.25, y1 + 3.2, "Target & Inputs")

    # ===== BAND 2: theory-guided hybrid modelling
    y2, h2 = 35.5, 30
    band(1, y2, h2, "Theory-Guided\nHybrid Modelling")
    _, e2, c2 = BAND[1]
    sub(13, y2 + 6, 31, h2 - 8, e2)
    nodes = {
        "PU": (17.5, y2 + 22.5),
        "PEU": (17.5, y2 + 18.0),
        "TP": (17.5, y2 + 13.5),
        "SP": (17.5, y2 + 9.0),
        "FSC": (28.5, y2 + 9.0),
        "IB": (29.0, y2 + 18.0),
        "AUB": (39.5, y2 + 15.0),
    }
    for nm, (x, y) in nodes.items():
        ax.add_patch(
            Ellipse(
                (x, y),
                7.2 if nm != "AUB" else 7.6,
                3.4,
                fc=e2 if nm in ("IB", "AUB") else c2[0],
                ec=e2,
                lw=1.2,
                zorder=4,
            )
        )
        ax.text(
            x,
            y,
            nm,
            ha="center",
            va="center",
            fontsize=11.5,
            fontweight="bold",
            color="white" if nm in ("IB", "AUB") else INKC,
            zorder=5,
        )
    for s_ in ["PU", "PEU", "TP", "SP"]:
        arrow(nodes[s_][0] + 3.6, nodes[s_][1], nodes["IB"][0] - 3.6, nodes["IB"][1], c="#6b7aa8", lw=1.2, ms=10)
    arrow(nodes["IB"][0] + 3.6, nodes["IB"][1], nodes["AUB"][0] - 3.6, nodes["AUB"][1] + 0.6, c=e2, lw=1.8, ms=12)
    arrow(
        nodes["FSC"][0] + 3.6, nodes["FSC"][1] + 0.6, nodes["AUB"][0] - 2.8, nodes["AUB"][1] - 1.2, c=e2, lw=1.8, ms=12
    )
    if r2_aub is not None:
        ax.text(
            36.0,
            y2 + 19.6,
            f"R² = {r2_aub:.2f}",
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            color=e2,
            zorder=5,
        )
    label(28.5, y2 + 3.2, "Composite SEM (5,000 bootstraps)")
    arrow(44.6, y2 + 17, 48.2, y2 + 17)

    sub(48.5, y2 + 6, 30, h2 - 8, e2)
    feats = [
        "6 Construct composites",
        "6 Within-construct SDs",
        "PU×PEU   IB×FSC",
        "IB×TP   SP×TP",
        "Response-style index",
    ]
    for k, t in enumerate(feats):
        chip(50.0, y2 + 22.0 - k * 3.85, 27, 3.25, t, c2[k % 2], e2, fs=12.3, bold=(k in (2, 3)))
    label(63.5, y2 + 3.2, "Theory-Guided Features")
    arrow(79.1, y2 + 17, 82.7, y2 + 17)

    sub(83, y2 + 6, 54, h2 - 8, e2)
    ax.text(
        110,
        y2 + 25.6,
        "TG-Stack (proposed)",
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
        color=e2,
        zorder=4,
    )
    learners = ["Logistic Reg.", "SVM-RBF", "Extra-Trees", "CatBoost", "LightGBM", "MLP"]
    for k, t in enumerate(learners):
        cx = 85.0 + (k % 2) * 15.2
        cy = y2 + 19.6 - (k // 2) * 4.6
        chip(cx, cy, 14.2, 3.7, t, c2[k % 2], e2, fs=12.3)
    for k in range(3):
        arrow(115.2, y2 + 21.4 - k * 4.6, 118.4, y2 + 16.9, c="#6b7aa8", lw=1.1, ms=9)
    chip(118.6, y2 + 14.9, 8.6, 4.2, "OOF\nprob.", c2[1], e2, fs=11.5, bold=True)
    arrow(127.3, y2 + 17, 129.3, y2 + 17, lw=1.6, ms=12)
    chip(129.4, y2 + 14.4, 6.6, 5.2, "Meta\nLR", e2, e2, fs=11.5, bold=True, color="white")
    ax.text(
        126.5,
        y2 + 10.6,
        "→ P(high actual use)",
        ha="center",
        va="center",
        fontsize=11.5,
        fontweight="bold",
        color=INKC,
        zorder=4,
    )
    label(110, y2 + 3.2, "Heterogeneous Stacking Ensemble")

    # ===== BAND 3: validation and explainable AI
    y3, h3 = 2, 30
    band(2, y3, h3, "Validation &\nExplainable AI")
    _, e3, c3 = BAND[2]
    sub(13, y3 + 6, 27, h3 - 8, e3)
    prot = ["5×5 Repeated Group CV", "Nested tuning: 14 baselines", "Corrected t-test", "Friedman + Nemenyi"]
    for k, t in enumerate(prot):
        chip(14.5, y3 + 22.6 - k * 4.0, 24, 3.3, t, c3[k % 2], e3, fs=11.6 if k == 1 else 12.2, bold=(k == 0))
    ax.text(26.5, y3 + 8.0, "Acc · F1 · AUC · MCC · Brier", ha="center", va="center", fontsize=11.2, zorder=4)
    label(26.5, y3 + 3.2, "Rigorous Evaluation")
    arrow(40.6, y3 + 17, 44.2, y3 + 17)

    sub(44.5, y3 + 6, 32, h3 - 8, e3)
    thumb(FD / "fig2_auc_models.png", (1500, 0, 3000, 1380), 45.3, y3 + 7.0, 15.4, 20.0)
    thumb(FD / "fig4_ranks.png", None, 61.0, y3 + 7.0, 15.0, 20.0)
    label(60.5, y3 + 3.2, "Benchmark vs 14 Models")
    arrow(77.0, y3 + 17, 80.1, y3 + 17)

    sub(80.5, y3 + 6, 24.5, h3 - 8, e3)
    thumb(FD / "fig5_ablation.png", (1500, 0, 3000, 1440), 81.3, y3 + 11.0, 23.0, 16.2)
    chip(81.6, y3 + 7.0, 22.3, 3.3, "R1 cleaned · R2 threshold", c3[0], e3, fs=11.3)
    label(92.75, y3 + 3.2, "Ablation & Robustness")
    arrow(105.4, y3 + 17, 108.1, y3 + 17)

    sub(108.5, y3 + 6, 28.5, h3 - 8, e3)
    thumb(FD / "fig6_explain.png", (0, 0, 1640, 1380), 109.2, y3 + 11.0, 13.4, 16.2)
    thumb(FD / "fig6_explain.png", (1640, 0, 3300, 1380), 122.8, y3 + 11.0, 13.6, 16.2)
    chip(110.0, y3 + 7.0, 25.5, 3.3, "Triangulated with SEM paths", c3[0], e3, fs=11.3)
    label(122.75, y3 + 3.2, "Explainable AI")

    # band-to-band connectors (routed through free gaps)
    arrow(109.3, y1 + 5.5, 109.3, y2 + h2 - 1.2, c=e1, lw=2.6, ms=22)
    gy = (y3 + h3 + y2) / 2
    ax.plot([83.0, 80.8, 80.8, 26.5], [y2 + 9.0, y2 + 9.0, gy, gy], color=e2, lw=2.6, zorder=5, solid_joinstyle="round")
    arrow(26.5, gy + 0.05, 26.5, y3 + h3 - 2.0, c=e2, lw=2.6, ms=22)
    plt.savefig(FIG1, facecolor="white", dpi=300)
    plt.close()
    print("saved", FIG1)


def run(ctx: Context) -> None:
    R = ctx.out
    F = R / "figs"
    F.mkdir(parents=True, exist_ok=True)
    _style()
    _benchmark_figures(R, F)
    _ablation_figure(R, F)
    _explain_figure(R, F)
    print("figures written")
    _overview_figure(ctx, F)
