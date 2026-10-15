"""Analysis stages. Each stage is ``run(ctx: Context) -> None`` and writes into ``ctx.out``."""

from __future__ import annotations

from . import ablation, audit, benchmark, explain, figures, leakage, robustness, sem

#: Stage name -> callable, in the order used by a full run.
STAGES = {
    "audit": audit.run,  # Table 3
    "sem": sem.run,  # Tables 4-6
    "main": benchmark.run,  # Tables 7-8, Figures 2-4
    "ablation": ablation.run,  # Table 9, Figure 5
    "robust": robustness.run,  # Table 10
    "explain": explain.run,  # Figure 6
    "leak": leakage.run,  # Table 11
    "figures": figures.run,  # Figures 1-6
}

ALL_STAGES = list(STAGES)

__all__ = ["STAGES", "ALL_STAGES"]
