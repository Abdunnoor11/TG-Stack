"""Command-line interface: ``tg-stack --data <csv> --out results [--stages ...]``."""

from __future__ import annotations

import argparse
import logging
import os
import time
import warnings
from pathlib import Path

from . import __version__
from .context import Context
from .reproducibility import check_folds, check_versions
from .settings import Settings
from .stages import ALL_STAGES, STAGES


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tg-stack", description="Reproduce the TG-Stack manuscript results (tables, statistics and figures)."
    )
    p.add_argument("--data", required=True, help="path to S-COMMERCE_GenZ_UniversityStudent_757_DIB.csv")
    p.add_argument("--out", default="results", help="output directory (CSV/JSON results and figs/)")
    p.add_argument(
        "--stages",
        nargs="*",
        default=None,
        choices=ALL_STAGES,
        metavar="STAGE",
        help=f"subset of: {' '.join(ALL_STAGES)} (default: all, in that order)",
    )
    p.add_argument("--quick", action="store_true", help="1 x 5 outer folds instead of 5 x 5 (smoke test)")
    p.add_argument("--threads", type=int, default=2, help="threads per model (paper: 2)")
    p.add_argument("--no-strict", action="store_true", help="warn instead of stopping on version/fold mismatch")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    warnings.filterwarnings("ignore")  # as in the original script (library deprecation noise)
    os.environ["PYTHONWARNINGS"] = "ignore"
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    strict = not args.no_strict

    print("Environment check:")
    check_versions(strict)

    settings = Settings(repeats=1 if args.quick else 5, threads=args.threads)
    ctx = Context.create(Path(args.data), Path(args.out), settings)
    check_folds(ctx.X, ctx.y, ctx.groups, strict)

    stages = args.stages or ALL_STAGES
    t0 = time.time()
    for name in ALL_STAGES:  # fixed order, regardless of the order given on the command line
        if name in stages:
            print(f"\n========== stage: {name} ==========", flush=True)
            STAGES[name](ctx)
    print(f"\nFinished {stages} in {(time.time() - t0) / 60:.1f} min. Outputs in: {ctx.out.resolve()}")
    return 0
