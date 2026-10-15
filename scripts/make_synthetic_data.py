#!/usr/bin/env python3
"""Write a synthetic, schema-compatible CSV for smoke-testing the pipeline.

The file is NOT real data (see ``tg_stack.synthetic``). Usage::

    python scripts/make_synthetic_data.py data/synthetic_demo.csv --n 400
"""

import argparse

from tg_stack.synthetic import make_synthetic_dataset

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=757)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    df = make_synthetic_dataset(a.n, a.seed)
    df.to_csv(a.out, index=False)
    print(f"wrote {a.out}: {df.shape[0]} rows x {df.shape[1]} columns (SYNTHETIC)")
