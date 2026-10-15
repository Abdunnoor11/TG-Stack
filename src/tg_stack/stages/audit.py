"""Data-quality audit (Table 3)."""

from __future__ import annotations

import json

from ..context import Context
from ..data import data_quality_audit


def run(ctx: Context) -> None:
    aud = data_quality_audit(ctx.raw, ctx.schema, ctx.y, ctx.groups)
    demo = {c: ctx.raw[c].value_counts().sort_index().to_dict() for c in ["Gender", "Income", "Area", "Frequently"]}
    with open(ctx.out / "audit.json", "w") as fh:
        json.dump(
            {"audit": aud, "demographics": {k: {str(a): int(b) for a, b in v.items()} for k, v in demo.items()}},
            fh,
            indent=1,
        )
    print(aud)
