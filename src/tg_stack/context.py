"""Shared state handed to every stage."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .data import load_dataset, make_xy
from .schema import Schema
from .settings import Settings


@dataclass
class Context:
    """Data, schema, settings and output folder for one run."""

    data_path: Path
    out: Path
    settings: Settings
    raw: pd.DataFrame
    schema: Schema
    X: pd.DataFrame
    y: np.ndarray
    groups: np.ndarray

    @property
    def n(self) -> int:
        return len(self.y)

    @classmethod
    def create(cls, data_path, out, settings: Settings) -> "Context":
        raw, schema = load_dataset(data_path)
        X, y, groups = make_xy(raw, schema)
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        return cls(Path(data_path), out, settings, raw, schema, X, y, groups)
