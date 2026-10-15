"""TG-Stack: a theory-guided stacking ensemble for explainable prediction of
Generation Z social-commerce use behaviour.

Public API
----------
>>> from tg_stack import Schema, TheoryFeatures, build_tg_stack
"""

from __future__ import annotations

from .features import TheoryFeatures
from .models import base_learners, baselines, build_tg_stack
from .schema import Schema
from .settings import Settings

__version__ = "1.0.0"

__all__ = [
    "Schema",
    "Settings",
    "TheoryFeatures",
    "base_learners",
    "baselines",
    "build_tg_stack",
    "__version__",
]
