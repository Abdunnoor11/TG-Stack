"""Theory-guided feature engineering (TAM / UTAUT / S-O-R)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .schema import Schema


class TheoryFeatures(BaseEstimator, TransformerMixin):
    """Append theory-guided features to the 26 raw predictors (26 -> 43 columns).

    * six construct composites (item means of PU, PEU, FSC, SP, TP, IB);
    * six within-construct dispersions (item standard deviations);
    * four theory-specified interactions
      (PU x PEU, IB x FSC, IB x TP, SP x TP);
    * one respondent-level response-style index (SD over all 22 predictor items).

    The transformer is stateless: it is recomputed inside every cross-validation
    fold and therefore cannot leak information.
    """

    def __init__(self, schema: Schema, interactions: bool = True, dispersion: bool = True):
        self.schema = schema
        self.interactions = interactions
        self.dispersion = dispersion

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        s = self.schema
        X = pd.DataFrame(np.asarray(X), columns=s.features).astype(float)
        for g in s.predictor_constructs:
            X[g] = X[s.constructs[g]].mean(axis=1)
            if self.dispersion:
                X[g + "_sd"] = X[s.constructs[g]].std(axis=1)
        if self.interactions:
            X["PUxPEU"] = X.PU * X.PEU  # TAM: usefulness x ease
            X["IBxFSC"] = X.IB * X.FSC  # UTAUT: intention x facilitating conditions
            X["IBxTP"] = X.IB * X.TP  # trust moderates the intention-behaviour gap
            X["SPxTP"] = X.SP * X.TP  # social presence -> trust (S-O-R)
        if self.dispersion:
            X["resp_sd"] = X[s.items].std(axis=1)
        return X.values

    def names(self) -> list:
        """Column names of the transformed matrix."""
        s = self.schema
        n = s.features + [v for g in s.predictor_constructs for v in ([g, g + "_sd"] if self.dispersion else [g])]
        if self.interactions:
            n += ["PUxPEU", "IBxFSC", "IBxTP", "SPxTP"]
        if self.dispersion:
            n += ["resp_sd"]
        return n
