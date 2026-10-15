"""Column schema of the S-COMMERCE GenZ dataset (Data in Brief, n = 757)."""

from __future__ import annotations

from dataclasses import dataclass

#: Likert constructs; the six predictors plus the outcome construct (AUB).
CONSTRUCTS = ["PU", "PEU", "FSC", "SP", "TP", "IB", "AUB"]
#: Constructs used as predictors (AUB is the outcome and is never a predictor).
PREDICTOR_CONSTRUCTS = ["PU", "PEU", "FSC", "SP", "TP", "IB"]
#: Demographic predictors ('Job' is constant in the data and therefore dropped).
DEMOGRAPHICS = ["Gender", "Income", "Area", "Frequently"]
#: Item counts reported in the paper (used only for validation warnings).
EXPECTED_ITEM_COUNTS = {"PU": 4, "PEU": 4, "FSC": 3, "SP": 4, "TP": 3, "IB": 4, "AUB": 4}


@dataclass(frozen=True)
class Schema:
    """Maps constructs to their item columns.

    The item order follows the column order of the CSV file, exactly as in the
    original reproduction script, because feature order influences tree-based
    learners and therefore the reproduced numbers.
    """

    constructs: dict  # construct name -> list of item column names

    @classmethod
    def from_columns(cls, columns) -> "Schema":
        cons = {g: [c for c in columns if c.rstrip("0123456789") == g] for g in CONSTRUCTS}
        missing = [g for g, cols in cons.items() if not cols]
        if missing:
            raise ValueError(
                f"Construct item columns not found for {missing}. Expected columns such as "
                f"'PU1', 'PEU1', ... 'AUB4'. Got: {list(columns)[:12]}..."
            )
        absent = [d for d in DEMOGRAPHICS if d not in columns]
        if absent:
            raise ValueError(f"Demographic columns not found: {absent}")
        return cls(constructs=cons)

    @property
    def predictor_constructs(self) -> list:
        return list(PREDICTOR_CONSTRUCTS)

    @property
    def demographics(self) -> list:
        return list(DEMOGRAPHICS)

    @property
    def items(self) -> list:
        """The 22 predictor Likert items."""
        return sum([self.constructs[g] for g in PREDICTOR_CONSTRUCTS], [])

    @property
    def outcome_items(self) -> list:
        return list(self.constructs["AUB"])

    @property
    def features(self) -> list:
        """The 26 raw predictors (4 demographics + 22 items)."""
        return DEMOGRAPHICS + self.items

    @property
    def all_items(self) -> list:
        """All 26 Likert items including the outcome items."""
        return sum(self.constructs.values(), [])
