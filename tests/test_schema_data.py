import numpy as np
import pytest

from tg_stack.data import data_quality_audit, make_xy
from tg_stack.schema import EXPECTED_ITEM_COUNTS, Schema


def test_schema_item_counts(schema):
    assert {g: len(c) for g, c in schema.constructs.items()} == EXPECTED_ITEM_COUNTS
    assert len(schema.items) == 22
    assert len(schema.features) == 26
    assert len(schema.all_items) == 26
    assert "Job" not in schema.features  # constant variable is dropped
    assert not set(schema.outcome_items) & set(schema.features)  # AUB never a predictor


def test_schema_item_order_follows_csv(raw):
    shuffled = list(raw.columns[::-1])
    s = Schema.from_columns(shuffled)
    assert s.constructs["PU"] == ["PU4", "PU3", "PU2", "PU1"]


def test_schema_missing_construct_raises(raw):
    with pytest.raises(ValueError, match="not found"):
        Schema.from_columns([c for c in raw.columns if not c.startswith("TP")])


def test_make_xy_target_and_shapes(raw, schema, xy):
    X, y, groups = xy
    assert X.shape == (len(raw), 26)
    assert set(np.unique(y)) <= {0, 1}
    expected = (raw[schema.outcome_items].mean(axis=1) >= 4.0).astype(int).values
    assert (y == expected).all()
    _, y2, _ = make_xy(raw, schema, thr=3.75)
    assert y2.mean() >= y.mean()  # lower threshold -> at least as many positives


def test_duplicates_share_group_ids(raw, xy):
    _, _, groups = xy
    dup_mask = raw.duplicated(keep=False).values
    assert dup_mask.any()
    # number of groups = number of distinct rows
    assert groups.max() + 1 == len(raw.drop_duplicates())
    # every group id of a duplicated row occurs more than once
    ids, counts = np.unique(groups, return_counts=True)
    assert (counts[ids[counts > 1]] > 1).all()


def test_audit_keys(raw, schema, xy):
    _, y, groups = xy
    aud = data_quality_audit(raw, schema, y, groups)
    assert aud["n"] == len(raw)
    assert aud["n_features"] == 26
    assert aud["exact_duplicates"] == int(raw.duplicated().sum())
    assert aud["unique_groups"] == len(raw) - aud["exact_duplicates"]
    assert aud["straightliners"] > 0
