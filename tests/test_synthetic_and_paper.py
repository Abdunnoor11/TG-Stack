import os

import pytest

from tg_stack.context import Context
from tg_stack.data import data_quality_audit
from tg_stack.reproducibility import PAPER_FOLD_MD5, fold_fingerprint
from tg_stack.schema import Schema
from tg_stack.settings import Settings
from tg_stack.synthetic import make_synthetic_dataset


def test_synthetic_matches_schema_and_is_deterministic():
    a = make_synthetic_dataset(100, 3)
    assert a.shape == (100, 31)
    Schema.from_columns(list(a.columns))
    assert a.equals(make_synthetic_dataset(100, 3))
    assert a.Job.nunique() == 1  # constant, as in the real data
    assert a[[c for c in a.columns if c[:2] in ("PU", "TP")]].isin(range(1, 6)).all().all()


DATA = os.environ.get("TG_STACK_DATA")


@pytest.mark.paper
@pytest.mark.skipif(not DATA, reason="set TG_STACK_DATA to the real dataset to run the paper checks")
def test_real_data_matches_paper(tmp_path):
    ctx = Context.create(DATA, tmp_path, Settings())
    aud = data_quality_audit(ctx.raw, ctx.schema, ctx.y, ctx.groups)
    assert (aud["n"], aud["straightliners"], aud["exact_duplicates"], aud["unique_groups"]) == (757, 158, 129, 628)
    assert int(ctx.y.sum()) == 378
    assert fold_fingerprint(ctx.X, ctx.y, ctx.groups) == PAPER_FOLD_MD5
