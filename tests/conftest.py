"""Shared fixtures. All tests run on a small *synthetic* dataset with the real schema."""

import pytest

from tg_stack.context import Context
from tg_stack.data import make_xy
from tg_stack.schema import Schema
from tg_stack.settings import Settings
from tg_stack.synthetic import make_synthetic_dataset


@pytest.fixture(scope="session")
def raw():
    return make_synthetic_dataset(n=240, seed=7)


@pytest.fixture(scope="session")
def schema(raw):
    return Schema.from_columns(list(raw.columns))


@pytest.fixture(scope="session")
def xy(raw, schema):
    return make_xy(raw, schema)


@pytest.fixture()
def csv_path(tmp_path, raw):
    p = tmp_path / "survey.csv"
    raw.to_csv(p, index=False)
    return p


@pytest.fixture()
def ctx(csv_path, tmp_path):
    return Context.create(csv_path, tmp_path / "out", Settings(repeats=1, threads=1))
