import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from tg_stack.features import TheoryFeatures


def test_shape_and_names_consistent(schema, xy):
    X, _, _ = xy
    tf = TheoryFeatures(schema)
    Z = tf.transform(X)
    assert Z.shape == (len(X), 43)  # 26 raw + 17 theory-guided
    assert len(tf.names()) == 43
    assert tf.names()[:26] == schema.features


@pytest.mark.parametrize("inter,disp,n_cols", [(False, True, 39), (True, False, 36), (False, False, 32)])
def test_ablation_switches(schema, xy, inter, disp, n_cols):
    X, _, _ = xy
    tf = TheoryFeatures(schema, interactions=inter, dispersion=disp)
    assert tf.transform(X).shape[1] == n_cols == len(tf.names())


def test_feature_values(schema, xy):
    X, _, _ = xy
    tf = TheoryFeatures(schema)
    Z = pd.DataFrame(tf.transform(X), columns=tf.names())
    for g in schema.predictor_constructs:
        np.testing.assert_allclose(Z[g], X[schema.constructs[g]].mean(axis=1))
        np.testing.assert_allclose(Z[g + "_sd"], X[schema.constructs[g]].std(axis=1))
    np.testing.assert_allclose(Z["PUxPEU"], Z["PU"] * Z["PEU"])
    np.testing.assert_allclose(Z["IBxFSC"], Z["IB"] * Z["FSC"])
    np.testing.assert_allclose(Z["IBxTP"], Z["IB"] * Z["TP"])
    np.testing.assert_allclose(Z["SPxTP"], Z["SP"] * Z["TP"])
    np.testing.assert_allclose(Z["resp_sd"], X[schema.items].std(axis=1))


def test_stateless_numpy_input_and_clone(schema, xy):
    X, y, _ = xy
    tf = TheoryFeatures(schema)
    a = tf.fit(X, y).transform(X.values)
    b = clone(tf).transform(X)
    np.testing.assert_allclose(a, b)
    # transforming a subset gives the same rows as transforming everything (no leakage across rows)
    np.testing.assert_allclose(tf.transform(X.iloc[:10]), a[:10])
