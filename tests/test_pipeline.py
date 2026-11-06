import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from latedeliveries.pipeline import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_pipeline,
    split_orders,
)


def test_feature_lists():
    assert NUMERIC_FEATURES == [
        "distance_km",
        "items",
        "hour",
        "courier_months",
        "prep_minutes",
    ]
    assert CATEGORICAL_FEATURES == [
        "weather",
        "day_type",
        "vehicle",
    ]


def test_features_exclude_unavailable_columns():
    selected = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES)
    excluded = {"order_id", "customer_rating", "late"}

    assert selected.isdisjoint(excluded)


def test_split_sizes_and_no_overlap(orders):
    X_train, X_test, y_train, y_test = split_orders(orders)

    assert len(X_train) == len(y_train) == 2250
    assert len(X_test) == len(y_test) == 750
    assert set(X_train.index).isdisjoint(X_test.index)
    assert set(X_train.index) | set(X_test.index) == set(orders.index)

    pd.testing.assert_index_equal(X_train.index, y_train.index)
    pd.testing.assert_index_equal(X_test.index, y_test.index)


def test_split_is_repeatable(orders):
    first = split_orders(orders)
    second = split_orders(orders)

    for first_part, second_part in zip(first, second, strict=True):
        if isinstance(first_part, pd.DataFrame):
            pd.testing.assert_frame_equal(first_part, second_part)
        else:
            pd.testing.assert_series_equal(first_part, second_part)


@pytest.mark.parametrize("part", ["train", "test"])
def test_split_is_stratified(orders, part):
    _, _, y_train, y_test = split_orders(orders)
    target = y_train if part == "train" else y_test

    assert abs(target.mean() - orders["late"].mean()) <= 1 / len(target)


def test_split_contains_only_selected_features(orders):
    X_train, X_test, _, _ = split_orders(orders)
    expected = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    assert list(X_train.columns) == expected
    assert list(X_test.columns) == expected


def test_pipeline_handles_missing_value_and_unseen_category(orders):
    X_train, _, y_train, _ = split_orders(orders)
    pipeline = build_pipeline(LogisticRegression(max_iter=1000))
    pipeline.fit(X_train, y_train)

    new_order = X_train.iloc[[0]].copy()
    new_order["distance_km"] = np.nan
    new_order["weather"] = "hail"
    new_order["prep_minutes"] = np.nan

    probabilities = pipeline.predict_proba(new_order)

    assert probabilities.shape == (1, 2)
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
