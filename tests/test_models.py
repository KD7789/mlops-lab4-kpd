"""Tests for the model pipelines, the sweep grid and the skops trust list."""
import numpy as np
import pytest
import skops.io as sio
from sklearn.compose import TransformedTargetRegressor

from src.data import features_and_target, load_data, split_by_time
from src.models import GRID, build_model, candidates, evaluate, run_name
from src.train import SKOPS_TRUSTED_TYPES

FIRST_PARAMS = {family: next(p for p in candidates() if p["model"] == family) for family in GRID}


@pytest.fixture(scope="module")
def small_data():
    train, valid, _ = split_by_time(load_data())
    X_train, y_train = features_and_target(train.head(2000))
    X_valid, _ = features_and_target(valid.head(200))
    return X_train, y_train, X_valid


def test_candidates_cover_the_grid():
    params = list(candidates())
    expected = sum(2 * int(np.prod([len(values) for values in grid.values()])) for grid in GRID.values())
    assert len(params) == expected
    assert len({run_name(p) for p in params}) == len(params)


def test_log_target_wraps_the_regressor():
    model = build_model({"model": "ridge", "alpha": 1.0, "log_target": True})
    assert isinstance(model.named_steps["regressor"], TransformedTargetRegressor)


@pytest.mark.parametrize("family", list(GRID))
def test_each_family_fits_and_predicts(family, small_data):
    X_train, y_train, X_valid = small_data
    predictions = build_model(FIRST_PARAMS[family]).fit(X_train, y_train).predict(X_valid)
    assert predictions.shape == (len(X_valid),)
    assert np.isfinite(predictions).all()


def test_evaluate_on_perfect_predictions():
    y = np.array([10.0, 20.0, 30.0])
    assert evaluate(y, y) == {"rmse": 0.0, "mae": 0.0, "r2": 1.0}


@pytest.mark.parametrize("family", ["random_forest", "hist_gb"])
def test_skops_round_trip_needs_only_the_listed_types(family, small_data):
    X_train, y_train, X_valid = small_data
    model = build_model(FIRST_PARAMS[family]).fit(X_train, y_train)
    blob = sio.dumps(model)
    assert set(sio.get_untrusted_types(data=blob)) <= set(SKOPS_TRUSTED_TYPES)
    loaded = sio.loads(blob, trusted=SKOPS_TRUSTED_TYPES)
    np.testing.assert_allclose(loaded.predict(X_valid), model.predict(X_valid))
