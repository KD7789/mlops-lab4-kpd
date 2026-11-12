"""Tests for loading and splitting the Bike Sharing data."""
import pandas as pd

from src import config
from src.data import features_and_target, load_data, split_by_time


def test_load_data_drops_leaky_and_unused_columns():
    data = load_data()
    for column in ["casual", "registered", "instant", "atemp"]:
        assert column not in data.columns
    assert len(data) == 17379


def test_weather_code_4_is_merged_into_3():
    assert set(load_data()["weathersit"]) == {1, 2, 3}


def test_split_is_chronological_and_complete():
    data = load_data()
    train, valid, test = split_by_time(data)
    assert len(train) + len(valid) + len(test) == len(data)
    assert train["dteday"].max() <= pd.Timestamp(config.TRAIN_END) < valid["dteday"].min()
    assert valid["dteday"].max() <= pd.Timestamp(config.VALID_END) < test["dteday"].min()


def test_features_and_target_columns():
    X, y = features_and_target(load_data())
    assert list(X.columns) == config.FEATURES
    assert y.name == config.TARGET
