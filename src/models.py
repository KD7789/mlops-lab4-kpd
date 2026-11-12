"""Model pipelines, the sweep grid, and the metrics used to compare candidates."""
import itertools

import numpy as np
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config

SEED = 42

GRID = {
    "ridge": {"alpha": [1.0, 10.0]},
    "random_forest": {"max_depth": [8, 14]},
    "hist_gb": {"learning_rate": [0.05, 0.1], "max_leaf_nodes": [31, 63]},
}


def candidates():
    """Yield one parameter dict per child run in the sweep."""
    for model, grid in GRID.items():
        names = list(grid)
        for values in itertools.product(*(grid[name] for name in names)):
            for log_target in (False, True):
                yield {"model": model, **dict(zip(names, values)), "log_target": log_target}


def run_name(params):
    """Readable run name, e.g. 'hist_gb learning_rate=0.1 max_leaf_nodes=63 log_target=True'."""
    settings = " ".join(f"{key}={value}" for key, value in params.items() if key != "model")
    return f"{params['model']} {settings}"


def build_preprocessor():
    """One-hot encode the categorical columns, scale the numeric ones, pass the 0/1 flags through."""
    return ColumnTransformer(
        [
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), config.CATEGORICAL),
            ("numeric", StandardScaler(), config.NUMERIC),
            ("binary", "passthrough", config.BINARY),
        ]
    )


def build_model(params):
    """Return an unfitted Pipeline (preprocessing + regressor) for one candidate."""
    if params["model"] == "ridge":
        regressor = Ridge(alpha=params["alpha"])
    elif params["model"] == "random_forest":
        regressor = RandomForestRegressor(
            n_estimators=100,
            max_depth=params["max_depth"],
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=SEED,
        )
    elif params["model"] == "hist_gb":
        regressor = HistGradientBoostingRegressor(
            learning_rate=params["learning_rate"],
            max_leaf_nodes=params["max_leaf_nodes"],
            max_iter=300,
            early_stopping=False,
            random_state=SEED,
        )
    else:
        raise ValueError(f"Unknown model: {params['model']}")

    if params["log_target"]:
        regressor = TransformedTargetRegressor(regressor=regressor, func=np.log1p, inverse_func=np.expm1)

    return Pipeline([("preprocess", build_preprocessor()), ("regressor", regressor)])


def evaluate(y_true, y_pred):
    """RMSE, MAE and R² in rental counts."""
    return {
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }
