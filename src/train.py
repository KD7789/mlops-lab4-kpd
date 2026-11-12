"""Run one hyperparameter sweep and log it to MLflow.

The sweep is a parent run with one nested child run per candidate. Each child logs its
parameters, validation metrics, a validation plot, the datasets it used, and the fitted
pipeline as an MLflow model.

Usage:
    python -m src.train                                   # every model family
    python -m src.train --families ridge random_forest    # a subset
"""
import argparse
import time
import warnings

import matplotlib

matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import sklearn
import skops
from mlflow.models import infer_signature

from src import config
from src.data import features_and_target, load_data, split_by_time
from src.models import GRID, build_model, candidates, evaluate, run_name

SKOPS_TRUSTED_TYPES = [
    "sklearn.tree._tree.Tree",
    "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
]

PIP_REQUIREMENTS = [
    f"mlflow=={mlflow.__version__}",
    f"scikit-learn=={sklearn.__version__}",
    f"skops=={skops.__version__}",
    f"numpy=={np.__version__}",
    f"pandas=={pd.__version__}",
]

warnings.filterwarnings("ignore", message=".*Inferred schema contains integer column.*")
warnings.filterwarnings("ignore", message=".*dataset source can be interpreted in multiple ways.*")


def validation_plot(y_true, y_pred, hours):
    """Predicted vs actual rentals, and the mean absolute error for each hour of the day."""
    y_pred = pd.Series(y_pred, index=y_true.index)
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4))
    left.scatter(y_true, y_pred, s=4, alpha=0.3)
    top = max(y_true.max(), y_pred.max())
    left.plot([0, top], [0, top], color="black", linewidth=1)
    left.set(xlabel="actual rentals", ylabel="predicted rentals", title="Validation: predicted vs actual")
    hourly_error = (y_pred - y_true).abs().groupby(hours).mean()
    right.bar(hourly_error.index, hourly_error.values)
    right.set(xlabel="hour of day", ylabel="mean absolute error", title="Validation: error by hour")
    fig.tight_layout()
    return fig


def train_candidate(params, X_train, y_train, X_valid, y_valid, datasets):
    """Fit one candidate inside its own child run and return (run_id, val_rmse)."""
    with mlflow.start_run(run_name=run_name(params), nested=True) as run:
        mlflow.set_tags({"run_type": "candidate", "model_family": params["model"]})
        mlflow.log_params(params)
        for dataset, context in datasets:
            mlflow.log_input(dataset, context=context)

        start = time.time()
        model = build_model(params).fit(X_train, y_train)
        mlflow.log_metric("fit_seconds", time.time() - start)

        valid_pred = model.predict(X_valid)
        metrics = {f"val_{name}": value for name, value in evaluate(y_valid, valid_pred).items()}
        metrics["train_rmse"] = evaluate(y_train, model.predict(X_train))["rmse"]
        mlflow.log_metrics(metrics)

        fig = validation_plot(y_valid, valid_pred, X_valid["hr"])
        mlflow.log_figure(fig, "plots/validation.png")
        plt.close(fig)

        model_info = mlflow.sklearn.log_model(
            model,
            name="model",
            signature=infer_signature(X_valid, valid_pred),
            input_example=X_valid.head(3),
            serialization_format="skops",
            skops_trusted_types=SKOPS_TRUSTED_TYPES,
            pip_requirements=PIP_REQUIREMENTS,
        )
        mlflow.set_tag("model_uri", model_info.model_uri)
        return run.info.run_id, metrics["val_rmse"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--families", nargs="+", choices=list(GRID), default=list(GRID),
                        help="model families to include (default: all)")
    args = parser.parse_args()

    mlflow.set_tracking_uri(config.TRACKING_URI)
    experiment = mlflow.set_experiment(config.EXPERIMENT_NAME)

    train, valid, _ = split_by_time(load_data())
    X_train, y_train = features_and_target(train)
    X_valid, y_valid = features_and_target(valid)
    datasets = [
        (mlflow.data.from_pandas(train, source=str(config.DATA_PATH), targets=config.TARGET,
                                 name="bike-hour-train"), "training"),
        (mlflow.data.from_pandas(valid, source=str(config.DATA_PATH), targets=config.TARGET,
                                 name="bike-hour-valid"), "validation"),
    ]
    chosen = [params for params in candidates() if params["model"] in args.families]

    with mlflow.start_run(run_name=f"sweep: {', '.join(args.families)}") as parent:
        mlflow.set_tags({"run_type": "sweep", "families": ",".join(args.families)})
        mlflow.log_params({"train_end": config.TRAIN_END, "valid_end": config.VALID_END,
                           "n_candidates": len(chosen)})

        results = []
        for params in chosen:
            run_id, val_rmse = train_candidate(params, X_train, y_train, X_valid, y_valid, datasets)
            results.append((val_rmse, run_name(params), run_id))
            print(f"  {run_name(params):<62} val_rmse {val_rmse:6.1f}")

        best_rmse, best_name, best_id = min(results)
        mlflow.log_metric("best_val_rmse", best_rmse)
        mlflow.set_tags({"best_run_name": best_name, "best_run_id": best_id})

    print(f"\nBest candidate: {best_name} (val_rmse {best_rmse:.1f})")
    print(f"Sweep run: {config.TRACKING_URI}/#/experiments/{experiment.experiment_id}/runs/{parent.info.run_id}")


if __name__ == "__main__":
    main()
