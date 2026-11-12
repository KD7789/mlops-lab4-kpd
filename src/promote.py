"""Register the best candidate from the latest sweep and decide whether it becomes champion.

1. Find the most recent finished sweep and search its child runs for the lowest val_rmse.
2. Register that candidate's model as a new version of the registered model.
3. Score the new version and the current champion on the test set (Q4 2012).
4. Move the 'champion' alias to the new version only if its test RMSE is lower.

Usage:
    python -m src.promote
"""
import os

os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from src import config
from src.data import features_and_target, load_data, split_by_time
from src.models import evaluate


def find_best_candidate(client, experiment_id):
    """Return (sweep run, best child run) for the most recent finished sweep."""
    sweeps = client.search_runs(
        [experiment_id],
        filter_string="tags.run_type = 'sweep' and attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not sweeps:
        raise SystemExit("No finished sweep found. Run `python -m src.train` first.")
    sweep = sweeps[0]
    children = client.search_runs(
        [experiment_id],
        filter_string=f"tags.mlflow.parentRunId = '{sweep.info.run_id}'",
        order_by=["metrics.val_rmse ASC"],
        max_results=1,
    )
    return sweep, children[0]


def test_rmse(model_uri, X_test, y_test):
    """Load a registered model and return its RMSE on the test set."""
    model = mlflow.pyfunc.load_model(model_uri)
    return evaluate(y_test, model.predict(X_test))["rmse"]


def main():
    mlflow.set_tracking_uri(config.TRACKING_URI)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(config.EXPERIMENT_NAME)
    if experiment is None:
        raise SystemExit("No experiment found. Run `python -m src.train` first.")

    sweep, best = find_best_candidate(client, experiment.experiment_id)
    print(f"Latest sweep: {sweep.info.run_name}")
    print(f"Best candidate: {best.info.run_name} (val_rmse {best.data.metrics['val_rmse']:.1f})")

    for version in client.search_model_versions(f"name = '{config.MODEL_NAME}'"):
        if version.tags.get("source_run_id") == best.info.run_id:
            print(f"Already registered as version {version.version}; nothing to do.")
            return

    new = mlflow.register_model(best.data.tags["model_uri"], config.MODEL_NAME)
    client.set_model_version_tag(config.MODEL_NAME, new.version, "source_run_id", best.info.run_id)
    client.update_model_version(
        config.MODEL_NAME, new.version, description=f"{best.info.run_name}, from '{sweep.info.run_name}'"
    )

    _, _, test = split_by_time(load_data())
    X_test, y_test = features_and_target(test)
    new_rmse = test_rmse(f"models:/{config.MODEL_NAME}/{new.version}", X_test, y_test)

    try:
        champion = client.get_model_version_by_alias(config.MODEL_NAME, config.CHAMPION_ALIAS)
    except MlflowException:
        champion = None

    if champion is None:
        promote, compared_with = True, "no current champion"
    else:
        champion_rmse = test_rmse(f"models:/{config.MODEL_NAME}@{config.CHAMPION_ALIAS}", X_test, y_test)
        promote = new_rmse < champion_rmse
        compared_with = f"champion v{champion.version} test_rmse {champion_rmse:.1f}"

    tags = {
        "val_rmse": f"{best.data.metrics['val_rmse']:.2f}",
        "test_rmse": f"{new_rmse:.2f}",
        "decision": "promoted" if promote else "rejected",
        "compared_with": compared_with,
    }
    for key, value in tags.items():
        client.set_model_version_tag(config.MODEL_NAME, new.version, key, value)

    print(f"Registered version {new.version}: test_rmse {new_rmse:.1f} vs {compared_with}")
    if promote:
        client.set_registered_model_alias(config.MODEL_NAME, config.CHAMPION_ALIAS, new.version)
        print(f"Promoted: @{config.CHAMPION_ALIAS} now points to version {new.version}")
    else:
        print(f"Rejected: @{config.CHAMPION_ALIAS} stays on version {champion.version}")


if __name__ == "__main__":
    main()
