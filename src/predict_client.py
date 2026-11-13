"""Send requests to the champion model served by `mlflow models serve`.

1. A valid request with five test-set hours, printed next to the actual rentals.
2. Two bad requests (a missing column, and text in a numeric column), which the
   server rejects with HTTP 400.

Usage, with the model server running on port 5002:
    python -m src.predict_client
"""
import json
import os

import mlflow
import requests
from mlflow import MlflowClient

from src import config
from src.data import features_and_target, load_data, split_by_time

SERVING_URL = os.environ.get("SERVING_URL", "http://127.0.0.1:5002/invocations")


def post(frame):
    """Send a DataFrame to the model server in MLflow's dataframe_split format."""
    payload = {"dataframe_split": json.loads(frame.to_json(orient="split", index=False))}
    return requests.post(SERVING_URL, json=payload, timeout=30)


def predict(frame):
    """Return the served model's predictions for a DataFrame of features."""
    response = post(frame)
    response.raise_for_status()
    return response.json()["predictions"]


def main():
    mlflow.set_tracking_uri(config.TRACKING_URI)
    champion = MlflowClient().get_model_version_by_alias(config.MODEL_NAME, config.CHAMPION_ALIAS)
    print(f"Registry @{config.CHAMPION_ALIAS}: version {champion.version} ({champion.description})")

    _, _, test = split_by_time(load_data())
    sample = test.sample(5, random_state=1).sort_values(["dteday", "hr"])
    X, _ = features_and_target(sample)

    try:
        response = post(X)
    except requests.ConnectionError:
        raise SystemExit(f"No model server at {SERVING_URL}. Start it with `mlflow models serve` first.")
    response.raise_for_status()
    predictions = [round(value, 1) for value in response.json()["predictions"]]
    result = sample[["dteday", "hr", config.TARGET]].assign(predicted=predictions)
    print(f"\nValid request -> HTTP {response.status_code}")
    print(result.to_string(index=False))

    bad_requests = {
        "missing 'hr' column": X.drop(columns=["hr"]),
        "text in 'temp' column": X.assign(temp="warm"),
    }
    for label, frame in bad_requests.items():
        response = post(frame)
        message = response.json().get("message", response.text)
        print(f"\nBad request ({label}) -> HTTP {response.status_code}")
        print("  " + message.rsplit("Error: ", 1)[-1][:200])


if __name__ == "__main__":
    main()
