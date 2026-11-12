"""End-to-end test of the sweep and the champion gate.

Uses a throwaway SQLite store in a temp folder, so no tracking server is needed.
"""
import sys

from mlflow import MlflowClient

from src import config, models, promote, train


def run_ridge_sweep(monkeypatch, alpha):
    """Run train.main() with a one-value Ridge grid (two candidates: raw and log target)."""
    monkeypatch.setattr(models, "GRID", {"ridge": {"alpha": [alpha]}})
    monkeypatch.setattr(sys, "argv", ["train", "--families", "ridge"])
    train.main()


def champion_version(client):
    return int(client.get_model_version_by_alias(config.MODEL_NAME, config.CHAMPION_ALIAS).version)


def test_promotion_gate(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setattr(config, "TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")

    run_ridge_sweep(monkeypatch, alpha=1.0)
    promote.main()
    client = MlflowClient()
    assert champion_version(client) == 1

    run_ridge_sweep(monkeypatch, alpha=1e5)
    promote.main()
    assert client.get_model_version(config.MODEL_NAME, "2").tags["decision"] == "rejected"
    assert champion_version(client) == 1

    promote.main()
    assert len(client.search_model_versions(f"name = '{config.MODEL_NAME}'")) == 2
