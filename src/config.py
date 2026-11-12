"""Settings shared by the training, promotion, serving and dashboard code."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "hour.csv"

TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")
EXPERIMENT_NAME = "bike-demand"
MODEL_NAME = "bike-demand-model"
CHAMPION_ALIAS = "champion"

TRAIN_END = "2012-06-30"
VALID_END = "2012-09-30"

TARGET = "cnt"
CATEGORICAL = ["season", "mnth", "hr", "weekday", "weathersit"]
BINARY = ["yr", "holiday", "workingday"]
NUMERIC = ["temp", "hum", "windspeed"]
FEATURES = CATEGORICAL + BINARY + NUMERIC
