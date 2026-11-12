"""Streamlit dashboard for the bike-demand champion model.

- What-if: predicted rentals for conditions you choose, with the 24-hour curve.
- Real day: actual vs predicted rentals for a day in the test period (Q4 2012).
- Compare versions: the same day scored by two registered model versions.

What-if and Real day call the model server; Compare versions loads models from the
Model Registry. Needs the tracking server (port 5001) and the model server (port 5002).

Run from the repo root:
    streamlit run dashboard/app.py
"""
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  

import mlflow
import pandas as pd
import requests
import streamlit as st
from mlflow import MlflowClient

from src import config
from src.data import features_and_target, load_data
from src.models import evaluate
from src.predict_client import SERVING_URL, predict

TEMP_MIN, TEMP_MAX = -8, 39
WIND_MAX = 67
WEATHER = {1: "Clear or partly cloudy", 2: "Mist or cloudy", 3: "Rain or snow"}
TEST_START, TEST_END = date(2012, 10, 1), date(2012, 12, 31)

st.set_page_config(page_title="Bike demand", page_icon="🚲", layout="wide")
mlflow.set_tracking_uri(config.TRACKING_URI)


@st.cache_data
def get_data():
    return load_data()


@st.cache_resource
def load_version(version):
    """Load a registered model version once and keep it in memory."""
    return mlflow.pyfunc.load_model(f"models:/{config.MODEL_NAME}/{version}")


def day_rows(data, day):
    """All hours recorded in hour.csv for one date."""
    return data[data["dteday"] == pd.Timestamp(day)]


def what_if_frame(calendar, weather, temp_c, humidity, wind):
    """24 rows, one per hour, using the chosen day's calendar fields and the chosen weather."""
    return pd.DataFrame({
        "season": int(calendar["season"]),
        "mnth": int(calendar["mnth"]),
        "hr": list(range(24)),
        "weekday": int(calendar["weekday"]),
        "weathersit": int(weather),
        "yr": int(calendar["yr"]),
        "holiday": int(calendar["holiday"]),
        "workingday": int(calendar["workingday"]),
        "temp": (temp_c - TEMP_MIN) / (TEMP_MAX - TEMP_MIN),
        "hum": humidity / 100,
        "windspeed": wind / WIND_MAX,
    })[config.FEATURES]


def served_predictions(frame):
    """Predictions from the model server, or None after showing an error if it is down."""
    try:
        return predict(frame)
    except requests.ConnectionError:
        st.error(f"No model server at {SERVING_URL}. Start it with `mlflow models serve` (see the README).")
        return None


def describe_day(day, calendar):
    kind = "holiday" if calendar["holiday"] else ("working day" if calendar["workingday"] else "weekend")
    return f"{day:%A, %d %B %Y} ({kind})"


data = get_data()
try:
    client = MlflowClient()
    champion = client.get_model_version_by_alias(config.MODEL_NAME, config.CHAMPION_ALIAS)
    versions = sorted(client.search_model_versions(f"name = '{config.MODEL_NAME}'"), key=lambda v: int(v.version))
except Exception as error:
    st.error(f"Could not read the champion from the Model Registry at {config.TRACKING_URI}: {error}")
    st.stop()

st.title("🚲 Bike demand")
st.markdown(
    f"**@{config.CHAMPION_ALIAS} = version {champion.version}**: {champion.description}. "
    f"Test RMSE (Q4 2012): {champion.tags.get('test_rmse')} rentals/hour."
)
st.caption("The model server loads @champion when it starts, so restart it after a promotion.")

what_if, real_day, compare = st.tabs(["What-if", "Real day", "Compare versions"])

with what_if:
    controls, results = st.columns([1, 2])
    with controls:
        day = st.date_input("Date", value=date(2012, 6, 15), min_value=date(2011, 1, 1),
                            max_value=date(2012, 12, 31), key="what_if_day")
        hour = st.slider("Hour", 0, 23, 17)
        weather = st.selectbox("Weather", list(WEATHER), format_func=WEATHER.get)
        temp_c = st.slider("Temperature (°C)", TEMP_MIN, TEMP_MAX, 22)
        humidity = st.slider("Humidity (%)", 0, 100, 50)
        wind = st.slider("Wind speed (0 to 67)", 0, WIND_MAX, 12,
                         help="hour.csv divides wind speed by 67; the dataset does not state the units.")
    calendar = day_rows(data, day).iloc[0]
    predictions = served_predictions(what_if_frame(calendar, weather, temp_c, humidity, wind))
    if predictions is not None:
        with results:
            st.metric(f"Predicted rentals at {hour:02d}:00", f"{max(predictions[hour], 0):.0f}")
            curve = pd.DataFrame({"hour": range(24), "predicted": [max(p, 0) for p in predictions]})
            st.line_chart(curve, x="hour", y="predicted", y_label="rentals per hour")
            st.caption(f"{describe_day(day, calendar)}. Calendar fields come from hour.csv; weather comes from the form.")

with real_day:
    day = st.date_input("Day in the test period", value=date(2012, 11, 1), min_value=TEST_START,
                        max_value=TEST_END, key="real_day")
    rows = day_rows(data, day)
    X, y = features_and_target(rows)
    predictions = served_predictions(X)
    if predictions is not None:
        a, b, c = st.columns(3)
        a.metric("Actual rentals", f"{int(y.sum()):,}")
        b.metric("Predicted rentals", f"{sum(predictions):,.0f}")
        c.metric("Hourly RMSE this day", f"{evaluate(y, predictions)['rmse']:.1f}")
        chart = pd.DataFrame({"hour": rows["hr"].to_numpy(), "actual": y.to_numpy(), "predicted": predictions})
        st.line_chart(chart, x="hour", y=["actual", "predicted"], y_label="rentals per hour")
        note = f" Only {len(rows)} of 24 hours are in hour.csv for this day." if len(rows) < 24 else ""
        st.caption(describe_day(day, rows.iloc[0]) + "." + note)

with compare:
    labels = {v.version: f"v{v.version} ({v.tags.get('decision')}): {v.description}" for v in versions}
    numbers = list(labels)
    previous = [v.version for v in versions if v.version != champion.version and v.tags.get("decision") == "promoted"]
    others = [n for n in numbers if n != champion.version]
    default_b = (previous or others or [champion.version])[-1]
    left, right = st.columns(2)
    version_a = left.selectbox("Version A", numbers, index=numbers.index(champion.version), format_func=labels.get)
    version_b = right.selectbox("Version B", numbers, index=numbers.index(default_b), format_func=labels.get)
    day = st.date_input("Day in the test period", value=date(2012, 12, 5), min_value=TEST_START,
                        max_value=TEST_END, key="compare_day")
    rows = day_rows(data, day)
    X, y = features_and_target(rows)
    chart = pd.DataFrame({"hour": rows["hr"].to_numpy(), "actual": y.to_numpy()})
    for column, version in zip(st.columns(2), [version_a, version_b]):
        predictions = load_version(version).predict(X)
        chart[f"v{version}"] = predictions
        test_rmse = next(v.tags.get("test_rmse") for v in versions if v.version == version)
        column.metric(f"v{version}: hourly RMSE this day", f"{evaluate(y, predictions)['rmse']:.1f}",
                      help=f"Test RMSE over all of Q4 2012: {test_rmse}")
    st.line_chart(chart, x="hour", y=[c for c in chart.columns if c != "hour"], y_label="rentals per hour")
    st.caption(describe_day(day, rows.iloc[0]))
