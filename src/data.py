"""Load the UCI Bike Sharing hourly data and split it by time.

Run `python -m src.data` to print a summary of the three splits.
"""
import pandas as pd

from src import config

LEAKY_COLUMNS = ["casual", "registered"]
UNUSED_COLUMNS = ["instant", "atemp"]


def load_data(path=config.DATA_PATH):
    """Read hour.csv and drop the columns the models must not see."""
    df = pd.read_csv(path, parse_dates=["dteday"])
    df = df.drop(columns=LEAKY_COLUMNS + UNUSED_COLUMNS)
    df["weathersit"] = df["weathersit"].clip(upper=3)
    return df


def split_by_time(df):
    """Train up to TRAIN_END, validate up to VALID_END, and test on the rest."""
    train_end = pd.Timestamp(config.TRAIN_END)
    valid_end = pd.Timestamp(config.VALID_END)
    train = df[df["dteday"] <= train_end]
    valid = df[(df["dteday"] > train_end) & (df["dteday"] <= valid_end)]
    test = df[df["dteday"] > valid_end]
    return train, valid, test


def features_and_target(df):
    """Return the model inputs (X) and the target (y)."""
    return df[config.FEATURES], df[config.TARGET]


if __name__ == "__main__":
    for name, part in zip(["train", "valid", "test"], split_by_time(load_data())):
        hours_per_day = part.groupby("dteday").size()
        print(
            f"{name:<5} | {part['dteday'].min().date()} to {part['dteday'].max().date()} | "
            f"{len(part):>5} rows | {len(hours_per_day)} days ({int((hours_per_day < 24).sum())} partial) | "
            f"mean cnt {part[config.TARGET].mean():.1f}"
        )
