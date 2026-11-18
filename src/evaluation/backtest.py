# The shared "exam" for all models: a rolling-origin backtest.
#
# We pretend to stand at a date in the past (the origin), give the model ONLY the prices up to that date,
# ask for the next 8 weeks, and compare with what really happened. Then we move 4 weeks forward and repeat.
# Every model gets exactly the same origins, so the comparison is fair.
#
# Test period = the last 104 weeks (2 years) of our frozen data. Origins every 4 weeks.
# Only origins where all 8 weeks are inside the data are used, so every origin has horizons 1 to 8.

from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.metrics import mase_scale

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed" / "prices_weekly.parquet"
SAMPLE = ROOT / "results" / "tables" / "sample_series.csv"

H = 8             # forecast 1 to 8 weeks ahead
TEST_WEEKS = 104  # last 2 years are the test period
STEP = 4          # a new origin every 4 weeks


def load_data():
    df = pd.read_parquet(DATA)
    ids = pd.read_csv(SAMPLE)["series_id"]
    df = df[df["series_id"].isin(ids)].copy()

    # y_true = the real reported price, empty if that week was not reported (also when we filled it in cleaning)
    df["y_true"] = df["price"].where(~df["is_missing"])

    # y_hist = what the model is allowed to see as history.
    # We do NOT use the straight-line filled values from cleaning here, because a filled week uses
    # the price AFTER the gap (future information if the origin is inside the gap).
    # Instead we repeat the last real price (forward fill). This only looks back in time,
    # so it is safe at every origin, also for long gaps.
    df["y_hist"] = df.groupby("series_id")["y_true"].ffill()

    df = df.rename(columns={"series_id": "unique_id", "date": "ds"})
    return df[["unique_id", "ds", "y_true", "y_hist"]].sort_values(["unique_id", "ds"])


def make_origins(dates):
    dates = pd.Series(sorted(dates.unique()))
    last = dates.iloc[-1]
    first_origin = last - pd.Timedelta(weeks=TEST_WEEKS)
    origins = []
    origin = first_origin
    while origin + pd.Timedelta(weeks=H) <= last:
        origins.append(origin)
        origin = origin + pd.Timedelta(weeks=STEP)
    return origins


def run_model(model_fn, name, data, origins):
    # model_fn(history, h) must return a table with: unique_id, ds, y_pred, q10, q50, q90
    rows = []
    for i, origin in enumerate(origins):
        hist = data[data["ds"] <= origin][["unique_id", "ds", "y_hist"]].rename(columns={"y_hist": "y"})
        hist = hist.dropna(subset=["y"])
        fc = model_fn(hist, H)

        # the MASE scale uses only the history up to this origin
        scales = hist.groupby("unique_id")["y"].apply(mase_scale).rename("scale").reset_index()

        truth = data[(data["ds"] > origin) & (data["ds"] <= origin + pd.Timedelta(weeks=H))]
        out = fc.merge(truth[["unique_id", "ds", "y_true"]], on=["unique_id", "ds"], how="left")
        out = out.merge(scales, on="unique_id", how="left")
        out["origin"] = origin
        out["h"] = ((out["ds"] - origin) / pd.Timedelta(weeks=1)).round().astype(int)
        out["model"] = name
        rows.append(out)
        print(f"  {name}: origin {i + 1}/{len(origins)} ({origin.date()})", end="\r")
    print()
    res = pd.concat(rows, ignore_index=True)
    # we only score weeks with a real reported price
    res = res[res["y_true"].notna()]
    cols = ["unique_id", "model", "origin", "h", "ds", "y_true", "y_pred", "q10", "q50", "q90", "scale"]
    for c in ["q10", "q50", "q90"]:
        if c not in res:
            res[c] = np.nan
    return res[cols].rename(columns={"unique_id": "series_id", "ds": "target_date"})
