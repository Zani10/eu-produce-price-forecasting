# Make the "current" forecasts that the assistant uses.
# We stand at the last week of our frozen data (week of 19 Oct 2026) and forecast 1 to 8 weeks ahead
# for all 29 series with ARIMA, the best classic model of the backtest (Chronos-2 was best overall,
# but its results came later; ARIMA is very close, fast and needs no GPU).
# Output: results/tables/current_forecasts.csv  (one row per series per week ahead)

from pathlib import Path

import pandas as pd

from src.evaluation.backtest import H, load_data
from src.models.classic import arima

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "tables" / "current_forecasts.csv"
MODEL = "ARIMA"


def main():
    data = load_data()
    origin = data["ds"].max()
    hist = data[["unique_id", "ds", "y_hist"]].rename(columns={"y_hist": "y"}).dropna(subset=["y"])
    fc = arima(hist, H)
    fc["origin"] = origin
    fc["h"] = ((fc["ds"] - origin) / pd.Timedelta(weeks=1)).round().astype(int)
    fc["model"] = MODEL
    # a price can't be negative, so we cut the low end of the range at 0
    for col in ["y_pred", "q10", "q50", "q90"]:
        fc[col] = fc[col].clip(lower=0).round(4)
    fc = fc.rename(columns={"unique_id": "series_id", "ds": "week"})
    fc = fc[["series_id", "model", "origin", "h", "week", "y_pred", "q10", "q50", "q90"]]
    fc.to_csv(OUT, index=False)
    print(f"origin {origin.date()}, {fc['series_id'].nunique()} series, saved {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
