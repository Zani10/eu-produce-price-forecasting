# Run the backtest for the simple, classic and LightGBM models and save the results.
#
# Run from the project folder:  python -m src.run_backtest
#
# Output:
#   results/tables/backtest_day4.parquet       = every single forecast (one row per series, model, origin, week)
#   results/tables/metrics_day4_overall.csv    = one row per model
#   results/tables/metrics_day4_by_h.csv       = per model and horizon (1 to 8 weeks)
#   results/tables/metrics_day4_by_series.csv  = per model and series
# (day4 / day5 in the file names: first part of the backtest = these models, second part = N-BEATS and Chronos-2)

import time
from pathlib import Path

import pandas as pd

from src.evaluation.backtest import load_data, make_origins, run_model
from src.evaluation.metrics import add_errors, summarize
from src.models.classic import arima, ets, naive, seasonal_naive
from src.models.lgbm import lightgbm

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"

MODELS = {
    "Naive": naive,
    "SeasonalNaive": seasonal_naive,
    "ETS": ets,
    "ARIMA": arima,
    "LightGBM": lightgbm,
}


def main():
    data = load_data()
    origins = make_origins(data["ds"])
    print(f"{data['unique_id'].nunique()} series, {len(origins)} origins "
          f"from {origins[0].date()} to {origins[-1].date()}, horizons 1-8")

    all_res = []
    for name, fn in MODELS.items():
        start = time.time()
        res = run_model(fn, name, data, origins)
        print(f"{name}: {len(res)} forecasts in {time.time() - start:.0f}s")
        all_res.append(res)

    res = add_errors(pd.concat(all_res, ignore_index=True))
    res.to_parquet(TABLES / "backtest_day4.parquet", index=False)

    overall = summarize(res, ["model"]).sort_values("MASE")
    by_h = summarize(res, ["model", "h"])
    by_series = summarize(res, ["model", "series_id"])
    overall.to_csv(TABLES / "metrics_day4_overall.csv", index=False)
    by_h.to_csv(TABLES / "metrics_day4_by_h.csv", index=False)
    by_series.to_csv(TABLES / "metrics_day4_by_series.csv", index=False)

    print("\nresults per model (lower MASE = better, below 1 = better than 'same week last year' on training data):")
    print(overall.to_string(index=False))
    print("\nMASE per horizon (weeks ahead):")
    print(by_h.pivot(index="h", columns="model", values="MASE").round(3).to_string())


if __name__ == "__main__":
    main()
