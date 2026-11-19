# RQ2: does weather from the growing regions make the forecasts better?
#
# Run from the project folder:  python -m src.run_weather
#
# Same exam as the other models (same 25 origins, horizons 1-8, only real weeks scored, same MASE scale).
# Four versions of the same direct LightGBM (see src/models/lgbm_weather.py):
#   LGBM-direct            = prices only (the fair comparison)
#   LGBM-direct+weather    = + weather known at the forecast origin (no leakage)
#   LGBM-direct+weather-light = + only 3 weather features (check if fewer features work better)
#   LGBM-direct+oracle     = + the REAL weather of the target weeks (uses the future, look-back analysis only)
#
# Output:
#   results/tables/backtest_weather.parquet
#   results/tables/metrics_rq2_overall.csv, metrics_rq2_by_h.csv, metrics_rq2_by_series.csv, metrics_rq2_by_product.csv
#   results/tables/rq2_dm_tests.csv
#   results/figures/rq2_weather_effect.png

import math
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.evaluation.backtest import STEP, load_data, make_origins, run_model  # noqa: E402
from src.evaluation.dm_test import holm  # noqa: E402
from src.evaluation.metrics import add_errors, summarize  # noqa: E402
from src.models.lgbm_weather import (ORACLE_FEATURES, PRICE_FEATURES, WEATHER_FEATURES, WEATHER_LIGHT,  # noqa: E402
                                     build_table, make_model)  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"

BASE = "LGBM-direct"
WEATHER = "LGBM-direct+weather"
LIGHT = "LGBM-direct+weather-light"
ORACLE = "LGBM-direct+oracle"


def dm_test(d, q):
    # Diebold-Mariano test on a series of loss differences d (one value per forecast origin).
    # d < 0 means the first model has smaller errors.
    # q = how many neighbour origins overlap (origins are 4 weeks apart, so forecasts 5-8 weeks ahead overlap
    # with the next origin). We use a Newey-West variance with q lags and the Harvey-Leybourne-Newbold
    # small sample correction, and a t distribution with n-1 degrees of freedom.
    d = np.asarray(d, dtype=float)
    d = d[~np.isnan(d)]
    n = len(d)
    mean = d.mean()
    dc = d - mean
    var = np.sum(dc ** 2) / n
    for lag in range(1, q + 1):
        var += 2 * np.sum(dc[lag:] * dc[:-lag]) / n
    if var <= 0:
        return mean, np.nan, np.nan
    dm = mean / math.sqrt(var / n)
    k = q + 1
    dm *= math.sqrt((n + 1 - 2 * k + k * (k - 1) / n) / n)
    p = 2 * stats.t.sf(abs(dm), df=n - 1)
    return mean, dm, p


def compare(res, a, b):
    # test model a against model b, per horizon and over all horizons
    piv = res[res["model"].isin([a, b])].pivot_table(
        index=["series_id", "origin", "h"], columns="model", values="scaled_err").dropna()
    piv["d"] = piv[a] - piv[b]
    rows = []
    for h, g in piv.groupby(level="h"):
        per_origin = g.groupby(level="origin")["d"].mean()
        mean, dm, p = dm_test(per_origin, q=math.ceil(h / STEP) - 1)
        rows.append({"model": a, "vs": b, "h": h, "mase_a": g[a].mean(), "mase_b": g[b].mean(),
                     "improvement_pct": 100 * (g[b].mean() - g[a].mean()) / g[b].mean(), "dm": dm, "p_value": p})
    per_origin = piv.groupby(level="origin")["d"].mean()
    mean, dm, p = dm_test(per_origin, q=1)
    rows.append({"model": a, "vs": b, "h": "all", "mase_a": piv[a].mean(), "mase_b": piv[b].mean(),
                 "improvement_pct": 100 * (piv[b].mean() - piv[a].mean()) / piv[b].mean(), "dm": dm, "p_value": p})
    out = pd.DataFrame(rows)
    # Holm correction over the 9 tests of this comparison (8 horizons + all), same idea as in RQ1
    out["p_holm"] = holm(out["p_value"])
    return out


def main():
    start = time.time()
    data = load_data()
    origins = make_origins(data["ds"])
    sample = pd.read_csv(TABLES / "sample_series.csv")
    table = build_table(data, sample)
    print(f"feature table: {len(table):,} rows, built in {time.time() - start:.0f}s")

    versions = {
        BASE: PRICE_FEATURES,
        WEATHER: PRICE_FEATURES + WEATHER_FEATURES,
        LIGHT: PRICE_FEATURES + WEATHER_LIGHT,
        ORACLE: PRICE_FEATURES + ORACLE_FEATURES,
    }
    all_res = []
    for name, feats in versions.items():
        t0 = time.time()
        res = run_model(make_model(table, feats), name, data, origins)
        print(f"{name}: {len(res)} forecasts in {time.time() - t0:.0f}s")
        all_res.append(res)
    res = add_errors(pd.concat(all_res, ignore_index=True))
    res["product"] = res["series_id"].map(sample.set_index("series_id")["product"])
    res.to_parquet(TABLES / "backtest_weather.parquet", index=False)

    overall = summarize(res, ["model"]).sort_values("MASE")
    by_h = summarize(res, ["model", "h"])
    by_series = summarize(res, ["model", "series_id"])
    by_product = summarize(res, ["model", "product"])
    overall.to_csv(TABLES / "metrics_rq2_overall.csv", index=False)
    by_h.to_csv(TABLES / "metrics_rq2_by_h.csv", index=False)
    by_series.to_csv(TABLES / "metrics_rq2_by_series.csv", index=False)
    by_product.to_csv(TABLES / "metrics_rq2_by_product.csv", index=False)

    dm = pd.concat([compare(res, WEATHER, BASE), compare(res, LIGHT, BASE), compare(res, ORACLE, BASE)],
                   ignore_index=True).round(4)
    dm.to_csv(TABLES / "rq2_dm_tests.csv", index=False)

    # figure: change in MASE per product (negative = weather helps)
    p = by_product.pivot(index="product", columns="model", values="MASE")
    delta = pd.DataFrame({
        "weather (known at origin)": 100 * (p[WEATHER] - p[BASE]) / p[BASE],
        "weather-light (3 features)": 100 * (p[LIGHT] - p[BASE]) / p[BASE],
        "oracle (real future weather)": 100 * (p[ORACLE] - p[BASE]) / p[BASE],
    }).sort_values("weather (known at origin)")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    delta.plot.barh(ax=ax, color=["#2c6a4c", "#7fb89a", "#c9a227"])
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_xlabel("change in MASE vs. no weather (%), negative = better")
    ax.set_ylabel("")
    ax.set_title("Effect of weather on forecast error, per product")
    fig.savefig(FIGURES / "rq2_weather_effect.png", dpi=110, bbox_inches="tight")

    print(f"\nall done in {time.time() - start:.0f}s\n")
    print(overall.to_string(index=False))
    print("\nDiebold-Mariano tests (negative dm = first model better):")
    print(dm.to_string(index=False))
    print("\nchange in MASE per product (%):")
    print(delta.round(1).to_string())


if __name__ == "__main__":
    main()
