# Check for RQ2: does the weather result change with another random seed?
#
# LightGBM uses some randomness when it builds the trees. The main run (src/run_weather.py) uses seed 42.
# Here I run the prices-only model and the two weather versions again with seeds 1 and 2,
# on the same exam (same 25 origins, horizons 1-8), and compare the MASE.
#
# Run from the project folder:  python -m src.check_weather_seeds   (takes a few minutes)
# Output: results/tables/rq2_seed_check.csv

import time
from pathlib import Path

import pandas as pd

from src.evaluation.backtest import load_data, make_origins, run_model
from src.evaluation.metrics import add_errors
from src.models.lgbm_weather import PRICE_FEATURES, WEATHER_FEATURES, WEATHER_LIGHT, build_table, make_model

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"

VERSIONS = {
    "LGBM-direct": PRICE_FEATURES,
    "LGBM-direct+weather": PRICE_FEATURES + WEATHER_FEATURES,
    "LGBM-direct+weather-light": PRICE_FEATURES + WEATHER_LIGHT,
}


def main():
    data = load_data()
    origins = make_origins(data["ds"])
    sample = pd.read_csv(TABLES / "sample_series.csv")
    table = build_table(data, sample)

    rows = []
    # seed 42 is the main run, take it from the saved results
    main_run = pd.read_csv(TABLES / "metrics_rq2_overall.csv").set_index("model")["MASE"]
    for name in VERSIONS:
        rows.append({"seed": 42, "model": name, "MASE": main_run[name]})

    for seed in (1, 2):
        for name, feats in VERSIONS.items():
            t0 = time.time()
            res = add_errors(run_model(make_model(table, feats, seed=seed), name, data, origins))
            rows.append({"seed": seed, "model": name, "MASE": res["scaled_err"].mean()})
            print(f"seed {seed} {name}: MASE {rows[-1]['MASE']:.4f} ({time.time() - t0:.0f}s)")

    out = pd.DataFrame(rows)
    base = out[out["model"] == "LGBM-direct"].set_index("seed")["MASE"]
    out["change_vs_prices_only_pct"] = 100 * (out["MASE"] / out["seed"].map(base) - 1)
    out = out.round(4)
    out.to_csv(TABLES / "rq2_seed_check.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
