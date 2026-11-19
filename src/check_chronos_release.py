# Check for RQ1: is Chronos-2 still the best when we only look at forecasts made after it was released?
#
# Chronos-2 was made public in October 2025. Our test period starts in October 2024, so in theory
# public prices from before its release could be in the data it was pre-trained on.
# Here I split the 25 forecast origins in two parts and compute the MASE of every model again:
#   before release = origins up to 22 Sep 2025 (13 origins)
#   after release  = origins from 20 Oct 2025 (12 origins)
# No model is run again, it only uses the saved backtest results.
#
# Run from the project folder:  python -m src.check_chronos_release
# Output: results/tables/rq1_after_chronos_release.csv

from pathlib import Path

import pandas as pd

from src.evaluation.metrics import summarize

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
RELEASE = pd.Timestamp("2025-10-20")  # first forecast origin after the public release of Chronos-2


def main():
    parts = [pd.read_parquet(TABLES / "backtest_day4.parquet")]
    for f in sorted(TABLES.glob("backtest_day5_*.parquet")):
        parts.append(pd.read_parquet(f))
    res = pd.concat(parts, ignore_index=True)
    res["period"] = (res["origin"] >= RELEASE).map({True: "after release", False: "before release"})

    out = []
    for name, part in [("all origins", res)] + list(res.groupby("period")):
        s = summarize(part, ["model"])
        s.insert(0, "period", name)
        s.insert(1, "n_origins", part["origin"].nunique())
        s["rank"] = s["MASE"].rank().astype(int)
        out.append(s.sort_values("MASE"))
    out = pd.concat(out, ignore_index=True)
    out.to_csv(TABLES / "rq1_after_chronos_release.csv", index=False)
    print(out[["period", "n_origins", "model", "MASE", "MAE", "rank"]].to_string(index=False))


if __name__ == "__main__":
    main()
