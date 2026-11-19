# Extra checks for RQ1 and RQ3, made after the main results.
# No model is run again, it only uses the saved backtest results.
#
#  1. sign test: on how many series is Chronos-2 better than ARIMA
#  2. on how many forecast origins is Chronos-2 better than ARIMA (mean over all series and horizons)
#  3. how much of the differences in MASE comes from the series and how much from the model
#  4. advantage of Chronos-2 over ARIMA before and after the release of Chronos-2 (Oct 2025)
#  5. fine-tuned vs zero-shot Chronos-2, first and second half of the test period
#  6. Diebold-Mariano test again, with a lag order that fits origins 4 weeks apart
#  7. error of Chronos-2 as a share of the price (relative MAE) per horizon
#  8. weather experiment: Holm correction over all 27 tests together
#  9. MASE scale of the pear series (why Spanish pears have a high MASE)
#
# Run from the project folder:  python -m src.check_extra_stats
# Output: results/tables/extra_stats.csv

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.evaluation.dm_test import holm

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
RELEASE = pd.Timestamp("2025-10-20")


def load():
    parts = [pd.read_parquet(TABLES / "backtest_day4.parquet")]
    for f in sorted(TABLES.glob("backtest_day5_*.parquet")):
        parts.append(pd.read_parquet(f))
    return pd.concat(parts, ignore_index=True)


def dm_lag(d, q):
    # plain Diebold-Mariano test with autocovariances up to lag q
    d = np.asarray(d, dtype=float)
    n = len(d)
    mean = d.mean()
    gamma = [np.mean((d[k:] - mean) * (d[:n - k] - mean)) for k in range(q + 1)]
    var = (gamma[0] + 2 * sum(gamma[1:])) / n
    if var <= 0:
        var = gamma[0] / n
    stat = mean / math.sqrt(var)
    return stat, 2 * stats.t.sf(abs(stat), df=n - 1)


def main():
    res = load()
    rows = []

    def add(name, value, note=""):
        rows.append({"check": name, "value": value, "note": note})

    # 1. sign test over series
    by_series = res.groupby(["series_id", "model"])["scaled_err"].mean().unstack()
    wins = int((by_series["Chronos2"] < by_series["ARIMA"]).sum())
    n = len(by_series)
    add("series where Chronos-2 beats ARIMA", wins, f"of {n}")
    add("sign test p (two-sided)", round(stats.binomtest(wins, n, 0.5).pvalue, 4))

    # 2. origins where Chronos-2 beats ARIMA (mean scaled error over series and horizons)
    both = res[res["model"].isin(["Chronos2", "ARIMA"])]
    by_origin = both.groupby(["origin", "model"])["scaled_err"].mean().unstack()
    add("origins where Chronos-2 beats ARIMA", int((by_origin["Chronos2"] < by_origin["ARIMA"]).sum()),
        f"of {len(by_origin)}")

    # 3. two-way split of the MASE matrix (series x model), without Seasonal Naive
    m = by_series.drop(columns=["SeasonalNaive"])
    grand = m.values.mean()
    ss_series = m.shape[1] * ((m.mean(axis=1) - grand) ** 2).sum()
    ss_model = m.shape[0] * ((m.mean(axis=0) - grand) ** 2).sum()
    ss_total = ((m.values - grand) ** 2).sum()
    add("share of MASE variation from the series (%)", round(100 * ss_series / ss_total, 1), "7 models, 29 series")
    add("share of MASE variation from the model (%)", round(100 * ss_model / ss_total, 1), "7 models, 29 series")

    # 4. before and after the release of Chronos-2
    for label, part in (("before release", res[res["origin"] < RELEASE]), ("after release", res[res["origin"] >= RELEASE])):
        mase = part.groupby("model")["scaled_err"].mean()
        add(f"Chronos-2 vs ARIMA, {label}: % lower MASE",
            round(100 * (1 - mase["Chronos2"] / mase["ARIMA"]), 1),
            f"{part['origin'].nunique()} origins, MASE {mase['Chronos2']:.3f} vs {mase['ARIMA']:.3f}")
        add(f"fine-tuned vs zero-shot, {label}: % higher MASE",
            round(100 * (mase["Chronos2-FT"] / mase["Chronos2"] - 1), 1),
            f"MASE {mase['Chronos2-FT']:.3f} vs {mase['Chronos2']:.3f}")

    # 6. DM test with lag ceil(h/4)-1, Holm over all 56 tests
    piv = res.pivot_table(index=["origin", "h", "series_id"], columns="model", values="scaled_err")
    tests = []
    for other in [c for c in piv.columns if c != "Chronos2"]:
        for h in range(1, 9):
            part = piv.xs(h, level="h")[["Chronos2", other]].dropna()
            per_origin = part.groupby(level="origin").mean().sort_index()
            d = per_origin["Chronos2"] - per_origin[other]
            stat, p = dm_lag(d.to_numpy(), math.ceil(h / 4) - 1)
            tests.append({"other": other, "h": h, "p": p})
    t = pd.DataFrame(tests)
    t["p_holm"] = holm(t["p"])
    for other, g in t.groupby("other"):
        add(f"DM lag ceil(h/4)-1: significant horizons vs {other}", int((g["p_holm"] < 0.05).sum()),
            f"unadjusted p<0.05 at {int((g['p'] < 0.05).sum())} horizons")

    # 7. relative MAE of Chronos-2 per horizon (sum |error| / sum price)
    c2 = res[res["model"] == "Chronos2"].copy()
    c2["abs_err"] = (c2["y_true"] - c2["y_pred"]).abs()
    for h in (1, 4, 8):
        g = c2[c2["h"] == h]
        add(f"Chronos-2 relative MAE at h={h} (% of price)", round(100 * g["abs_err"].sum() / g["y_true"].sum(), 1))
    add("Chronos-2 relative MAE, all horizons (% of price)", round(100 * c2["abs_err"].sum() / c2["y_true"].sum(), 1))

    # 9. MASE scale of the three pear series (mean in-sample seasonal naive error, €/kg)
    scale = res[res["model"] == "ARIMA"].groupby("series_id")["scale"].mean()
    for sid in [x for x in scale.index if "Pears" in x]:
        add(f"MASE scale {sid.split(' | ')[0]} pears (EUR/kg)", round(scale[sid], 3))

    # 8. weather experiment: Holm over all 27 Diebold-Mariano tests together (3 versions x 9 tests)
    w = pd.read_csv(TABLES / "rq2_dm_tests.csv")
    w["p_holm_all"] = holm(w["p_value"])
    add("RQ2: significant tests with Holm over all 27 tests", int((w["p_holm_all"] < 0.05).sum()),
        f"smallest adjusted p {w['p_holm_all'].min():.3f}")

    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "extra_stats.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
