# Put all backtest results together and make the tables, tests and figures for RQ1 and RQ3.
#
# Run from the project folder:  python -m src.run_results
#
# Output:
#   results/tables/metrics_rq1_overall.csv, metrics_rq1_by_h.csv, metrics_rq1_by_series.csv
#   results/tables/metrics_rq3_by_type.csv, wins_by_type.csv
#   results/tables/dm_tests.csv
#   results/figures/rq1_mase_by_horizon.png, rq1_overall.png, rq3_heatmap_series.png, rq3_by_type.png

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.evaluation.dm_test import dm_test, holm  # noqa: E402
from src.evaluation.metrics import summarize  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
FIG = ROOT / "results" / "figures"

ORDER = ["SeasonalNaive", "Naive", "ETS", "ARIMA", "LightGBM", "NBEATS", "Chronos2", "Chronos2-FT"]
NICE = {"SeasonalNaive": "Seasonal Naive", "Naive": "Naive", "ETS": "ETS", "ARIMA": "ARIMA",
        "LightGBM": "LightGBM", "NBEATS": "N-BEATS", "Chronos2": "Chronos-2 zero-shot",
        "Chronos2-FT": "Chronos-2 fine-tuned"}

# product types (from the data exploration)
TYPES = {
    "Apples": "storable", "Pears": "storable", "Lemons": "storable",
    "Tomatoes": "greenhouse", "Cucumbers": "greenhouse", "Peppers": "greenhouse",
    "Courgettes": "greenhouse", "Egg Plants": "greenhouse",
    "Lettuces": "open field", "Cauliflowers": "open field",
    "Strawberries": "strawberries",
}


def short(series_id):
    # "BE | Tomatoes - Round" -> "BE Tomatoes"
    country, rest = series_id.split(" | ")
    return country + " " + rest.split(" - ")[0]


def load_all():
    parts = [pd.read_parquet(TABLES / "backtest_day4.parquet")]
    for f in sorted(TABLES.glob("backtest_day5_*.parquet")):
        parts.append(pd.read_parquet(f))
    res = pd.concat(parts, ignore_index=True)
    res["name"] = res["series_id"].map(short)
    res["product"] = res["name"].str.split(" ", n=1).str[1]
    res["type"] = res["product"].map(TYPES)
    models = [m for m in ORDER if m in res["model"].unique()]
    return res, models


def dm_table(res, models, best):
    # compare the best model with every other model, per horizon
    # one value per origin: the mean scaled error over the 29 series (same forecasts for both models)
    rows = []
    piv = res.pivot_table(index=["origin", "h", "series_id"], columns="model", values="scaled_err")
    for other in models:
        if other == best:
            continue
        for h in range(1, 9):
            part = piv.xs(h, level="h")[[best, other]].dropna()
            per_origin = part.groupby(level="origin").mean()
            d = (per_origin[best] - per_origin[other]).sort_index()
            stat, p = dm_test(d.to_numpy(), h)
            improvement = 100 * (1 - per_origin[best].mean() / per_origin[other].mean())
            rows.append({"best": best, "other": other, "h": h, "n_origins": len(d),
                         "mase_best": per_origin[best].mean(), "mase_other": per_origin[other].mean(),
                         "improvement_pct": improvement, "dm_stat": stat, "p_value": p})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p_value"])
    out["significant_5pct"] = out["p_holm"] < 0.05
    return out.round(4)


def fig_by_horizon(by_h, models):
    piv = by_h.pivot(index="h", columns="model", values="MASE")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for i, m in enumerate(models):
        lw = 2.4 if m.startswith("Chronos") or m == "ARIMA" else 1.3
        ax.plot(piv.index, piv[m], marker="o", lw=lw, color=colors[i], label=NICE[m])
    ax.axhline(1, color="grey", ls="--", lw=0.8)
    ax.set_xticks(range(1, 9))
    ax.set_xlabel("weeks ahead (horizon)")
    ax.set_ylabel("MASE (lower is better)")
    ax.set_title("Forecast error by horizon, 29 series, 25 forecast dates")
    ax.legend(fontsize=9)
    fig.savefig(FIG / "rq1_mase_by_horizon.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def fig_overall(overall):
    o = overall.sort_values("MASE", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh([NICE[m] for m in o["model"]], o["MASE"], color="#4a7a9c")
    for y, v in enumerate(o["MASE"]):
        ax.text(v + 0.01, y, f"{v:.2f}", va="center", fontsize=9)
    ax.axvline(1, color="grey", ls="--", lw=0.8)
    ax.set_xlabel("MASE, all series and horizons (lower is better)")
    ax.set_title("Overall forecast error per model")
    fig.savefig(FIG / "rq1_overall.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def fig_heatmap(by_series, models, res):
    piv = by_series.pivot(index="series_id", columns="model", values="MASE")[models]
    piv.index = [short(s) for s in piv.index]
    order = res.groupby("name")["type"].first().sort_values().index
    piv = piv.loc[[n for n in order if n in piv.index]]
    fig, ax = plt.subplots(figsize=(11, 12))
    im = ax.imshow(piv.to_numpy(), cmap="RdYlGn_r", vmin=0.1, vmax=1.4, aspect="auto")
    ax.set_xticks(range(len(models)), [NICE[m] for m in models], rotation=35, ha="right")
    ax.set_yticks(range(len(piv.index)), [f"{n} ({res.loc[res['name'] == n, 'type'].iloc[0]})" for n in piv.index])
    best = piv.to_numpy().argmin(axis=1)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.iat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                    fontweight="bold" if j == best[i] else "normal")
    fig.colorbar(im, label="MASE (lower is better)")
    ax.set_title("MASE per series and model (bold = best model for that series)")
    fig.savefig(FIG / "rq3_heatmap_series.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def fig_by_type(by_type, models):
    piv = by_type.pivot(index="type", columns="model", values="MASE")[models]
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(piv.index))
    w = 0.8 / len(models)
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for i, m in enumerate(models):
        ax.bar(x + i * w - 0.4 + w / 2, piv[m], width=w, label=NICE[m], color=colors[i])
    ax.set_xticks(x, piv.index)
    ax.set_ylabel("MASE (lower is better)")
    ax.set_title("Forecast error per product type")
    ax.legend(fontsize=8, ncol=4)
    fig.savefig(FIG / "rq3_by_type.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    res, models = load_all()
    print(f"models: {models}")

    overall = summarize(res, ["model"]).sort_values("MASE")
    by_h = summarize(res, ["model", "h"])
    by_series = summarize(res, ["model", "series_id"])
    by_type = summarize(res, ["model", "type"])
    by_type_h = summarize(res, ["model", "type", "h"])
    overall.to_csv(TABLES / "metrics_rq1_overall.csv", index=False)
    by_h.to_csv(TABLES / "metrics_rq1_by_h.csv", index=False)
    by_series.to_csv(TABLES / "metrics_rq1_by_series.csv", index=False)
    by_type.to_csv(TABLES / "metrics_rq3_by_type.csv", index=False)
    by_type_h.to_csv(TABLES / "metrics_rq3_by_type_h.csv", index=False)

    # which model wins how many series, per product type
    piv = by_series.pivot(index="series_id", columns="model", values="MASE")[models]
    winners = piv.idxmin(axis=1).rename("winner").to_frame()
    winners["type"] = [TYPES[short(s).split(" ", 1)[1]] for s in winners.index]
    wins = pd.crosstab(winners["type"], winners["winner"])
    wins.to_csv(TABLES / "wins_by_type.csv")

    best = overall["model"].iloc[0]
    dm = dm_table(res, models, best)
    dm.to_csv(TABLES / "dm_tests.csv", index=False)

    fig_by_horizon(by_h, models)
    fig_overall(overall)
    fig_heatmap(by_series, models, res)
    fig_by_type(by_type, models)

    print(overall.to_string(index=False))
    print("\nMASE per horizon:")
    print(by_h.pivot(index="h", columns="model", values="MASE")[models].round(3).to_string())
    print("\nMASE per product type:")
    print(by_type.pivot(index="type", columns="model", values="MASE")[models].round(3).to_string())
    print("\nwins per type:")
    print(wins.to_string())
    print(f"\nDM tests, best model = {best}:")
    print(dm.pivot(index="h", columns="other", values="p_holm").to_string())


if __name__ == "__main__":
    main()
