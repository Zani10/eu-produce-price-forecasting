# Choose the final sample of series with five fixed rules, and make the dataset table for the thesis.
#
# Run from the project folder:  python -m src.data.select_sample
#
# Input:  results/tables/series_summary.csv (made by src/data/clean.py)
# Output: results/tables/sample_series.csv  = the 29 chosen series
#         results/tables/dataset_table.csv  = one row per series with period, missing and stale weeks, price stats

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "results" / "tables"

MAX_MISSING = 20
MAX_STALE = 25
COUNTRIES = ["BE", "NL", "DE", "FR", "ES", "IT"]
STILL_REPORTING = "2026-06-01"
PRODUCTS = ["Tomatoes", "Apples", "Pears", "Cucumbers", "Peppers", "Courgettes",
            "Egg Plants", "Lettuces", "Cauliflowers", "Lemons", "Strawberries"]


def main():
    s = pd.read_csv(TABLES / "series_summary.csv")
    print(f"start: {len(s)} series")

    s = s[s["missing_pct"] <= MAX_MISSING]
    print(f"rule 1, max {MAX_MISSING}% missing weeks: {len(s)}")

    s = s[s["stale_pct"] <= MAX_STALE]
    print(f"rule 2, max {MAX_STALE}% stale weeks: {len(s)}")

    s = s[s["country"].isin(COUNTRIES) & (s["last_week"] >= STILL_REPORTING)]
    print(f"rule 3, main countries and still reporting: {len(s)}")

    # rule 4: one series per product and country
    # prefer "All types and varieties", then the one with the least missing weeks
    s["all_types"] = s["variety"].str.contains("All types|National weighted")
    s = s.sort_values(["all_types", "missing_pct", "stale_pct"], ascending=[False, True, True])
    s = s.groupby(["country", "product"]).head(1)
    print(f"rule 4, one series per product and country: {len(s)}")

    # rule 5: only our products
    s = s[s["product"].isin(PRODUCTS)]
    print(f"rule 5, our products: {len(s)}")

    s = s.sort_values(["product", "country"])
    s[["series_id", "country", "product", "variety"]].to_csv(TABLES / "sample_series.csv", index=False)

    table = pd.DataFrame({
        "country": s["country"],
        "product": s["product"],
        "variety": s["variety"],
        "period": s["first_week"] + " to " + s["last_week"],
        "observations": s["weeks_observed"],
        "missing_weeks": s["weeks_missing"],
        "missing_pct": s["missing_pct"],
        "filled_weeks": s["weeks_filled"],
        "stale_pct": s["stale_pct"],
        "mean_eur_kg": s["mean"],
        "min_eur_kg": s["min"],
        "max_eur_kg": s["max"],
        "std_eur_kg": s["std"],
    })
    table.to_csv(TABLES / "dataset_table.csv", index=False)

    print(table.to_string(index=False))
    print(f"\nfinal sample: {len(table)} series, {table['product'].nunique()} products, "
          f"{table['country'].nunique()} countries")


if __name__ == "__main__":
    main()
