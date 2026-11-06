# Clean the EU fruit and vegetable prices and make one weekly table per series.
#
# Input:  data/interim/eu_fv_prices.parquet  (made by fetch_eu_prices.py)
# Output: data/processed/prices_weekly.parquet   = clean weekly prices, one row per series per week
#         results/tables/series_summary.csv        = one row per series (counts, missing, flags, stats)
#         results/tables/cleaning_report.txt       = what the cleaning did, for the thesis
#
# Rules (see thesis chapter 3):
# - only weekly prices at the "Ex-packaging station" stage, from 2015 until the week of 19 Oct 2026
# - price in euro per kg
# - one row per series per week (duplicates removed)
# - gaps of max 3 weeks are filled with a straight line, longer gaps stay empty
# - strange values and stale prices get a flag, they are NOT removed here

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
IN_FILE = ROOT / "data" / "interim" / "eu_fv_prices.parquet"
OUT_FILE = ROOT / "data" / "processed" / "prices_weekly.parquet"
TABLES = ROOT / "results" / "tables"

STAGE = "Ex-packaging station price"
START = pd.Timestamp("2015-01-05")  # first monday of 2015
END = pd.Timestamp("2026-10-19")    # last week in our frozen data (19-25 oct 2026)
MAX_FILL = 3                         # fill gaps up to 3 weeks
OUTLIER_FACTOR = 3                   # flag if price is 3x higher or lower than the normal level around it
STALE_WEEKS = 4                      # flag if the exact same price is repeated 4 weeks or more

report = []


def say(text):
    print(text)
    report.append(text)


# 1. load
def load():
    df = pd.read_parquet(IN_FILE)
    say(f"raw records: {len(df):,}")
    df = df[df["periodType"] == "Week"]
    say(f"weekly records: {len(df):,} (monthly and yearly rows dropped)")
    df = df[df["productStage"] == STAGE]
    say(f"weekly ex-packaging records: {len(df):,}")
    df = df[(df["begin_date"] >= START) & (df["begin_date"] <= END)]
    say(f"in study period {START.date()} to {END.date()}: {len(df):,}")
    return df


# 2. tidy
def tidy(df):
    out = pd.DataFrame({
        "country": df["memberStateCode"],
        "product": df["product"],
        "variety": df["variety"],
        "date": df["begin_date"],
        "price": df["price_eur"] / 100,  # €/100 kg -> €/kg
    })
    out["series_id"] = out["country"] + " | " + out["variety"]
    # all dates should be mondays (start of the week)
    not_monday = (out["date"].dt.day_of_week != 0).sum()
    say(f"dates that are not a monday: {not_monday}")
    say(f"unit: all prices were €/100 kg, converted to €/kg")
    return out


# 3. duplicates
def remove_duplicates(df):
    key = ["series_id", "date"]
    dup = df.duplicated(key, keep=False)
    # are there duplicates with a different price? then something is really wrong
    conflicts = df[dup].groupby(key)["price"].nunique().gt(1).sum()
    before = len(df)
    df = df.groupby(key, as_index=False).agg(
        country=("country", "first"), product=("product", "first"),
        variety=("variety", "first"), price=("price", "mean"))
    say(f"duplicates removed: {before - len(df):,} (of which with a different price: {conflicts})")
    return df


def run_lengths(mask):
    # for every row: how long is the run of True values it belongs to (0 if False)
    groups = (mask != mask.shift()).cumsum()
    sizes = mask.groupby(groups).transform("size")
    return sizes.where(mask, 0)


# 4. weekly grid and gaps
def weekly_grid(df):
    weeks = pd.date_range(START, END, freq="W-MON")
    parts = []
    for sid, g in df.groupby("series_id"):
        s = g.set_index("date")["price"].reindex(weeks)
        missing = s.isna()
        gap_len = run_lengths(missing)
        # inside = there is a real price before and after the gap
        inside = s.ffill().notna() & s.bfill().notna()
        to_fill = missing & inside & (gap_len <= MAX_FILL)
        filled = s.interpolate(limit_area="inside").where(to_fill, s)
        part = pd.DataFrame({
            "series_id": sid,
            "country": g["country"].iloc[0],
            "product": g["product"].iloc[0],
            "variety": g["variety"].iloc[0],
            "date": weeks,
            "price": filled.values,
            "is_missing": missing.values,
            "is_filled": to_fill.values,
            "gap_len": gap_len.values,
        })
        parts.append(part)
    out = pd.concat(parts, ignore_index=True)
    say(f"series: {out['series_id'].nunique()}, weeks per series: {len(weeks)}")
    say(f"missing weeks: {out['is_missing'].sum():,}, filled (gap <= {MAX_FILL} weeks): {out['is_filled'].sum():,}")
    return out


# 5. flags
def add_flags(df):
    df = df.sort_values(["series_id", "date"]).copy()
    flags_out, flags_stale = [], []
    for sid, g in df.groupby("series_id", sort=False):
        p = g["price"]
        # normal level = median of the 13 weeks before and after (about 6 months)
        level = p.rolling(27, center=True, min_periods=5).median()
        ratio = p / level
        flags_out.append((ratio > OUTLIER_FACTOR) | (ratio < 1 / OUTLIER_FACTOR))
        # stale = same price as the week before, and this happens STALE_WEEKS times in a row
        # (4 equal prices in a row = 3 times "same as last week")
        same = p.diff().eq(0) & ~g["is_filled"]
        run = run_lengths(same)
        flags_stale.append(same & (run >= STALE_WEEKS - 1))
    df["is_outlier"] = pd.concat(flags_out).reindex(df.index).fillna(False)
    df["is_stale"] = pd.concat(flags_stale).reindex(df.index).fillna(False)
    say(f"flagged as possible outlier (>{OUTLIER_FACTOR}x or <1/{OUTLIER_FACTOR} of local level): {df['is_outlier'].sum():,}")
    say(f"flagged as stale (same price {STALE_WEEKS}+ weeks in a row): {df['is_stale'].sum():,}")
    return df


# 6. summary per series
def summary(df):
    rows = []
    for sid, g in df.groupby("series_id"):
        observed = g.loc[~g["is_missing"], "price"]
        real_gaps = g["is_missing"] & ~g["is_filled"]
        rows.append({
            "series_id": sid,
            "country": g["country"].iloc[0],
            "product": g["product"].iloc[0],
            "variety": g["variety"].iloc[0],
            "first_week": g.loc[~g["is_missing"], "date"].min().date(),
            "last_week": g.loc[~g["is_missing"], "date"].max().date(),
            "weeks_total": len(g),
            "weeks_observed": int((~g["is_missing"]).sum()),
            "weeks_missing": int(g["is_missing"].sum()),
            "missing_pct": round(100 * g["is_missing"].mean(), 1),
            "weeks_filled": int(g["is_filled"].sum()),
            "weeks_still_empty": int(real_gaps.sum()),
            "longest_gap": int(g["gap_len"].max()),
            "outliers": int(g["is_outlier"].sum()),
            "stale_pct": round(100 * g.loc[~g["is_missing"], "is_stale"].mean(), 1),
            "mean": round(observed.mean(), 3),
            "median": round(observed.median(), 3),
            "min": round(observed.min(), 3),
            "max": round(observed.max(), 3),
            "std": round(observed.std(), 3),
            "cv": round(observed.std() / observed.mean(), 2),
        })
    return pd.DataFrame(rows).sort_values("missing_pct")


def main():
    df = load()
    df = tidy(df)
    df = remove_duplicates(df)
    df = weekly_grid(df)
    df = add_flags(df)

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_FILE, index=False)

    summ = summary(df)
    summ.to_csv(TABLES / "series_summary.csv", index=False)
    good = summ[summ["missing_pct"] <= 20]
    say(f"series with max 20% missing weeks: {len(good)} "
        f"({good['product'].nunique()} products, {good['country'].nunique()} countries)")

    (TABLES / "cleaning_report.txt").write_text("\n".join(report) + "\n")
    print(f"\nsaved {OUT_FILE.relative_to(ROOT)}, results/tables/series_summary.csv, results/tables/cleaning_report.txt")


if __name__ == "__main__":
    main()
