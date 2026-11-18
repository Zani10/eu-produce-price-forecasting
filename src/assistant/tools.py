# The tools of the assistant. Every tool reads our own data and returns a small dict with the numbers,
# so the language model never has to guess a price. All prices are in euro per kg.
#
# Data:  data/processed/prices_weekly.parquet  (real weekly prices, only our 29 series)
#        results/tables/current_forecasts.csv  (ARIMA forecasts from the last data week, 1 to 8 weeks ahead)

from functools import lru_cache
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PRICES = ROOT / "data" / "processed" / "prices_weekly.parquet"
SAMPLE = ROOT / "results" / "tables" / "sample_series.csv"
FORECASTS = ROOT / "results" / "tables" / "current_forecasts.csv"

SOURCE_PRICES = "EU Agri-food Data Portal, ex-packaging station price, weekly national average"
SOURCE_FORECAST = "ARIMA forecast made at the last data week"

COUNTRIES = {"BE": "Belgium", "NL": "Netherlands", "DE": "Germany", "FR": "France", "ES": "Spain", "IT": "Italy"}

# other words people use for the same product
PRODUCT_WORDS = {
    "tomato": "Tomatoes", "apple": "Apples", "pear": "Pears", "cucumber": "Cucumbers", "pepper": "Peppers",
    "courgette": "Courgettes", "zucchini": "Courgettes", "aubergine": "Egg Plants", "eggplant": "Egg Plants",
    "egg plant": "Egg Plants", "lettuce": "Lettuces", "cauliflower": "Cauliflowers", "lemon": "Lemons",
    "strawberr": "Strawberries",
}


@lru_cache
def prices():
    df = pd.read_parquet(PRICES)
    ids = pd.read_csv(SAMPLE)["series_id"]
    df = df[df["series_id"].isin(ids)]
    # only real reported weeks (no filled or missing weeks)
    df = df[~df["is_missing"]][["series_id", "country", "product", "variety", "date", "price"]]
    return df.sort_values(["series_id", "date"]).reset_index(drop=True)


@lru_cache
def forecasts():
    fc = pd.read_csv(FORECASTS, parse_dates=["origin", "week"])
    info = prices()[["series_id", "country", "product", "variety"]].drop_duplicates()
    return fc.merge(info, on="series_id", how="left")


def find_product(text):
    t = str(text).lower().strip()
    for word, product in PRODUCT_WORDS.items():
        if word in t:
            return product
    return None


def find_country(text):
    t = str(text).lower().strip()
    for code, name in COUNTRIES.items():
        if t in (code.lower(), name.lower()) or name.lower() in t:
            return code
    return None


def valid_series():
    s = prices()[["country", "product"]].drop_duplicates()
    return sorted(f"{COUNTRIES[c]} {p}" for c, p in zip(s["country"], s["product"]))


def get_series(product, country):
    # returns (series_id, variety) or an error dict
    p, c = find_product(product), find_country(country)
    if p is None or c is None:
        return None, {"error": f"unknown product or country: {product}, {country}", "available": valid_series()}
    s = prices()
    s = s[(s["product"] == p) & (s["country"] == c)]
    if s.empty:
        return None, {"error": f"no data for {p} in {COUNTRIES[c]}", "available": valid_series()}
    return s["series_id"].iloc[0], None


def r2(x):
    return None if pd.isna(x) else round(float(x), 2)


def latest(series_id):
    s = prices()[prices()["series_id"] == series_id]
    row = s.iloc[-1]
    return row


def forecast_at(series_id, weeks_ahead):
    fc = forecasts()
    row = fc[(fc["series_id"] == series_id) & (fc["h"] == weeks_ahead)]
    return None if row.empty else row.iloc[0]


# ---------- the 5 tools ----------

def get_latest_price(product, country):
    sid, err = get_series(product, country)
    if err:
        return err
    row = latest(sid)
    return {"product": row["product"], "country": COUNTRIES[row["country"]], "variety": row["variety"],
            "week_starting": str(row["date"].date()), "price_eur_per_kg": r2(row["price"]), "source": SOURCE_PRICES}


def get_price_history(product, country, start_date, end_date):
    sid, err = get_series(product, country)
    if err:
        return err
    s = prices()[prices()["series_id"] == sid]
    s = s[(s["date"] >= pd.Timestamp(start_date)) & (s["date"] <= pd.Timestamp(end_date))]
    if s.empty:
        return {"error": f"no reported prices between {start_date} and {end_date}"}
    weeks = [{"week_starting": str(d.date()), "price": r2(p)} for d, p in zip(s["date"], s["price"])]
    if len(weeks) > 60:
        weeks = weeks[-60:]  # keep the answer small, the statistics below still use all weeks
    return {"product": s["product"].iloc[0], "country": COUNTRIES[s["country"].iloc[0]],
            "weeks_reported": len(s), "average_eur_per_kg": r2(s["price"].mean()),
            "min_eur_per_kg": r2(s["price"].min()), "max_eur_per_kg": r2(s["price"].max()),
            "weekly_prices": weeks, "source": SOURCE_PRICES}


def get_forecast(product, country, weeks_ahead):
    sid, err = get_series(product, country)
    if err:
        return err
    weeks_ahead = int(weeks_ahead)
    if not 1 <= weeks_ahead <= 8:
        return {"error": "forecasts exist only for 1 to 8 weeks ahead"}
    last = latest(sid)
    rows = forecasts()[(forecasts()["series_id"] == sid) & (forecasts()["h"] <= weeks_ahead)]
    out = [{"weeks_ahead": int(r["h"]), "week_starting": str(r["week"].date()), "expected_price": r2(r["q50"]),
            "low_80": r2(r["q10"]), "high_80": r2(r["q90"])} for _, r in rows.iterrows()]
    end = out[-1]["expected_price"]
    return {"product": last["product"], "country": COUNTRIES[last["country"]],
            "last_reported_week": str(last["date"].date()), "last_reported_price": r2(last["price"]),
            "forecast": out, "change_vs_last_price_eur": r2(end - last["price"]),
            "change_vs_last_price_pct": r2(100 * (end - last["price"]) / last["price"]),
            "source": SOURCE_FORECAST}


def compare_countries(product, weeks_ahead):
    p = find_product(product)
    if p is None:
        return {"error": f"unknown product: {product}", "available": valid_series()}
    weeks_ahead = int(weeks_ahead)
    s = prices()[prices()["product"] == p]
    rows = []
    for sid, g in s.groupby("series_id"):
        last = g.iloc[-1]
        f = forecast_at(sid, weeks_ahead)
        rows.append({"country": COUNTRIES[last["country"]], "variety": last["variety"],
                     "last_reported_price": r2(last["price"]), "last_reported_week": str(last["date"].date()),
                     f"expected_price_in_{weeks_ahead}_weeks": None if f is None else r2(f["q50"])})
    rows = sorted(rows, key=lambda r: r[f"expected_price_in_{weeks_ahead}_weeks"] or 999)
    return {"product": p, "weeks_ahead": weeks_ahead, "countries": rows, "source": SOURCE_FORECAST}


def rank_expected_movement(weeks_ahead):
    weeks_ahead = int(weeks_ahead)
    rows = []
    for sid, g in prices().groupby("series_id"):
        last = g.iloc[-1]
        f = forecast_at(sid, weeks_ahead)
        if f is None:
            continue
        rows.append({"product": last["product"], "country": COUNTRIES[last["country"]],
                     "last_reported_price": r2(last["price"]), "expected_price": r2(f["q50"]),
                     "change_pct": r2(100 * (f["q50"] - last["price"]) / last["price"])})
    rows = sorted(rows, key=lambda r: abs(r["change_pct"]), reverse=True)
    return {"weeks_ahead": weeks_ahead, "ranked_by_size_of_change": rows, "source": SOURCE_FORECAST}


TOOLS = {
    "get_latest_price": get_latest_price,
    "get_price_history": get_price_history,
    "get_forecast": get_forecast,
    "compare_countries": compare_countries,
    "rank_expected_movement": rank_expected_movement,
}

# the descriptions the language model sees
SCHEMAS = [
    {"type": "function", "function": {
        "name": "get_latest_price",
        "description": "Latest reported weekly price (euro per kg) of a product in a country.",
        "parameters": {"type": "object", "properties": {
            "product": {"type": "string", "description": "e.g. tomatoes, apples, pears"},
            "country": {"type": "string", "description": "Belgium, Netherlands, Germany, France, Spain or Italy"}},
            "required": ["product", "country"]}}},
    {"type": "function", "function": {
        "name": "get_price_history",
        "description": "Reported weekly prices of a product in a country between two dates, with the average, min and max.",
        "parameters": {"type": "object", "properties": {
            "product": {"type": "string"}, "country": {"type": "string"},
            "start_date": {"type": "string", "description": "YYYY-MM-DD"},
            "end_date": {"type": "string", "description": "YYYY-MM-DD"}},
            "required": ["product", "country", "start_date", "end_date"]}}},
    {"type": "function", "function": {
        "name": "get_forecast",
        "description": "Forecast of the weekly price for the next 1 to 8 weeks, with an 80% range and the change versus the last reported price.",
        "parameters": {"type": "object", "properties": {
            "product": {"type": "string"}, "country": {"type": "string"},
            "weeks_ahead": {"type": "integer", "minimum": 1, "maximum": 8}},
            "required": ["product", "country", "weeks_ahead"]}}},
    {"type": "function", "function": {
        "name": "compare_countries",
        "description": "Last reported price and expected price in N weeks for one product in every country we have, sorted from cheapest to most expensive expected price.",
        "parameters": {"type": "object", "properties": {
            "product": {"type": "string"}, "weeks_ahead": {"type": "integer", "minimum": 1, "maximum": 8}},
            "required": ["product", "weeks_ahead"]}}},
    {"type": "function", "function": {
        "name": "rank_expected_movement",
        "description": "All products and countries ranked by the size of the expected price change (in %) between the last reported price and the forecast N weeks ahead.",
        "parameters": {"type": "object", "properties": {
            "weeks_ahead": {"type": "integer", "minimum": 1, "maximum": 8}},
            "required": ["weeks_ahead"]}}},
]
