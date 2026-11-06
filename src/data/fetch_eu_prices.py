# Download the fruit and vegetable prices from the EU Agri-food Data Portal.
# API docs: https://agridata.ec.europa.eu/extensions/API_Documentation/fruitandvegetables.html
# No API key needed. I ask one year at a time so the answers stay small.
#
# Output:
#   data/raw/eu_fv_prices_<year>.json   = what the API gives back, not changed
#   data/interim/eu_fv_prices.parquet   = all years together in one table

import json
import time
from pathlib import Path

import pandas as pd
import requests

BASE = "https://api.tech.ec.europa.eu/agrifood/api/fruitAndVegetable/pricesSupplyChain"
ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"


def fetch_year(year, retries=3):
    for attempt in range(retries):
        r = requests.get(BASE, params={"years": year}, timeout=300)
        if r.status_code == 429 or r.status_code >= 500:
            # too many requests or server problem, wait longer every time and try again
            wait = 15 * 2**attempt
            print(f"  got {r.status_code}, waiting {wait}s")
            time.sleep(wait)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"could not get {year} after {retries} tries")


def to_table(records):
    df = pd.DataFrame(records)
    # price comes as text like "€78.17", make it a number
    df["price_eur"] = (
        df["price"].str.replace("€", "", regex=False).str.replace(",", "", regex=False).astype(float)
    )
    df["begin_date"] = pd.to_datetime(df["beginDate"], format="%d/%m/%Y")
    df["end_date"] = pd.to_datetime(df["endDate"], format="%d/%m/%Y")
    # "Apples - Gala" -> "Apples"
    df["product"] = df["variety"].str.split(r" [-–] ", n=1, regex=True).str[0]
    return df


def main(first_year=2000, last_year=2026):
    RAW.mkdir(parents=True, exist_ok=True)
    INTERIM.mkdir(parents=True, exist_ok=True)
    tables = []
    for year in range(first_year, last_year + 1):
        out = RAW / f"eu_fv_prices_{year}.json"
        # old years dont change so I reuse the saved file, the current year I always download again
        if out.exists() and year < last_year:
            records = json.loads(out.read_text())
        else:
            try:
                records = fetch_year(year)
                out.write_text(json.dumps(records))
                time.sleep(5)
            except (requests.exceptions.RequestException, RuntimeError) as e:
                # server is down, use the file we saved before if we have it
                if out.exists():
                    print(f"  could not download {year}, using the saved file ({e})")
                    records = json.loads(out.read_text())
                else:
                    raise
        print(year, len(records))
        if records:
            tables.append(to_table(records))
    df = pd.concat(tables, ignore_index=True)
    df.to_parquet(INTERIM / "eu_fv_prices.parquet", index=False)
    print("total", len(df), "->", INTERIM / "eu_fv_prices.parquet")


if __name__ == "__main__":
    main()
