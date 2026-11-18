# Weather for the main growing region of every price series (RQ2).
#
# Source: Open-Meteo historical weather API, ERA5 reanalysis (free, no key).
# One location per region, daily values 2014-01-01 to 2026-10-25, made weekly (weeks start on monday,
# same as the prices).
#
# Output: data/raw/weather/<region>.json  = what the API gives back, not changed
#         data/interim/weather_weekly.parquet = one row per region per week
#
# Run:  python -m src.features.weather

import json
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "weather"
OUT = ROOT / "data" / "interim" / "weather_weekly.parquet"

URL = "https://archive-api.open-meteo.com/v1/archive"
START = "2014-01-01"
END = "2026-10-25"
DAILY = ["temperature_2m_mean", "temperature_2m_max", "temperature_2m_min",
         "precipitation_sum", "shortwave_radiation_sum"]

# region name: (latitude, longitude, why)
REGIONS = {
    "almeria": (36.84, -2.46, "biggest greenhouse area of Spain (tomatoes, peppers, cucumbers, courgettes, aubergines)"),
    "murcia": (37.98, -1.13, "Spanish lemons, lettuce and cauliflower (Murcia region)"),
    "lleida": (41.62, 0.62, "main Spanish apple and pear area (Lleida, Catalonia)"),
    "ragusa": (36.93, 14.73, "Sicilian greenhouse vegetables and cherry tomatoes (Ragusa, Pachino)"),
    "catania": (37.50, 15.09, "Sicilian lemons (Catania)"),
    "salerno": (40.60, 14.98, "Italian lettuce (Piana del Sele, Campania)"),
    "bolzano": (46.50, 11.35, "main Italian apple area (South Tyrol)"),
    "sint_truiden": (50.82, 5.19, "Belgian apples and pears (Haspengouw)"),
    "roeselare": (50.95, 3.12, "Belgian field vegetables like cauliflower (West Flanders)"),
    "hoogstraten": (51.40, 4.76, "Belgian strawberries (Hoogstraten auction area)"),
    "sint_katelijne_waver": (51.07, 4.53, "Belgian greenhouse tomatoes (Mechelen area)"),
    "westland": (52.00, 4.20, "main Dutch greenhouse area (Westland)"),
    "betuwe": (51.88, 5.29, "Dutch apples and pears (Betuwe)"),
    "altes_land": (53.53, 9.70, "main German apple area (Altes Land, near Hamburg)"),
    "angers": (47.47, -0.55, "French apples (Val de Loire)"),
    "saint_pol": (48.68, -3.99, "French cauliflower (north Brittany)"),
}

# which region belongs to which series (country + product)
SERIES_REGION = {
    ("BE", "Apples"): "sint_truiden", ("BE", "Pears"): "sint_truiden",
    ("BE", "Cauliflowers"): "roeselare", ("BE", "Strawberries"): "hoogstraten",
    ("BE", "Tomatoes"): "sint_katelijne_waver",
    ("DE", "Apples"): "altes_land",
    ("ES", "Apples"): "lleida", ("ES", "Pears"): "lleida",
    ("ES", "Cauliflowers"): "murcia", ("ES", "Lemons"): "murcia", ("ES", "Lettuces"): "murcia",
    ("ES", "Courgettes"): "almeria", ("ES", "Cucumbers"): "almeria", ("ES", "Egg Plants"): "almeria",
    ("ES", "Peppers"): "almeria", ("ES", "Tomatoes"): "almeria",
    ("FR", "Apples"): "angers", ("FR", "Cauliflowers"): "saint_pol",
    ("IT", "Apples"): "bolzano", ("IT", "Lemons"): "catania", ("IT", "Lettuces"): "salerno",
    ("IT", "Courgettes"): "ragusa", ("IT", "Cucumbers"): "ragusa", ("IT", "Egg Plants"): "ragusa",
    ("IT", "Peppers"): "ragusa", ("IT", "Tomatoes"): "ragusa",
    ("NL", "Apples"): "betuwe", ("NL", "Pears"): "betuwe", ("NL", "Tomatoes"): "westland",
}


def download(region, lat, lon, retries=4):
    out = RAW / f"{region}.json"
    if out.exists():
        return json.loads(out.read_text())
    params = {"latitude": lat, "longitude": lon, "start_date": START, "end_date": END,
              "daily": ",".join(DAILY), "models": "era5", "timezone": "UTC"}
    for attempt in range(retries):
        r = requests.get(URL, params=params, timeout=120)
        if r.status_code == 429 or r.status_code >= 500:
            wait = 15 * 2**attempt
            print(f"  {region}: got {r.status_code}, waiting {wait}s")
            time.sleep(wait)
            continue
        r.raise_for_status()
        out.write_text(r.text)
        time.sleep(1)
        return r.json()
    raise RuntimeError(f"could not get weather for {region}")


def to_weekly(region, js):
    d = pd.DataFrame(js["daily"])
    d["time"] = pd.to_datetime(d["time"])
    # week starts on monday, same as the price weeks
    d["week"] = d["time"] - pd.to_timedelta(d["time"].dt.day_of_week, unit="D")
    w = d.groupby("week").agg(
        tmean=("temperature_2m_mean", "mean"),
        tmax=("temperature_2m_max", "max"),
        tmin=("temperature_2m_min", "min"),
        rain=("precipitation_sum", "sum"),
        sun=("shortwave_radiation_sum", "mean"),
        days=("time", "size"),
    ).reset_index()
    w = w[w["days"] == 7].drop(columns="days")  # only full weeks
    w.insert(0, "region", region)
    return w


def load_weekly():
    if OUT.exists():
        return pd.read_parquet(OUT)
    return main()


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    parts = []
    for region, (lat, lon, _) in REGIONS.items():
        js = download(region, lat, lon)
        w = to_weekly(region, js)
        print(f"{region}: {len(w)} weeks, {w['week'].min().date()} to {w['week'].max().date()}")
        parts.append(w)
    weekly = pd.concat(parts, ignore_index=True)
    weekly.to_parquet(OUT, index=False)
    print("saved", OUT.relative_to(ROOT))
    return weekly


if __name__ == "__main__":
    main()
