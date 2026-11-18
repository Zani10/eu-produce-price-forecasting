# Make the test set: about 100 buyer questions in 5 categories, with the correct answer computed from our data.
#
# Answer types:
#   number       one value in euro per kg (correct if within 0.01 or 1% of the true value)
#   multi_number several values that must all be in the answer
#   item         the right country or product (e.g. "Spain")
#
# Output: results/tables/rq4_questions.csv

import json
import random
from pathlib import Path

import pandas as pd

from src.assistant.tools import COUNTRIES, forecasts, prices

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "tables" / "rq4_questions.csv"

NAMES = {"Egg Plants": "aubergines"}  # how we write the product in a question
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
          "October", "November", "December"]
rows = []


def pname(product):
    return NAMES.get(product, product.lower())


def add(category, question, kind, truth, note=""):
    rows.append({"id": len(rows) + 1, "category": category, "question": question, "kind": kind,
                 "truth": json.dumps(truth), "note": note})


def last_price(sid):
    return prices()[prices()["series_id"] == sid].iloc[-1]["price"]


def fc(sid, h):
    f = forecasts()
    return f[(f["series_id"] == sid) & (f["h"] == h)].iloc[0]["q50"]


def main():
    rng = random.Random(42)
    p = prices()
    series = p[["series_id", "country", "product"]].drop_duplicates().reset_index(drop=True)
    ser = list(series.itertuples(index=False))

    # 1. historical prices (20)
    for s in rng.sample(ser, 12):
        year = rng.choice([2024, 2025])
        month = rng.randint(1, 12)
        g = p[(p["series_id"] == s.series_id) & (p["date"].dt.year == year) & (p["date"].dt.month == month)]
        if g.empty:
            continue
        add("historical", f"What was the average price of {pname(s.product)} in {COUNTRIES[s.country]} in "
            f"{MONTHS[month - 1]} {year}?", "number", round(g["price"].mean(), 4))
    for s in rng.sample(ser, 20 - len(rows)):
        year = rng.choice([2023, 2024, 2025])
        g = p[(p["series_id"] == s.series_id) & (p["date"].dt.year == year)]
        add("historical", f"What was the highest weekly price of {pname(s.product)} in {COUNTRIES[s.country]} "
            f"in {year}?", "number", round(g["price"].max(), 4))

    # 2. current reported prices (20), the first one is the teacher's example
    be_tom = series[(series["country"] == "BE") & (series["product"] == "Tomatoes")].iloc[0]
    add("current", "What is the current reported average price of tomatoes in Belgium?", "number",
        round(last_price(be_tom.series_id), 4), "teacher example")
    others = [s for s in ser if s.series_id != be_tom.series_id]
    for s in rng.sample(others, 19):
        add("current", f"What is the latest reported price of {pname(s.product)} in {COUNTRIES[s.country]}?",
            "number", round(last_price(s.series_id), 4))

    # 3. short-term forecasts (20)
    add("forecast", "What is the expected tomato price in Belgium in the coming four weeks?", "number",
        round(fc(be_tom.series_id, 4), 4), "teacher example (value 4 weeks ahead)")
    add("forecast", "How does the forecast for next week compare with the last reported price for tomatoes "
        "in Belgium?", "number", round(abs(fc(be_tom.series_id, 1) - last_price(be_tom.series_id)), 4),
        "teacher example (size of the change in euro per kg)")
    for s in rng.sample(others, 12):
        h = rng.randint(1, 8)
        add("forecast", f"What is the expected price of {pname(s.product)} in {COUNTRIES[s.country]} "
            f"{h} week{'s' if h > 1 else ''} from now?", "number", round(fc(s.series_id, h), 4))
    for s in rng.sample(others, 6):
        add("forecast", f"How much is the price of {pname(s.product)} in {COUNTRIES[s.country]} expected to "
            f"change next week compared with the last reported price, in euro per kg?", "number",
            round(abs(fc(s.series_id, 1) - last_price(s.series_id)), 4))

    # 4. comparisons between countries (20)
    start = len(rows)
    multi = series.groupby("product").filter(lambda g: len(g) >= 2)
    tom = series[series["product"] == "Tomatoes"].set_index("country")["series_id"]
    add("countries", "How does the forecast for tomatoes in 4 weeks vary between Belgium, Spain and the "
        "Netherlands?", "multi_number", [round(fc(tom[c], 4), 4) for c in ["BE", "ES", "NL"]], "teacher example")
    for product, g in multi.groupby("product"):
        countries = [COUNTRIES[c] for c in g["country"]]
        h = rng.randint(2, 8)
        vals = {COUNTRIES[c]: fc(sid, h) for c, sid in zip(g["country"], g["series_id"])}
        add("countries", f"Which country has the lowest expected price for {pname(product)} in {h} weeks: "
            f"{', '.join(countries)}?", "item", {"answer": min(vals, key=vals.get), "options": countries})
        vals = {COUNTRIES[c]: last_price(sid) for c, sid in zip(g["country"], g["series_id"])}
        if len(rows) - start < 20:
            add("countries", f"Which country has the highest current reported price for {pname(product)}: "
                f"{', '.join(countries)}?", "item", {"answer": max(vals, key=vals.get), "options": countries})
    while len(rows) - start < 20:
        product = rng.choice(sorted(multi["product"].unique()))
        g = multi[multi["product"] == product]
        h = rng.randint(1, 8)
        countries = [COUNTRIES[c] for c in g["country"]]
        vals = {COUNTRIES[c]: fc(sid, h) for c, sid in zip(g["country"], g["series_id"])}
        add("countries", f"In which country is {pname(product)} expected to be most expensive {h} weeks from now: "
            f"{', '.join(countries)}?", "item", {"answer": max(vals, key=vals.get), "options": countries})

    # 5. comparisons between products (20)
    f = forecasts()
    for h, word in [(4, "movement"), (8, "movement"), (4, "increase"), (4, "drop")]:
        g = f[f["h"] == h].copy()
        g["last"] = g["series_id"].map(last_price)
        g["chg"] = (g["q50"] - g["last"]) / g["last"]
        pick = {"movement": g["chg"].abs().idxmax(), "increase": g["chg"].idxmax(), "drop": g["chg"].idxmin()}[word]
        r = g.loc[pick]
        text = {"movement": f"Which of the selected products shows the greatest expected price movement in the "
                            f"next {h} weeks?",
                "increase": f"Which product and country has the biggest expected price increase in the next {h} weeks?",
                "drop": f"Which product and country has the biggest expected price drop in the next {h} weeks?"}[word]
        add("products", text, "item", {"answer": f"{COUNTRIES[r['country']]} {pname(r['product'])}", "options": []},
            "teacher example" if (word == "movement" and h == 4) else "")
    for _ in range(10):
        c = rng.choice(["ES", "IT", "BE"])
        g = series[series["country"] == c]
        a, b = rng.sample(list(g.itertuples(index=False)), 2)
        pa, pb = last_price(a.series_id), last_price(b.series_id)
        add("products", f"In {COUNTRIES[c]}, which is cheaper right now: {pname(a.product)} or {pname(b.product)}?",
            "item", {"answer": pname(a.product if pa < pb else b.product),
                     "options": [pname(a.product), pname(b.product)]})
    for _ in range(6):
        c = rng.choice(["ES", "IT"])
        g = series[series["country"] == c]
        a, b = rng.sample(list(g.itertuples(index=False)), 2)
        add("products", f"In {COUNTRIES[c]}, what is the difference in euro per kg between the latest prices of "
            f"{pname(a.product)} and {pname(b.product)}?", "number",
            round(abs(last_price(a.series_id) - last_price(b.series_id)), 4))

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(df["category"].value_counts().to_dict(), "total", len(df))


if __name__ == "__main__":
    main()
