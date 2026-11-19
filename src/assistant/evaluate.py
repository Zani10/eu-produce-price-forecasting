# Test the assistant: every question is asked twice, with tools and without tools (same model, same instructions).
# Then every answer is scored with simple, clear rules:
#   correct    = the right number (within 0.01 euro or 1%) or the right country/product
#   wrong      = a concrete answer (a number or a choice) that is not right  -> we count this as a hallucination
#   no_answer  = the assistant says it doesn't know / has no data, or gives no concrete answer
# McNemar test: is the difference in correct answers between the two conditions real or luck?
#
# Output: results/tables/rq4_answers.csv, results/tables/rq4_summary.csv, results/figures/rq4_accuracy.png

import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib
import pandas as pd
from scipy.stats import binomtest

from src.assistant.agent import MODEL, ask

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "results" / "tables"
FIG = ROOT / "results" / "figures"
QUESTIONS = TABLES / "rq4_questions.csv"
ANSWERS = TABLES / "rq4_answers.csv"

PRICE_IN, PRICE_OUT = 0.40, 1.60  # dollar per 1 million tokens for the model we use

COUNTRY_WORDS = ["Belgium", "Netherlands", "Germany", "France", "Spain", "Italy"]
PRODUCT_WORDS = {"tomato": "tomatoes", "apple": "apples", "pear": "pears", "cucumber": "cucumbers",
                 "pepper": "peppers", "courgette": "courgettes", "zucchini": "courgettes",
                 "aubergine": "aubergines", "eggplant": "aubergines", "egg plant": "aubergines",
                 "lettuce": "lettuces", "cauliflower": "cauliflowers", "lemon": "lemons", "strawberr": "strawberries"}
REFUSAL = ["don't have", "do not have", "not available", "no data", "cannot provide", "can't provide",
           "unable to", "don't know", "do not know", "not able to", "no access", "real-time", "i'm sorry", "sorry"]


# ---------- scoring ----------

def numbers(text):
    t = text.replace("\u202f", " ")   # the model sometimes writes a narrow space inside numbers, make it a normal space
    t = re.sub(r"\d{4}-\d{2}-\d{2}", " ", t)                                   # dates
    t = re.sub(r"\b\d+\s*(weeks?|days?|months?|years?)\b", " ", t, flags=re.I)  # "4 weeks"
    t = re.sub(r"\bweek\s*\d+\b", " ", t, flags=re.I)                          # "week 23"
    t = re.sub(r"\b(19|20)\d{2}\b", " ", t)                                    # years
    found = re.findall(r"\d+(?:[.,]\d+)?", t)
    return [float(x.replace(",", ".")) for x in found]


def close(a, b):
    return abs(a - b) <= max(0.011, 0.01 * abs(b))


def first_sentence(text):
    return re.split(r"(?<=[.!?])\s|\n", text.strip(), maxsplit=1)[0]


def product_in(text):
    t = text.lower()
    hits = [(t.find(w), p) for w, p in PRODUCT_WORDS.items() if w in t]
    return [p for _, p in sorted(hits)]


def country_in(text):
    hits = [(text.find(c), c) for c in COUNTRY_WORDS if c in text]
    return [c for _, c in sorted(hits)]


def refuses(text):
    t = text.lower()
    return any(r in t for r in REFUSAL)


def score(kind, truth, answer):
    truth = json.loads(truth)
    if kind == "number":
        nums = numbers(answer)
        if any(close(n, truth) for n in nums):
            return "correct"
        return "wrong" if nums else "no_answer"
    if kind == "multi_number":
        nums = numbers(answer)
        if all(any(close(n, t) for n in nums) for t in truth):
            return "correct"
        return "wrong" if nums else "no_answer"
    # item: a refusal is never a choice, even if it names the right country ("I don't have data for Spain")
    if refuses(answer):
        return "no_answer"
    # look at what is named first in the first sentence
    first = first_sentence(answer)
    want = truth["answer"]
    if not truth["options"]:  # product + country, e.g. "Italy cucumbers"
        country, product = want.split(" ", 1)
        named_c, named_p = country_in(first), product_in(first)
        if not named_c and not named_p:
            return "no_answer"
        ok = bool(named_c) and named_c[0] == country and bool(named_p) and named_p[0] == product
        return "correct" if ok else "wrong"
    if want in COUNTRY_WORDS:
        named = country_in(first)
    else:
        named = product_in(first)
    if not named:
        return "no_answer"
    return "correct" if named[0] == want else "wrong"


# ---------- run ----------

def run_one(row):
    out = {}
    for cond, use in [("with_tools", True), ("without_tools", False)]:
        r = ask(row["question"], use_tools=use)
        out[f"answer_{cond}"] = r["answer"]
        out[f"tools_used_{cond}"] = json.dumps(r["tool_calls"])
        out[f"tokens_in_{cond}"] = r["usage"]["input"]
        out[f"tokens_out_{cond}"] = r["usage"]["output"]
    return out


def run_all():
    q = pd.read_csv(QUESTIONS)
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(run_one, [r for _, r in q.iterrows()]))
    ans = pd.concat([q, pd.DataFrame(results)], axis=1)
    ans.to_csv(ANSWERS, index=False)
    return ans


def summarize(ans):
    for cond in ["with_tools", "without_tools"]:
        ans[f"score_{cond}"] = [score(k, t, str(a)) for k, t, a in
                                zip(ans["kind"], ans["truth"], ans[f"answer_{cond}"].fillna(""))]
        # does the answer give a specific price at all? (without tools such a price can't come from our data)
        ans[f"gives_price_{cond}"] = [len(numbers(str(a))) > 0 for a in ans[f"answer_{cond}"].fillna("")]
    ans.to_csv(ANSWERS, index=False)

    rows = []
    for cond in ["with_tools", "without_tools"]:
        for cat, g in [("all", ans)] + list(ans.groupby("category")):
            s = g[f"score_{cond}"]
            rows.append({"condition": cond, "category": cat, "n": len(g),
                         "correct_pct": round(100 * (s == "correct").mean(), 1),
                         "wrong_hallucination_pct": round(100 * (s == "wrong").mean(), 1),
                         "no_answer_pct": round(100 * (s == "no_answer").mean(), 1),
                         "gives_a_price_pct": round(100 * g[f"gives_price_{cond}"].mean(), 1)})
    summ = pd.DataFrame(rows)

    a = ans["score_with_tools"] == "correct"
    b = ans["score_without_tools"] == "correct"
    only_tools, only_no_tools = int((a & ~b).sum()), int((~a & b).sum())
    p = binomtest(min(only_tools, only_no_tools), only_tools + only_no_tools, 0.5).pvalue if (only_tools + only_no_tools) else 1.0
    tin = ans[["tokens_in_with_tools", "tokens_in_without_tools"]].sum().sum()
    tout = ans[["tokens_out_with_tools", "tokens_out_without_tools"]].sum().sum()
    cost = tin / 1e6 * PRICE_IN + tout / 1e6 * PRICE_OUT
    extra = pd.DataFrame([{"condition": "mcnemar", "category": "all", "n": len(ans),
                           "correct_pct": None, "wrong_hallucination_pct": None, "no_answer_pct": None,
                           "note": f"model {MODEL}; correct only with tools {only_tools}, only without tools {only_no_tools}; "
                                   f"exact McNemar p = {p:.2e}; tokens in {tin}, out {tout}, cost about ${cost:.2f}"}])
    summ = pd.concat([summ, extra.dropna(axis=1, how="all")], ignore_index=True)
    summ.to_csv(TABLES / "rq4_summary.csv", index=False)

    plot = summ[summ["condition"] != "mcnemar"].pivot(index="category", columns="condition", values="correct_pct")
    plot = plot.loc[["all", "current", "historical", "forecast", "countries", "products"]]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = range(len(plot))
    ax.bar([i - 0.2 for i in x], plot["with_tools"], width=0.4, label="with tools")
    ax.bar([i + 0.2 for i in x], plot["without_tools"], width=0.4, label="without tools")
    ax.set_xticks(list(x), plot.index)
    ax.set_ylabel("% correct answers")
    ax.set_ylim(0, 105)
    ax.set_title(f"Assistant answers on 100 buyer questions ({MODEL})")
    ax.legend()
    for i, (w, wo) in enumerate(zip(plot["with_tools"], plot["without_tools"])):
        ax.text(i - 0.2, w + 1, f"{w:.0f}", ha="center", fontsize=8)
        ax.text(i + 0.2, wo + 1, f"{wo:.0f}", ha="center", fontsize=8)
    fig.savefig(FIG / "rq4_accuracy.png", dpi=110, bbox_inches="tight")
    print(summ.to_string(index=False))


if __name__ == "__main__":
    import sys
    answers = pd.read_csv(ANSWERS) if (ANSWERS.exists() and "--score-only" in sys.argv) else run_all()
    summarize(answers)
