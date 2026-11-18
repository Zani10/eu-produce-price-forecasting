# Error measures to compare the forecasts.
#
# MASE  = error divided by the error of a simple "same week last year" forecast on the training data.
#         No unit, so cheap and expensive products can be compared. Below 1 = better than that simple forecast.
# MAE   = average absolute error in €/kg (easy to explain to a buyer).
# RMSE  = like MAE but big errors count much more.
# WQL   = weighted quantile loss, scores the forecast range (q10, q50, q90), lower is better.

import numpy as np
import pandas as pd

SEASON = 52


def mase_scale(history, m=SEASON):
    # average error of the "same week last year" forecast inside the training data
    # if the history is too short for that, use "same as last week" (m=1)
    y = np.asarray(history, dtype=float)
    if len(y) < 2 * m:
        m = 1
    diffs = np.abs(y[m:] - y[:-m])
    scale = np.nanmean(diffs)
    if not scale or np.isnan(scale):
        return np.nan
    return scale


def pinball(y, q, tau):
    # quantile loss for one quantile: if the real price is above q we pay tau, below we pay (1 - tau)
    diff = y - q
    return np.maximum(tau * diff, (tau - 1) * diff)


def add_errors(res):
    # res has one row per forecast: y_true, y_pred, q10, q50, q90, scale
    res = res.copy()
    res["abs_err"] = (res["y_true"] - res["y_pred"]).abs()
    res["sq_err"] = (res["y_true"] - res["y_pred"]) ** 2
    res["scaled_err"] = res["abs_err"] / res["scale"]
    ql = 0
    for col, tau in [("q10", 0.1), ("q50", 0.5), ("q90", 0.9)]:
        ql = ql + pinball(res["y_true"], res[col], tau)
    res["qloss"] = ql  # NaN when a model has no range
    # is the real price inside the 80% range (between q10 and q90)? 1 = yes, 0 = no
    inside = (res["y_true"] >= res["q10"]) & (res["y_true"] <= res["q90"])
    res["in_80"] = inside.astype(float).where(res["q10"].notna())
    return res


def summarize(res, by):
    # one row per group with all measures
    g = res.groupby(by)
    out = pd.DataFrame({
        "n": g.size(),
        "MASE": g["scaled_err"].mean(),
        "MAE": g["abs_err"].mean(),
        "RMSE": np.sqrt(g["sq_err"].mean()),
        # WQL: total quantile loss / total real prices, divided by 3 quantiles (lower is better)
        "WQL": 2 * g["qloss"].sum(min_count=1) / g["y_true"].sum() / 3,
        "coverage_80": g["in_80"].mean(),
    })
    return out.round(4).reset_index()
