# Diebold-Mariano test: is model A really more accurate than model B, or is it luck?
#
# We compare the scaled absolute errors (same as MASE) of two models on exactly the same forecasts
# (same series, origin and horizon). d = error of A - error of B.
# If the mean of d is far enough from 0 compared to its uncertainty, the difference is "real".
#
# Forecasts h weeks ahead overlap, so their errors are correlated over time. The test takes this into
# account with the autocovariances of d up to lag h-1, and the Harvey, Leybourne and Newbold (1997)
# small-sample correction. We run the test per horizon. Inside one horizon we first average d over the
# 29 series per origin, so we get one value per origin (25 values) in time order.

import numpy as np
from scipy import stats


def dm_test(d, h):
    # d = loss differences in time order, h = forecast horizon
    d = np.asarray(d, dtype=float)
    d = d[~np.isnan(d)]
    n = len(d)
    mean = d.mean()
    # autocovariances up to lag h-1
    gamma = [np.mean((d[k:] - mean) * (d[:n - k] - mean)) for k in range(h)]
    var = (gamma[0] + 2 * sum(gamma[1:])) / n
    if var <= 0:
        var = gamma[0] / n  # fallback if the sum is not positive
    dm = mean / np.sqrt(var)
    # Harvey, Leybourne and Newbold correction
    correction = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    hln = dm * correction
    p = 2 * stats.t.sf(abs(hln), df=n - 1)
    return hln, p


def holm(pvalues):
    # Holm correction for many tests at the same time
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj
