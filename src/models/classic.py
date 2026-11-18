# Simple and classic forecasting models (statsforecast library).
#
# Naive          : next weeks = the last known price
# Seasonal Naive : next weeks = the price of the same week last year (52 weeks ago)
# ETS (STL)      : first remove the yearly pattern (STL), then ETS (exponential smoothing) on the rest,
#                  then add the yearly pattern back. Plain ETS can't handle a 52-week season well.
# ARIMA (Fourier): ARIMA with sine/cosine waves as the yearly pattern ("dynamic harmonic regression").
#                  Normal seasonal ARIMA with 52 weeks was far too slow (more than 5 minutes for one series).
#
# Every function gets the history (unique_id, ds, y) and returns unique_id, ds, y_pred, q10, q50, q90.
# q10 and q90 come from the 80% prediction interval of the model.

import warnings

import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import MSTL, AutoARIMA, AutoETS, Naive, SeasonalNaive

warnings.filterwarnings("ignore")

FREQ = "W-MON"
SEASON = 52
FOURIER_K = 4  # number of sine/cosine pairs for the yearly wave


def to_output(fc, col):
    out = pd.DataFrame({
        "unique_id": fc["unique_id"],
        "ds": fc["ds"],
        "y_pred": fc[col],
        "q10": fc[f"{col}-lo-80"],
        "q50": fc[col],
        "q90": fc[f"{col}-hi-80"],
    })
    return out


def run_statsforecast(model, col, hist, h):
    sf = StatsForecast(models=[model], freq=FREQ, n_jobs=-1)
    fc = sf.forecast(df=hist, h=h, level=[80])
    return to_output(fc, col)


def naive(hist, h):
    return run_statsforecast(Naive(), "Naive", hist, h)


def seasonal_naive(hist, h):
    return run_statsforecast(SeasonalNaive(season_length=SEASON), "SeasonalNaive", hist, h)


def ets(hist, h):
    model = MSTL(season_length=SEASON, trend_forecaster=AutoETS(model="ZZN"))
    return run_statsforecast(model, "MSTL", hist, h)


def fourier(dates, k=FOURIER_K):
    # sine and cosine waves of 1 year, 1/2 year, 1/3 year ... based on the day of the year
    t = dates.dt.dayofyear.to_numpy() / 365.25
    cols = {}
    for i in range(1, k + 1):
        cols[f"sin{i}"] = np.sin(2 * np.pi * i * t)
        cols[f"cos{i}"] = np.cos(2 * np.pi * i * t)
    return pd.DataFrame(cols, index=dates.index)


def arima(hist, h):
    train = pd.concat([hist, fourier(hist["ds"])], axis=1)
    # the future dates also need their sine/cosine values
    last = hist.groupby("unique_id")["ds"].max()
    future = []
    for uid, d in last.items():
        ds = pd.Series(pd.date_range(d + pd.Timedelta(weeks=1), periods=h, freq=FREQ))
        part = fourier(ds)
        part.insert(0, "ds", ds.values)
        part.insert(0, "unique_id", uid)
        future.append(part)
    future = pd.concat(future, ignore_index=True)

    sf = StatsForecast(models=[AutoARIMA(season_length=1)], freq=FREQ, n_jobs=-1)
    fc = sf.forecast(df=train, h=h, X_df=future, level=[80])
    return to_output(fc, "AutoARIMA")
