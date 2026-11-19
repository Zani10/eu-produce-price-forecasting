# LightGBM with and without weather, for RQ2.
#
# This is a "direct" LightGBM: one global model for all 29 series and all horizons. Every training row is
# (series, week t, horizon h) and the target is the price in week t + h. The horizon h is one of the features.
# Why direct and not recursive like src/models/lgbm.py: weather is only known up to the forecast origin, so the
# same weather features are used for every horizon. With a direct model that is simple and clean, because the
# model never needs weather of future weeks.
#
# No leakage:
# - every feature of week t only uses data of week t and before (prices and weather)
# - at an origin, the model is trained only on rows where the target week t + h is not after the origin
# - so at the origin it has never seen a price or weather value from after the origin
#
# Prices are divided by the series' "level" (average of the last 13 weeks at week t), so cheap and expensive
# products look the same to the model. The prediction is multiplied back by the level at the origin.
#
# Four versions:
#   no weather : price features only (the fair baseline for this experiment)
#   weather    : price features + weather known at the origin (lags, averages, anomalies)
#   weather-light: price features + only 3 weather features (temperature and rain anomaly, coldest night)
#   oracle     : price features + the REAL weather of the target weeks. This uses the future, so it is only a
#                look-back analysis ("how much could perfect weather knowledge help?"), not a real forecast.

import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from src.features.weather import SERIES_REGION, load_weekly

# hide library warnings (deprecation notes) so the run output stays readable, they do not change the results
warnings.filterwarnings("ignore")

H = 8
CLIMATE_YEARS = (2014, 2019)  # fixed "normal weather" reference, all before the test period (starts oct 2024)

PRICE_FEATURES = ["h", "woy_target", "product_code", "country_code",
                  "lag0", "lag1", "lag2", "lag3", "lag7", "lag12", "last_year_target", "roll4", "momentum4"]
WEATHER_FEATURES = ["w_tmean_1", "w_tmean_4", "w_tmean_13", "w_tmin_4", "w_tmax_4", "w_rain_4", "w_rain_13",
                    "w_sun_4", "a_tmean_4", "a_rain_4", "a_sun_4"]
# small set as a check: maybe 11 weather features are too many for our amount of data
WEATHER_LIGHT = ["a_tmean_4", "a_rain_4", "w_tmin_4"]
ORACLE_FEATURES = ["o_tmean", "o_tmin", "o_rain", "o_sun", "o_tmean_path", "o_tmin_path", "o_rain_path",
                   "o_sun_path", "o_a_tmean_path", "o_a_rain_path"]


def weather_features(weather):
    # weather features per region and week t, using only week t and before
    w = weather.sort_values(["region", "week"]).copy()
    w["woy"] = w["week"].dt.isocalendar().week.astype(int).clip(upper=52)
    clim = w[w["week"].dt.year.between(*CLIMATE_YEARS)].groupby(["region", "woy"])[["tmean", "rain", "sun"]].mean()
    clim.columns = ["c_tmean", "c_rain", "c_sun"]
    w = w.join(clim, on=["region", "woy"])
    g = w.groupby("region")
    out = pd.DataFrame({"region": w["region"], "week": w["week"]})
    out["w_tmean_1"] = w["tmean"]
    out["w_tmean_4"] = g["tmean"].transform(lambda s: s.rolling(4).mean())
    out["w_tmean_13"] = g["tmean"].transform(lambda s: s.rolling(13).mean())
    out["w_tmin_4"] = g["tmin"].transform(lambda s: s.rolling(4).min())
    out["w_tmax_4"] = g["tmax"].transform(lambda s: s.rolling(4).max())
    out["w_rain_4"] = g["rain"].transform(lambda s: s.rolling(4).sum())
    out["w_rain_13"] = g["rain"].transform(lambda s: s.rolling(13).sum())
    out["w_sun_4"] = g["sun"].transform(lambda s: s.rolling(4).mean())
    # anomaly = this period minus the normal value for the same weeks of the year
    out["a_tmean_4"] = out["w_tmean_4"] - g["c_tmean"].transform(lambda s: s.rolling(4).mean())
    out["a_rain_4"] = out["w_rain_4"] - g["c_rain"].transform(lambda s: s.rolling(4).sum())
    out["a_sun_4"] = out["w_sun_4"] - g["c_sun"].transform(lambda s: s.rolling(4).mean())
    # raw weekly values + normals, needed for the oracle
    for c in ["tmean", "tmin", "rain", "sun", "c_tmean", "c_rain"]:
        out[c] = w[c].values
    return out


def build_table(data, sample):
    # data = output of src.evaluation.backtest.load_data(): unique_id, ds, y_true, y_hist
    # returns one row per (series, week t, horizon h) with all features and the target
    d = data.sort_values(["unique_id", "ds"]).copy()
    info = sample.set_index("series_id")
    d["product"] = d["unique_id"].map(info["product"])
    d["country"] = d["unique_id"].map(info["country"])
    d["region"] = [SERIES_REGION[(c, p)] for c, p in zip(d["country"], d["product"])]
    g = d.groupby("unique_id")["y_hist"]
    d["level"] = g.transform(lambda s: s.rolling(13, min_periods=4).mean())
    for k in [0, 1, 2, 3, 7, 12]:
        d[f"lag{k}"] = g.shift(k) / d["level"]
    d["roll4"] = g.transform(lambda s: s.rolling(4).mean()) / d["level"]
    d["momentum4"] = d["y_hist"] / g.shift(4)

    wf = weather_features(load_weekly())
    d = d.merge(wf, left_on=["region", "ds"], right_on=["region", "week"], how="left").drop(columns="week")

    # oracle: real weather of each future week, and of the whole path t+1 .. t+h
    wk = wf.set_index(["region", "week"])[["tmean", "tmin", "rain", "sun", "c_tmean", "c_rain"]]

    rows = []
    for h in range(1, H + 1):
        r = d.copy()
        r["h"] = h
        r["target_ds"] = r["ds"] + pd.Timedelta(weeks=h)
        r["woy_target"] = r["target_ds"].dt.isocalendar().week.astype(int)
        # price of the same week last year as the target (52 - h weeks before t, so known at t)
        r["last_year_target"] = r.groupby("unique_id")["y_hist"].shift(52 - h) / r["level"]
        nxt = r.groupby("unique_id")["y_true"].shift(-h)
        r["target"] = nxt / r["level"]
        r["y_true_target"] = nxt
        # oracle features
        idx = pd.MultiIndex.from_arrays([r["region"], r["target_ds"]])
        t = wk.reindex(idx)
        r["o_tmean"], r["o_tmin"], r["o_rain"], r["o_sun"] = (t["tmean"].values, t["tmin"].values,
                                                             t["rain"].values, t["sun"].values)
        path = []
        for j in range(1, h + 1):
            pj = wk.reindex(pd.MultiIndex.from_arrays([r["region"], r["ds"] + pd.Timedelta(weeks=j)]))
            path.append(pj.reset_index(drop=True))
        p = pd.concat(path, keys=range(h)).groupby(level=1)
        r["o_tmean_path"] = p["tmean"].mean().values
        r["o_tmin_path"] = p["tmin"].min().values
        r["o_rain_path"] = p["rain"].sum().values
        r["o_sun_path"] = p["sun"].mean().values
        r["o_a_tmean_path"] = r["o_tmean_path"] - p["c_tmean"].mean().values
        r["o_a_rain_path"] = r["o_rain_path"] - p["c_rain"].sum().values
        rows.append(r)
    table = pd.concat(rows, ignore_index=True)
    table["product_code"] = table["product"].astype("category").cat.codes
    table["country_code"] = table["country"].astype("category").cat.codes
    return table


def make_model(table, features, seed=42):
    # returns a function in the shape the backtest wants: fn(history, h) -> forecasts
    def fn(hist, h):
        origin = hist["ds"].max()
        # train only on rows whose target week is not after the origin
        train = table[(table["target_ds"] <= origin) & table["target"].notna() & table["level"].notna()]
        model = LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=50,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                              n_jobs=2, verbose=-1, random_state=seed)
        model.fit(train[features], train["target"])
        test = table[(table["ds"] == origin) & (table["h"] <= h)]
        pred = model.predict(test[features]) * test["level"].values
        return pd.DataFrame({"unique_id": test["unique_id"].values, "ds": test["target_ds"].values,
                             "y_pred": pred, "q10": np.nan, "q50": pred, "q90": np.nan})
    return fn
