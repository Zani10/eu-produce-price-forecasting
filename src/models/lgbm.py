# LightGBM: a machine learning model made of many small decision trees (mlforecast library).
#
# - One model for all 29 series together (a "global" model). Each series is first scaled to its own
#   average and spread, so cheap and expensive products look alike to the model.
# - Features = the past: price 1, 2, 3, 4, 8, 13 and 52 weeks ago, the average of the last 4 and
#   13 weeks, and the week of the year.
# - Recursive: it predicts next week, then uses that prediction as "last week" to predict the week
#   after, and so on until 8 weeks. One model is enough this way.
# - The 80% range (q10, q90) comes from conformal prediction: the model is tested on the last
#   2 x 8 weeks of the history and the size of its errors there gives the width of the range.

import warnings

import pandas as pd
from lightgbm import LGBMRegressor
from mlforecast import MLForecast
from mlforecast.lag_transforms import RollingMean
from mlforecast.target_transforms import LocalStandardScaler
from mlforecast.utils import PredictionIntervals

# hide library warnings (deprecation notes) so the run output stays readable, they do not change the results
warnings.filterwarnings("ignore")

FREQ = "W-MON"


def week_of_year(dates):
    return dates.isocalendar().week.to_numpy().astype(int)


def lightgbm(hist, h):
    model = LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31, verbose=-1, random_state=42)
    mlf = MLForecast(
        models={"lgbm": model},
        freq=FREQ,
        lags=[1, 2, 3, 4, 8, 13, 52],
        lag_transforms={1: [RollingMean(window_size=4), RollingMean(window_size=13)]},
        date_features=[week_of_year],
        target_transforms=[LocalStandardScaler()],
    )
    mlf.fit(hist, prediction_intervals=PredictionIntervals(n_windows=2, h=h))
    fc = mlf.predict(h=h, level=[80])
    out = pd.DataFrame({
        "unique_id": fc["unique_id"],
        "ds": fc["ds"],
        "y_pred": fc["lgbm"],
        "q10": fc["lgbm-lo-80"],
        "q50": fc["lgbm"],
        "q90": fc["lgbm-hi-80"],
    })
    return out
