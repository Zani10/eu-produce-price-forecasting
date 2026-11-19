# Chronos-2: a pre-trained "foundation model" for time series (Amazon, chronos-forecasting library).
#
# Zero-shot  : we use the model as it is, we do not train it on our prices. It just reads our history
#              and forecasts the next 8 weeks. The model was released in Oct 2025, during our test window,
#              so public EU prices could be in its training data (see src/check_chronos_release.py).
# Fine-tuned : we train the model a little more on OUR data first. To keep it cheap and without leakage,
#              we fine-tune once, only on the data before the first test origin (21 Oct 2024),
#              and then use that same fine-tuned model at every origin.
#
# Both give quantiles, we keep 10%, 50% (= our forecast) and 90% (the 80% range).

import tempfile
import warnings

import pandas as pd
import torch
from chronos import Chronos2Pipeline

warnings.filterwarnings("ignore")

MODEL_ID = "amazon/chronos-2"
QUANTILES = [0.1, 0.5, 0.9]


def device():
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_pipeline():
    return Chronos2Pipeline.from_pretrained(MODEL_ID, device_map=device())


def forecast(pipeline, hist, h):
    fc = pipeline.predict_df(
        hist,
        prediction_length=h,
        quantile_levels=QUANTILES,
        id_column="unique_id",
        timestamp_column="ds",
        target="y",
    )
    out = pd.DataFrame({
        "unique_id": fc["unique_id"],
        "ds": pd.to_datetime(fc["ds"]),
        "y_pred": fc["0.5"],
        "q10": fc["0.1"],
        "q50": fc["0.5"],
        "q90": fc["0.9"],
    })
    return out


def make_chronos_zero_shot():
    state = {"pipe": None}

    def chronos_zero(hist, h):
        if state["pipe"] is None:
            state["pipe"] = load_pipeline()
        return forecast(state["pipe"], hist, h)

    return chronos_zero


def no_dropout(pipeline):
    # the Apple GPU (MPS) can't train the attention layers with dropout, so we set dropout to 0.
    # fit() builds a new model from the config, so we change it in the config.
    pipeline.model.config.dropout_rate = 0.0


def make_chronos_finetuned(num_steps=1000, learning_rate=1e-5, batch_size=32):
    state = {"pipe": None}

    def chronos_ft(hist, h):
        if state["pipe"] is None:
            # first call = first origin: fine-tune only on this history (all before 21 Oct 2024)
            base = load_pipeline()
            no_dropout(base)
            series = [g["y"].to_numpy(dtype="float32") for _, g in hist.groupby("unique_id")]
            # the checkpoint files go to a temporary folder, we don't need to keep them
            state["pipe"] = base.fit(
                inputs=series,
                prediction_length=h,
                num_steps=num_steps,
                learning_rate=learning_rate,
                batch_size=batch_size,
                output_dir=tempfile.mkdtemp(),
            )
        return forecast(state["pipe"], hist, h)

    return chronos_ft
