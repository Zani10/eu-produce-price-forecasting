# Run the backtest for the deep learning model (N-BEATS) and Chronos-2 (zero-shot and fine-tuned),
# with exactly the same origins and scoring as the classic models (run_backtest.py).
#
# Run from the project folder:
#   python -m src.run_backtest_dl                      (all 3 models)
#   python -m src.run_backtest_dl NBEATS Chronos2      (only some models)
#
# Output: results/tables/backtest_day5_<model>.parquet (every forecast of that model)

import sys
import time
from pathlib import Path

from src.evaluation.backtest import load_data, make_origins, run_model
from src.evaluation.metrics import add_errors, summarize

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"


def get_model(name):
    # the imports are inside, so a model's library is only loaded when we run that model
    if name == "NBEATS":
        from src.models.nbeats import make_nbeats
        return make_nbeats()
    if name == "Chronos2":
        from src.models.chronos import make_chronos_zero_shot
        return make_chronos_zero_shot()
    if name == "Chronos2-FT":
        from src.models.chronos import make_chronos_finetuned
        return make_chronos_finetuned()
    raise ValueError(name)


def main():
    names = sys.argv[1:] or ["NBEATS", "Chronos2", "Chronos2-FT"]
    data = load_data()
    origins = make_origins(data["ds"])
    print(f"{data['unique_id'].nunique()} series, {len(origins)} origins "
          f"from {origins[0].date()} to {origins[-1].date()}, horizons 1-8")

    for name in names:
        start = time.time()
        res = add_errors(run_model(get_model(name), name, data, origins))
        res.to_parquet(TABLES / f"backtest_day5_{name}.parquet", index=False)
        print(f"{name}: {len(res)} forecasts in {time.time() - start:.0f}s")
        print(summarize(res, ["model"]).to_string(index=False))


if __name__ == "__main__":
    main()
