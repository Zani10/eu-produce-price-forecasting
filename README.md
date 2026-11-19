# Forecasting weekly EU fruit and vegetable prices

My MSc Computing capstone project. I compare different forecasting methods for weekly fruit and vegetable prices in Europe, 1 to 8 weeks ahead: simple baselines, statistical models, machine learning, deep learning and the Chronos-2 foundation model. I also test if weather data from the growing regions helps, and I built a small assistant that answers buyer questions using the real prices and forecasts.

## Research questions

1. **RQ1** Which forecasting approach is the most accurate for weekly EU fruit and vegetable prices 1 to 8 weeks ahead: seasonal naive, ETS or ARIMA, LightGBM, N-BEATS, or the foundation model Chronos-2 used zero-shot and fine-tuned?
2. **RQ2** Does weather information that is available at the moment of the forecast improve forecast accuracy, and for which products?
3. **RQ3** How does model performance depend on the type of product and on how far ahead the forecast is made?
4. **RQ4** Does an AI assistant with tools answer fresh produce buyers' questions more accurately, and with fewer made-up facts, than the same assistant without tools?

## Data

- **Prices:** EU Agri-food Data Portal (European Commission), fruit and vegetable prices along the supply chain. I use the weekly *ex-packaging station* price (national average). The API gives €/100 kg and I convert it to €/kg. Downloaded on 31 October 2026, from 2000 until the week of 19–25 October 2026 (434,614 records). No API key is needed.
- **Study period:** January 2015 to October 2026 (616 weeks).
- **Sample:** 29 series, 11 products (tomatoes, cucumbers, peppers, courgettes, aubergines, lettuce, cauliflower, apples, pears, lemons, strawberries) in 6 countries (BE, NL, DE, FR, ES, IT). Rules: at most 20% missing weeks, at most 25% stale (repeated) prices, still reported in 2026, one series per product and country. See `src/data/select_sample.py` and `results/tables/dataset_table.csv`.
- **Cleaning:** duplicates are removed, gaps of up to 3 weeks are filled with a straight line, and longer gaps stay empty. Outliers and stale prices get a flag but are not removed. See `results/tables/cleaning_report.txt`.
- **Weather:** Open-Meteo historical API (ERA5), daily values made weekly, for one location in the main growing region of every series (for example Almería for Spanish greenhouse vegetables, Westland for Dutch tomatoes). Free, no key needed.

The data is **not** saved in this repo. The scripts download it and rebuild it in `data/`.

## How the models are tested

Rolling-origin backtest on the last 2 years (104 weeks): 25 forecast dates (origins), one every 4 weeks, from October 2024. At every origin each model only sees the prices up to that date and forecasts the next 8 weeks. All models get the same origins and are scored only on weeks with a real reported price.

Main measure: **MASE**, the error divided by the error of a "same week last year" forecast on the training data (lower is better, below 1 = better than that simple forecast). I also report MAE and RMSE (€/kg), the weighted quantile loss and the coverage of the 80% range. Differences are tested with the Diebold-Mariano test and a Holm correction.

## Setup

Python 3.12.

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Only for RQ4 (the assistant) you need an OpenAI API key. Make a file `.env` in the project folder with:

```
OPENAI_API_KEY=your-key-here
```

`.env` is in `.gitignore` and never committed.

## How to run

Run every command **from the project folder**. Most scripts import from `src`, so use `python -m`. The steps depend on each other, so run them in this order.

| Step | Command | What it does | Output |
|---|---|---|---|
| 1 | `python -m src.data.fetch_eu_prices` | download all prices, one year at a time | `data/raw/`, `data/interim/eu_fv_prices.parquet` |
| 2 | `python -m src.data.clean` | clean, weekly grid, fill short gaps, flags | `data/processed/prices_weekly.parquet`, `results/tables/series_summary.csv`, `cleaning_report.txt` |
| 3 | `python -m src.data.select_sample` | choose the 29 series | `results/tables/sample_series.csv`, `dataset_table.csv` |
| 4 | `notebooks/01_eda.ipynb` | data exploration (optional) | `results/figures/eda_*.png`, `results/tables/eda_*.csv` |
| 5 | `python -m src.run_backtest` | Naive, Seasonal Naive, ETS, ARIMA, LightGBM | `results/tables/backtest_day4.parquet`, `metrics_day4_*.csv` |
| 6 | `python -m src.run_backtest_dl` | N-BEATS, Chronos-2 zero-shot and fine-tuned | `results/tables/backtest_day5_<model>.parquet` |
| 7 | `python -m src.run_results` | RQ1 and RQ3 tables, DM tests and figures | `metrics_rq1_*.csv`, `metrics_rq3_*.csv`, `dm_tests.csv`, `rq1_*.png`, `rq3_*.png` |
| 8 | `python -m src.features.weather` | download the weather | `data/interim/weather_weekly.parquet` |
| 9 | `python -m src.run_weather` | RQ2: LightGBM with and without weather | `backtest_weather.parquet`, `metrics_rq2_*.csv`, `rq2_dm_tests.csv`, `rq2_weather_effect.png` |
| 10 | `python -m src.assistant.make_forecasts` | ARIMA forecasts from the last data week, used by the assistant | `results/tables/current_forecasts.csv` |
| 11 | `python -m src.assistant.questions` | make the 100 test questions with their correct answers | `results/tables/rq4_questions.csv` |
| 12 | `python -m src.assistant.evaluate` | RQ4: ask every question with and without tools and score the answers (needs `.env`) | `rq4_answers.csv`, `rq4_summary.csv`, `rq4_accuracy.png` |
| 13 | `python -m src.check_chronos_release`, `python -m src.check_weather_seeds` and `python -m src.check_extra_stats` | robustness checks: before/after the Chronos-2 release, other seeds, sign test, variance split, DM test with another lag order | `rq1_after_chronos_release.csv`, `rq2_seed_check.csv`, `extra_stats.csv` |

Notes:
- Step 6 downloads the Chronos-2 model (`amazon/chronos-2`) from Hugging Face the first time. On a Mac it uses the Apple GPU (MPS) if there is one, otherwise the CPU.
- `python -m src.assistant.evaluate --score-only` scores the saved answers again without calling the API.
- The file names with `day4` / `day5` are the first part (classic models and LightGBM) and second part (N-BEATS and Chronos-2) of the backtest.
- Steps 7 and 13 only read saved results, so they can be run again without running the models.

## Main results

**RQ1: accuracy (all 29 series, horizons 1–8, 25 origins)**

| Model | MASE | MAE (€/kg) |
|---|---|---|
| Chronos-2 zero-shot | **0.630** | 0.216 |
| ARIMA (Fourier) | 0.662 | 0.226 |
| ETS (STL) | 0.690 | 0.243 |
| Chronos-2 fine-tuned | 0.700 | 0.245 |
| N-BEATS | 0.701 | 0.246 |
| LightGBM | 0.720 | 0.255 |
| Naive | 0.809 | 0.297 |
| Seasonal Naive | 1.028 | 0.333 |

Chronos-2 without any training on our data is the most accurate overall, with ARIMA close behind (per horizon the difference is only significant at 7 weeks ahead). Fine-tuning Chronos-2 on our data made it worse.

**RQ2: weather.** Adding weather did not help: MASE 0.701 without weather, 0.714 with weather, 0.702 with only 3 weather features. Even the "oracle" version that uses the real future weather was not better (0.702).

**RQ3: product type and horizon.** The product matters more than the model: storable products (apples, pears, lemons) are the easiest (best MASE 0.43) and greenhouse vegetables the hardest (0.78). One week ahead all reasonable models are almost equal, the differences grow with the horizon. See `results/tables/metrics_rq3_by_type.csv`, `wins_by_type.csv` and `results/figures/rq3_heatmap_series.png`.

**RQ4: assistant.** On 100 buyer questions the assistant with tools gave 99% correct answers, the same model without tools 9% (35% wrong answers, 56% no answer). McNemar test p < 0.001. The whole run cost about $0.07.

## Folders

```
src/
  data/          download, clean and select the price series
  features/      weather data
  models/        the forecasting models (classic, LightGBM, N-BEATS, Chronos-2)
  evaluation/    backtest, error measures, Diebold-Mariano test
  assistant/     the buyer assistant: tools, test questions, evaluation
  run_*.py       run the experiments (RQ1–RQ3)
  check_*.py     robustness checks
notebooks/       data exploration
results/
  tables/        all result tables (csv, parquet)
  figures/       all figures
data/            downloaded and processed data (not in git)
```
