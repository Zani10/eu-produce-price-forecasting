# Forecasting weekly EU fruit and vegetable prices

My MSc Computing capstone. I compare different forecasting methods (simple baselines, statistics, machine learning, deep learning and the Chronos-2 foundation model) to predict weekly fruit and vegetable prices in Europe 1 to 8 weeks ahead. I also test if weather data helps, and I build a small assistant that answers buyer questions using the real data and forecasts.

## Data
Weekly fruit and vegetable prices from the EU Agri-food Data Portal (European Commission), price at the packing station in €/100 kg. Downloaded on 31 October 2026, from 2000 until the week of 19-25 October 2026 (about 434,000 prices).

## How to run
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/data/fetch_eu_prices.py
```

## Folders
- src/data: download and clean the data
- src/models: the forecasting models
- src/evaluation: backtest, metrics and tests to compare the models
- src/assistant: the assistant and its tools
- notebooks: data exploration and experiments
- results: figures and tables for the thesis
