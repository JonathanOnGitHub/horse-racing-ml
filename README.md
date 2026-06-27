# Horse Racing ML Betting System

ML model trained on UK/Ireland horse racing data (2015-2026) to predict win probabilities
and run a Kelly-criterion backtest simulation against market odds.

## Data

- Source: Kaggle - deltaromeo/horse-racing-results-ukireland-2015-2025
- 109K race entries, 2015-01-01 to present
- Features: course, distance, going, class, horse age/sex/weight,
  official rating, RPR, TS, pedigree, jockey, trainer, starting prices
- Betfair exchange data for alternate odds source

## Pipeline

1. `01_explore.py` - Data loading, cleaning, exploratory analysis
2. `02_features.py` - Feature engineering (odds conversion, form features)
3. `03_train.py` - Model training (XGBoost classifier)
4. `04_backtest.py` - Historical simulation with Kelly staking
5. `05_report.py` - Results visualisation and metrics

## Usage

```bash
python3 01_explore.py
python3 02_features.py
python3 03_train.py
python3 04_backtest.py
python3 05_report.py
```
