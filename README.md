# Horse Racing ML Betting System

ML model trained on UK/Ireland horse racing data (2005–2026) to predict win
probabilities and run Kelly-criterion backtest simulations against market odds.

## Pipeline

| Script | Purpose |
|---|---|
| `01_explore.py` | Load, clean, and explore the raw dataset |
| `stage1_features.py` | Efficient feature engineering (2005–2015 subset) |
| `horse_racing_ml.py` | Full pipeline: feature engineering, XGBoost training, Kelly backtest, results report |
| `run.py` | XGBoost pipeline with temporal split, calibration, Kelly backtest (shorter timeframe) |
| `betfair_analysis.py` | Explore Betfair exchange data columns & liquidity |

## Data

- **Source:** Kaggle — deltaromeo/horse-racing-results-ukireland-2015-2025
- **~109K race entries**, 2015-01-01 to present
- **Features:** course, distance, going, class, horse age/sex/weight, official
  rating, RPR, TS, pedigree, jockey, trainer, starting prices
- Betfair exchange data for alternate odds source

## Key Findings

- Best model: XGBoost with temporal train/calibrate/backtest split
- ROC-AUC ~0.66 — modest predictive power
- All Kelly staking strategies lost 95–99% of bankroll over the test period
- No profitable strategy identified despite extensive feature engineering
- Betfair OHLC data shows pre-race swing median ~70%; in-play range up to 22×

## Requirements

```
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
xgboost>=2.0
matplotlib>=3.7
seaborn>=0.13
scipy>=1.11
```

## Usage

```bash
# Explore dataset
python3 01_explore.py

# Feature engineering (2005-2015 subset)
python3 stage1_features.py

# Full pipeline
python3 horse_racing_ml.py

# Shorter pipeline
python3 run.py

# Betfair data analysis
python3 betfair_analysis.py
```
