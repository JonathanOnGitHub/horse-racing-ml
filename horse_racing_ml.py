#!/usr/bin/env python3
"""
horse_racing_ml.py — Full pipeline: data loading, feature engineering,
XGBoost training, Kelly backtest, and results report.

Runs as: python3 horse_racing_ml.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ── Config ──────────────────────────────────────────────────────────
DATA = Path.home() / ".cache/kagglehub/datasets/deltaromeo/horse-racing-results-ukireland-2015-2025/versions/118"
TRAIN_END = "2013-12-31"      # Train on 1988–2013
VAL_END   = "2014-12-31"      # Validate on 2014 (for early stopping)
TEST_START = "2015-01-01"     # Backtest on 2015

MIN_ODDS = 1.5     # Minimum decimal odds to bet (avoid odds-on favourites)
MAX_ODDS = 50.0    # Maximum decimal odds to bet (avoid extreme longshots)
BANKROLL = 10000   # Starting bankroll (£)
STAKE_FRAC = 0.02  # Fractional Kelly (0.02 = 2% of bankroll per bet)
COMMISSION = 0.05  # Betfair-style commission on winning bets (5%)

# ── 1. LOAD DATA ────────────────────────────────────────────────────
print("=" * 60)
print("HORSE RACING ML BETTING SYSTEM")
print("=" * 60)

print("\n[1/5] Loading data...")

# Common columns we use
USE_COLS = ['date', 'course', 'race_id', 'type', 'class', 'dist', 'going',
            'ran', 'num', 'pos', 'draw', 'horse', 'age', 'sex', 'wgt',
            'sp', 'jockey', 'trainer', 'or', 'rpr', 'ts']

def load_archive(path, label):
    """Load an archive CSV, normalise columns."""
    df = pd.read_csv(path, low_memory=False)
    # Only keep columns that exist
    avail = [c for c in USE_COLS if c in df.columns]
    df = df[avail]
    # Ensure standard dtypes
    for c in df.columns:
        if df[c].dtype == 'object':
            df[c] = df[c].astype(str)
    print(f"  {label}: {len(df):,} rows, {len(df.columns)} cols, "
          f"{df['date'].min()} to {df['date'].max()}")
    return df

# Load all archives (they share the same schema for 37-col files)
df_1988 = load_archive(DATA / "archive_1988-2004/archive_1988-2004/1988-2004.csv", "1988-2004")
df_2015 = load_archive(DATA / "form_2015-present/form_2015-present/raceform.csv", "2015")

# Combine
df = pd.concat([df_1988, df_2015], ignore_index=True)
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df = df.dropna(subset=['date']).sort_values('date').reset_index(drop=True)
print(f"\n  Combined: {len(df):,} rows ({df['date'].min().year}–{df['date'].max().year})")

# ── 2. CLEAN & ENGINEER FEATURES ────────────────────────────────────
print("\n[2/5] Cleaning & engineering features...")

def parse_sp(sp_str):
    """Convert UK fractional odds to decimal. Returns NaN if unparseable."""
    if pd.isna(sp_str):
        return np.nan
    sp_str = str(sp_str).strip()
    # Remove 'F' favourite marker
    sp_str = sp_str.replace('F', '').strip()
    try:
        if '/' in sp_str:
            num, den = sp_str.split('/')
            frac = float(num) / float(den)
            # If evens or better -> decimal = frac + 1
            return frac + 1.0
        else:
            return float(sp_str) + 1.0 if float(sp_str) >= 1 else np.nan
    except:
        return np.nan

def parse_weight(wgt_str):
    """Convert UK weight '11-6' to lbs (11st 6lb = 154lb)."""
    if pd.isna(wgt_str):
        return np.nan
    wgt_str = str(wgt_str).strip()
    try:
        if '-' in wgt_str:
            st, lb = wgt_str.split('-')
            return int(st) * 14 + int(lb)
        return float(wgt_str)
    except:
        return np.nan

def parse_distance(dist_str):
    """Convert distance string like '1m2f' or '2m3½f' to yards."""
    if pd.isna(dist_str):
        return np.nan
    d = str(dist_str).replace('½', '.5').replace('¼', '.25').replace('¾', '.75')
    d = d.replace(' ', '').replace('(', '').replace(')', '')
    yards = 0.0
    if 'm' in d:
        parts = d.split('m')
        if parts[0]:
            yards += float(parts[0]) * 1760  # miles to yards
        rest = parts[1] if len(parts) > 1 else ''
        if 'f' in rest:
            f = rest.replace('f', '')
            if f:
                yards += float(f) * 220  # furlongs to yards
    elif 'f' in d:
        f = d.replace('f', '')
        if f:
            yards += float(f) * 220
    else:
        try:
            yards = float(d) * 220
        except:
            return np.nan
    return yards

def is_winner(pos):
    """Check if horse finished 1st."""
    if pd.isna(pos):
        return np.nan
    try:
        return 1.0 if int(float(pos)) == 1 else 0.0
    except:
        return 0.0  # Fell, pulled up, etc. = not a win

def is_placed(pos, ran):
    """Check if horse placed (1st-3rd, or 1st-4th for 16+ runners)."""
    if pd.isna(pos) or pd.isna(ran):
        return np.nan
    try:
        p = int(float(pos))
        if ran >= 16:
            return 1.0 if 1 <= p <= 4 else 0.0
        else:
            return 1.0 if 1 <= p <= 3 else 0.0
    except:
        return 0.0

# Parse odds
df['decimal_odds'] = df['sp'].apply(parse_sp)

# Parse weight to lbs
df['wgt_lbs'] = df['wgt'].apply(parse_weight)

# Parse distance to yards
df['dist_yards'] = df['dist'].apply(parse_distance)

# Target variable
df['won'] = df['pos'].apply(is_winner)
df['placed'] = df.apply(lambda r: is_placed(r['pos'], r['ran']), axis=1)

# Parse numeric ratings
for col in ['or', 'rpr', 'ts']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Convert age and draw
df['age'] = pd.to_numeric(df['age'], errors='coerce')
df['draw'] = pd.to_numeric(df['draw'], errors='coerce')
df['num'] = pd.to_numeric(df['num'], errors='coerce')
df['ran'] = pd.to_numeric(df['ran'], errors='coerce')

# Fill missing with sensible defaults
df['draw'] = df['draw'].fillna(df['num'])
df['ts'] = df['ts'].fillna(0)
df['rpr'] = df['rpr'].fillna(0)
df['or'] = df['or'].fillna(0)

# ── Build rolling form features per horse ──
print("  Building horse-level rolling form features...")

# Sort by horse, date for rolling
df = df.sort_values(['horse', 'date']).reset_index(drop=True)

# Recent form: last 3 and last 6 runs
horse_groups = df.groupby('horse')
df['form_3_win'] = horse_groups['won'].transform(
    lambda x: x.rolling(4, min_periods=1).mean().shift(1))
df['form_6_win'] = horse_groups['won'].transform(
    lambda x: x.rolling(7, min_periods=1).mean().shift(1))

# Days since last run
df['days_since_last'] = horse_groups['date'].transform(
    lambda x: x.diff().dt.days)

# Recent RPR trend (last 3 runs average vs career average)
df['rpr_ma3'] = horse_groups['rpr'].transform(
    lambda x: x.rolling(4, min_periods=1).mean().shift(1))
df['rpr_career_avg'] = horse_groups['rpr'].transform(
    lambda x: x.expanding().mean().shift(1))
df['rpr_trend'] = df['rpr_ma3'] - df['rpr_career_avg']

# Jockey/trainer combo win rate (look-back, not look-ahead)
def expanding_win_rate(series):
    """Cumulative win rate, shifted to avoid look-ahead."""
    exp = series.expanding()
    return (exp.sum() - series) / (exp.count() - 1).clip(lower=1)

df['jockey_win_rate'] = df.groupby('jockey')['won'].transform(expanding_win_rate)
df['trainer_win_rate'] = df.groupby('trainer')['won'].transform(expanding_win_rate)

# Course familiarity (number of previous runs at this course)
df['course_runs'] = df.groupby(['horse', 'course']).cumcount()

# Clean up infinity and NaN from form features
for c in ['form_3_win', 'form_6_win', 'days_since_last', 'rpr_ma3',
          'rpr_career_avg', 'rpr_trend', 'jockey_win_rate', 'trainer_win_rate']:
    df[c] = df[c].replace([np.inf, -np.inf], np.nan)
    df[c] = df[c].fillna(0)

# ── Encode categoricals ──
cat_cols = ['course', 'type', 'class', 'going', 'sex', 'horse', 'jockey', 'trainer']

# Label encode (frequency-based) for tree model
from sklearn.preprocessing import LabelEncoder
for col in cat_cols:
    le = LabelEncoder()
    df[col + '_enc'] = le.fit_transform(df[col].astype(str))

# ── Compute market-implied probability ──
df['market_prob'] = 1.0 / df['decimal_odds']
df['market_prob'] = df['market_prob'].replace([np.inf, -np.inf], np.nan)

print(f"  Usable rows: {df['won'].notna().sum():,} "
      f"(winners: {int(df['won'].sum()):,}, "
      f"win rate: {df['won'].mean()*100:.2f}%)")

# ── 3. SPLIT DATA ───────────────────────────────────────────────────
print("\n[3/5] Splitting data temporally...")

train = df[df['date'] <= TRAIN_END].copy()
val = df[(df['date'] > TRAIN_END) & (df['date'] <= VAL_END)].copy()
test = df[df['date'] >= TEST_START].copy()

print(f"  Train:   {len(train):,} ({train['date'].min().year}–{train['date'].max().year})")
print(f"  Val:     {len(val):,} ({val['date'].min().year}–{val['date'].max().year})")
print(f"  Test:    {len(test):,} ({test['date'].min().year}–{test['date'].max().year})")

# Features
feature_cols = [
    'ran', 'num', 'age', 'wgt_lbs', 'dist_yards', 'draw', 'or', 'rpr', 'ts',
    'form_3_win', 'form_6_win', 'days_since_last', 'rpr_ma3', 'rpr_career_avg',
    'rpr_trend', 'jockey_win_rate', 'trainer_win_rate', 'course_runs',
    'market_prob',
    'course_enc', 'type_enc', 'class_enc', 'going_enc', 'sex_enc',
]

# Drop rows with missing features
train_ml = train.dropna(subset=feature_cols + ['won'])
val_ml = val.dropna(subset=feature_cols + ['won'])
test_ml = test.dropna(subset=feature_cols + ['won'])

X_train = train_ml[feature_cols].values
y_train = train_ml['won'].values
X_val = val_ml[feature_cols].values
y_val = val_ml['won'].values
X_test = test_ml[feature_cols].values
y_test = test_ml['won'].values

print(f"  Train features: {X_train.shape}")
print(f"  Val features:   {X_val.shape}")
print(f"  Test features:  {X_test.shape}")

# ── 4. TRAIN XGBoOST ────────────────────────────────────────────────
print("\n[4/5] Training XGBoost model...")

import xgboost as xgb

# Class weights: penalise false positives more since we're betting
neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
scale_pos_weight = neg / pos
print(f"  Class balance: {pos:,} wins / {neg:,} losses (scale={scale_pos_weight:.1f})")

model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    reg_lambda=2.0,
    reg_alpha=0.5,
    eval_metric='logloss',
    early_stopping_rounds=30,
    random_state=42,
    verbosity=0,
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False,
)

# Feature importance
importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n  Top 15 features:")
for i, row in importance.head(15).iterrows():
    print(f"    {row['feature']:25s}  {row['importance']:.4f}")

# ── 5. EVALUATE ─────────────────────────────────────────────────────
print("\n[5/5] Evaluating model...")

from sklearn.metrics import roc_auc_score, brier_score_loss

# Predict probabilities
train_prob = model.predict_proba(X_train)[:, 1]
val_prob = model.predict_proba(X_val)[:, 1]
test_prob = model.predict_proba(X_test)[:, 1]

print(f"\n  ROC-AUC:")
print(f"    Train: {roc_auc_score(y_train, train_prob):.4f}")
print(f"    Val:   {roc_auc_score(y_val, val_prob):.4f}")
print(f"    Test:  {roc_auc_score(y_test, test_prob):.4f}")

print(f"\n  Brier Score (lower is better):")
print(f"    Train: {brier_score_loss(y_train, train_prob):.4f}")
print(f"    Test:  {brier_score_loss(y_test, test_prob):.4f}")

# ── 6. BACKTEST SIMULATION ──────────────────────────────────────────
print("\n" + "=" * 60)
print("BACKTEST SIMULATION (2015)")
print("=" * 60)

# Use test set predictions
test_ml = test_ml.copy()
test_ml['model_prob'] = test_prob
test_ml['model_edge'] = test_ml['model_prob'] - test_ml['market_prob']

# Apply filters
test_ml['bet_signal'] = (
    (test_ml['decimal_odds'] >= MIN_ODDS) &
    (test_ml['decimal_odds'] <= MAX_ODDS) &
    (test_ml['model_prob'] > test_ml['market_prob']) &  # Only if model sees value
    (test_ml['model_prob'] >= 0.02)                     # At least 2% estimated chance
)

bets = test_ml[test_ml['bet_signal']].copy()
print(f"\n  Total race entries evaluated: {len(test_ml):,}")
print(f"  Bets placed: {len(bets):,} ({len(bets)/len(test_ml)*100:.1f}% of races)")

if len(bets) == 0:
    print("  ⚠ NO BETS — consider relaxing thresholds.")
else:
    # Simulate each bet
    results = []
    bankroll = BANKROLL
    peak_bankroll = BANKROLL
    max_drawdown = 0
    kelly_history = []

    for _, bet in bets.iterrows():
        # Full Kelly fraction for this bet
        p = bet['model_prob']
        q = 1 - p
        b = bet['decimal_odds'] - 1  # net odds

        if b <= 0:
            continue

        # Kelly fraction = (bp - q) / b
        kelly_pct = (p * b - q) / b
        # Apply fractional Kelly
        stake_pct = min(STAKE_FRAC, kelly_pct * STAKE_FRAC / 0.25) if kelly_pct > 0 else 0

        if stake_pct <= 0:
            continue

        stake = bankroll * stake_pct
        if stake < 1:  # Minimum £1 bet
            continue

        # Outcome
        won_bet = bet['won'] == 1
        if won_bet:
            net_return = stake * (bet['decimal_odds'] - 1) * (1 - COMMISSION)
            bankroll += net_return
        else:
            bankroll -= stake

        if bankroll > peak_bankroll:
            peak_bankroll = bankroll
        dd = (peak_bankroll - bankroll) / peak_bankroll
        max_drawdown = max(max_drawdown, dd)

        kelly_history.append({
            'date': bet['date'],
            'horse': bet['horse'],
            'odds': bet['decimal_odds'],
            'model_prob': p,
            'market_prob': bet['market_prob'],
            'edge': bet['model_edge'],
            'stake': stake,
            'won': won_bet,
            'bankroll': bankroll,
        })

    kelly_df = pd.DataFrame(kelly_history)

    if len(kelly_df) > 0:
        final_bankroll = kelly_df['bankroll'].iloc[-1]
        total_staked = kelly_df['stake'].sum()
        wins = kelly_df['won'].sum()
        total_bets = len(kelly_df)

        print(f"\n  ── P&L ──")
        print(f"  Starting bankroll: £{BANKROLL:,.2f}")
        print(f"  Final bankroll:    £{final_bankroll:,.2f}")
        print(f"  Profit/Loss:       £{final_bankroll - BANKROLL:+,.2f}")
        print(f"  Return:            {((final_bankroll / BANKROLL) - 1) * 100:+.2f}%")

        print(f"\n  ── Betting Stats ──")
        print(f"  Total bets:        {total_bets:,}")
        print(f"  Total staked:      £{total_staked:,.2f}")
        print(f"  Win rate:          {wins / total_bets * 100:.2f}%")
        print(f"  Avg odds:          {kelly_df['odds'].mean():.2f}")
        print(f"  Avg edge:          {kelly_df['edge'].mean():.4f}")

        print(f"\n  ── Risk Metrics ──")
        print(f"  Max drawdown:      {max_drawdown * 100:.2f}%")
        print(f"  Avg Kelly frac:    {STAKE_FRAC:.2f}")

        # Sharpe ratio (annualised, assuming ~200 betting days)
        daily_returns = kelly_df.set_index('date').resample('D')['bankroll'].last().ffill().pct_change().dropna()
        if len(daily_returns) > 1:
            sharpe = np.sqrt(252) * daily_returns.mean() / daily_returns.std()
            print(f"  Sharpe ratio:      {sharpe:.2f} (annualised)")

        # Compare to market baseline (betting on all market-favoured horses)
        market_bets = test_ml[test_ml['decimal_odds'].between(MIN_ODDS, MAX_ODDS)].copy()
        market_bets['favoured'] = market_bets['decimal_odds'] < \
            market_bets.groupby('race_id')['decimal_odds'].transform('min') * 1.05

        market_fav = market_bets[market_bets['favoured']]
        if len(market_fav) > 0:
            mkt_bankroll = BANKROLL
            for _, bet in market_fav.iterrows():
                stake = mkt_bankroll * STAKE_FRAC
                if bet['won'] == 1:
                    mkt_bankroll += stake * (bet['decimal_odds'] - 1) * (1 - COMMISSION)
                else:
                    mkt_bankroll -= stake
            print(f"\n  ── Comparison ──")
            print(f"  Market favourite strategy:")
            print(f"    Final bankroll: £{mkt_bankroll:,.2f}")
            print(f"    Profit: £{mkt_bankroll - BANKROLL:+,.2f} ({(mkt_bankroll/BANKROLL-1)*100:+.2f}%)")

        # ── Print some example bets ──
        print(f"\n  ── Example Bets (first 5) ──")
        for _, bet in kelly_df.head(5).iterrows():
            tick = "✅" if bet['won'] else "❌"
            print(f"  {bet['date'].strftime('%d/%m/%y')} | {bet['horse'][:20]:20s} | "
                  f"{bet['odds']:.2f} | stake £{bet['stake']:.0f} | {tick}")

        # ── Monthly breakdown ──
        print(f"\n  ── Monthly P&L ──")
        kelly_df['month'] = kelly_df['date'].dt.to_period('M')
        monthly = kelly_df.groupby('month').agg(
            bets=('won', 'count'),
            wins=('won', 'sum'),
            stake=('stake', 'sum'),
            pnl=('bankroll', 'last')
        )
        monthly['profit'] = monthly['pnl'] - monthly['pnl'].shift(1).fillna(BANKROLL)
        monthly['win_rate'] = monthly['wins'] / monthly['bets'] * 100
        for month, row in monthly.iterrows():
            pnl_str = f"£{row['profit']:+.0f}"
            print(f"    {month} | bets: {int(row['bets']):3d} | "
                  f"wins: {int(row['wins']):2d} | WR: {row['win_rate']:.0f}% | "
                  f"P&L: {pnl_str:>8s} | bal: £{row['pnl']:,.0f}")

        # ── Save results ──
        out_dir = Path("/home/burley/horse-racing-ml/output")
        out_dir.mkdir(exist_ok=True)
        kelly_df.to_csv(out_dir / "simulation_results.csv", index=False)
        monthly.to_csv(out_dir / "monthly_pnl.csv")
        importance.to_csv(out_dir / "feature_importance.csv", index=False)
        print(f"\n  Results saved to {out_dir}/")

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
