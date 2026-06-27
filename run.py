#!/usr/bin/env python3
"""
Horse Racing ML Betting System — Full Pipeline
==============================================
2015 UK/Ireland data.
Train: Jan-Apr   (pre-race features only)
Calibrate: May   (for probability calibration)
Backtest: Jun-Sep
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss
import xgboost as xgb
import warnings, time
warnings.filterwarnings('ignore')
t0 = time.time()

# ── Config ──
TRAIN_END = '2015-04-30'
CAL_END   = '2015-05-31'
MIN_ODDS, MAX_ODDS = 1.5, 50.0
BANKROLL = 10000
COMMISSION = 0.05

DATA = Path.home() / '.cache/kagglehub/datasets/deltaromeo/horse-racing-results-ukireland-2015-2025/versions/118'
OUT = Path('/home/burley/horse-racing-ml')
OUT.mkdir(exist_ok=True)

print("="*60)
print("HORSE RACING ML — V2 (CALIBRATED)")
print("="*60)

# ── 1. LOAD ──
print("\n[1] Loading...")
COLS = ['date','course','race_id','type','class','dist','going','ran',
        'num','pos','draw','horse','age','sex','wgt','sp',
        'jockey','trainer','or','rpr','ts']

df = pd.read_csv(DATA/'form_2015-present/form_2015-present/raceform.csv',
                 usecols=COLS, low_memory=False)
df['date'] = pd.to_datetime(df['date'], format='mixed', errors='coerce')
df = df.dropna(subset=['date'])
print(f"  {len(df):,} rows  ({df['date'].min().date()} to {df['date'].max().date()})")

# ── 2. FEATURE ENGINEERING ──
print("[2] Features...")

def sp_dec(s):
    if pd.isna(s): return np.nan
    s = str(s).strip().replace('F','').strip()
    try:
        if '/' in s: n,d=s.split('/'); return float(n)/float(d)+1.0
        return float(s)+1.0 if float(s)>=1 else np.nan
    except: return np.nan

def wgt_lbs(w):
    if pd.isna(w): return np.nan
    try:
        if '-' in str(w): st,lb=str(w).split('-'); return int(st)*14+int(lb)
        return float(w)
    except: return np.nan

def dist_yds(d):
    if pd.isna(d): return np.nan
    d=str(d).replace('\u00bd','.5').replace('\u00bc','.25').replace('\u00be','.75')
    d=d.replace(' ','').replace('(','').replace(')','')
    y=0.0
    if 'm' in d:
        p=d.split('m')
        y+=float(p[0])*1760 if p[0] else 0
        if len(p)>1:
            r=p[1].replace('f','')
            y+=float(r)*220 if r else 0
    elif 'f' in d:
        r=d.replace('f','')
        y+=float(r)*220 if r else 0
    else:
        try: y=float(d)*220
        except: return np.nan
    return y

def win(p):
    if pd.isna(p): return np.nan
    try: return 1 if int(float(p))==1 else 0
    except: return 0

df['decimal_odds'] = df['sp'].apply(sp_dec)
df['wgt_lbs'] = df['wgt'].apply(wgt_lbs)
df['dist_yards'] = df['dist'].apply(dist_yds)
df['won'] = df['pos'].apply(win)

for c in ['or','rpr','ts','age','draw','num','ran']:
    df[c] = pd.to_numeric(df[c], errors='coerce')
df['draw'] = df['draw'].fillna(df['num'])
df[['ts','rpr','or']] = df[['ts','rpr','or']].fillna(0)

# Rolling features (sorted by horse then date)
df = df.sort_values(['horse','date']).reset_index(drop=True)
h = df.groupby('horse')
df['form_3_win'] = h['won'].transform(lambda x: x.rolling(4,1).mean().shift(1))
df['form_6_win'] = h['won'].transform(lambda x: x.rolling(7,1).mean().shift(1))
df['days_since_last'] = h['date'].transform(lambda x: x.diff().dt.days)
# Past RPR averages (shifted to avoid look-ahead) - these are PRE-RACE features
df['rpr_ma3'] = h['rpr'].transform(lambda x: x.rolling(4,1).mean().shift(1))
df['rpr_career_avg'] = h['rpr'].transform(lambda x: x.expanding().mean().shift(1))
df['rpr_trend'] = df['rpr_ma3'] - df['rpr_career_avg']
df['course_runs'] = df.groupby(['horse','course']).cumcount()

# Jockey/trainer rates
for col in ['jockey','trainer']:
    g = df.groupby(col)['won']
    cs = g.cumsum()
    cc = g.cumcount()+1
    df[f'{col}_wr'] = ((cs-df['won'])/(cc-1)).fillna(0)

fill_cols = ['form_3_win','form_6_win','days_since_last','rpr_ma3',
             'rpr_career_avg','rpr_trend','jockey_wr','trainer_wr','course_runs']
for c in fill_cols:
    df[c] = df[c].replace([np.inf,-np.inf],np.nan).fillna(0)

# Encode categories
from sklearn.preprocessing import LabelEncoder
for c in ['course','type','class','going','sex']:
    le = LabelEncoder()
    df[c+'_enc'] = le.fit_transform(df[c].astype(str))

df['market_prob'] = (1.0/df['decimal_odds']).replace([np.inf,-np.inf],np.nan)
df = df.dropna(subset=['decimal_odds','won','market_prob'])
print(f"  {len(df):,} usable rows [{time.time()-t0:.0f}s]")

# ── 3. SPLIT ──
print("[3] Temporal split...")
train_df = df[df['date'] <= TRAIN_END].copy()
cal_df = df[(df['date'] > TRAIN_END) & (df['date'] <= CAL_END)].copy()
test_df = df[df['date'] > CAL_END].copy()
print(f"  Train:       {len(train_df):,}")
print(f"  Calibrate:   {len(cal_df):,}")
print(f"  Backtest:    {len(test_df):,}")

FEATURES = [
    'ran','num','age','wgt_lbs','dist_yards','draw','or',
    'form_3_win','form_6_win','days_since_last','rpr_ma3','rpr_career_avg',
    'rpr_trend','jockey_wr','trainer_wr','course_runs',
    'course_enc','type_enc','class_enc','going_enc','sex_enc',
]

def prep(df_):
    X = df_[FEATURES].values.astype(np.float32)
    y = df_['won'].values.astype(np.float32)
    return X, y

X_train, y_train = prep(train_df)
X_cal, y_cal = prep(cal_df)
X_test, y_test = prep(test_df)

# ── 4. TRAIN + CALIBRATE ──
print("[4] Training XGBoost + calibrating...")
neg, pos = (y_train==0).sum(), (y_train==1).sum()
scale = neg / pos

model = xgb.XGBClassifier(
    n_estimators=200, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=scale,
    reg_lambda=2.0, reg_alpha=0.5,
    eval_metric='logloss', random_state=42, verbosity=0,
)
model.fit(X_train, y_train, verbose=False)

# Calibrate on held-out May data using Platt scaling (logistic regression)
from sklearn.linear_model import LogisticRegression
calibrator = LogisticRegression()
calibrator.fit(model.predict_proba(X_cal)[:,1].reshape(-1,1), y_cal)

def calibrate_probs(raw_probs):
    """Apply Platt scaling to raw XGBoost probabilities."""
    return calibrator.predict_proba(raw_probs.reshape(-1,1))[:,1]

# Evaluate
train_prob_cal = calibrate_probs(model.predict_proba(X_train)[:,1])
test_prob_cal = calibrate_probs(model.predict_proba(X_test)[:,1])
test_prob_raw = model.predict_proba(X_test)[:,1]

print(f"  ROC-AUC (calibrated): {roc_auc_score(y_test, test_prob_cal):.4f}")
print(f"  Brier  (calibrated):  {brier_score_loss(y_test, test_prob_cal):.4f}")
print(f"  Brier  (raw):         {brier_score_loss(y_test, test_prob_raw):.4f}")

# Feature importance
imp = pd.DataFrame({'feat': FEATURES, 'imp': model.feature_importances_})
imp = imp.sort_values('imp', ascending=False)
print("\n  Top features:")
for _, r in imp.head(10).iterrows():
    print(f"    {r['feat']:20s}  {r['imp']:.4f}")

# ── 5. BACKTEST ──
print("\n[5] Backtesting (3 strategies)...")

test_df = test_df.copy()
test_df['model_prob'] = test_prob_cal
test_df['edge'] = test_df['model_prob'] - test_df['market_prob']
test_df['kelly'] = ((test_df['model_prob'] * (test_df['decimal_odds']-1) - (1-test_df['model_prob']))
                     / (test_df['decimal_odds']-1)).clip(0)

results_list = []
for frac, label in [(0.02, 'Kelly (2% frac)'), (0.01, 'Kelly (1% frac)'),
                    (0.005, 'Kelly (0.5% frac)')]:
    bankroll = BANKROLL
    peak = BANKROLL
    max_dd = 0
    trades = []

    for _, row in test_df.iterrows():
        k = row['kelly']
        if k <= 0: continue
        if not (MIN_ODDS <= row['decimal_odds'] <= MAX_ODDS): continue
        if row['model_prob'] < 0.02: continue
        if row['model_prob'] <= row['market_prob']: continue

        stake_pct = min(frac, k * frac / 0.25) if k > 0 else 0
        if stake_pct <= 0: continue
        stake = bankroll * stake_pct
        if stake < 1: continue

        if row['won']:
            bankroll += stake * (row['decimal_odds']-1) * (1-COMMISSION)
        else:
            bankroll -= stake

        peak = max(peak, bankroll)
        max_dd = max(max_dd, (peak-bankroll)/peak)
        trades.append({'won': row['won'], 'odds': row['decimal_odds'],
                       'prob': row['model_prob'], 'stake': stake,
                       'bankroll': bankroll, 'date': row['date'],
                       'horse': row['horse'], 'course': row['course']})

    rdf = pd.DataFrame(trades) if trades else pd.DataFrame()
    if len(rdf) > 0:
        final = rdf['bankroll'].iloc[-1]
        wins = rdf['won'].sum()
        n = len(rdf)
        print(f"\n  {label}:")
        print(f"    Bets: {n:,} | Win: {wins/n*100:.1f}% | "
              f"Avg odds: {rdf['odds'].mean():.1f}")
        print(f"    £{BANKROLL:,.0f} → £{final:,.0f} "
              f"(£{final-BANKROLL:+,.0f}, {(final/BANKROLL-1)*100:+.1f}%)")
        print(f"    Max DD: {max_dd*100:.1f}%")
        results_list.append((label, rdf))

# ── Strategy 4: Thresholded (bet when edge > 10%) ──
bankroll = BANKROLL
peak = BANKROLL
max_dd = 0
trades = []
for _, row in test_df.iterrows():
    if not (MIN_ODDS <= row['decimal_odds'] <= MAX_ODDS): continue
    if row['edge'] <= 0.10: continue  # min 10% edge over market
    if row['model_prob'] < 0.02: continue

    k = row['kelly']
    k = min(max(k, 0), 0.25)  # cap Kelly at 25%
    stake_pct = 0.02  # fixed 2% regardless
    stake = bankroll * stake_pct
    if stake < 1: continue

    if row['won']:
        bankroll += stake * (row['decimal_odds']-1) * (1-COMMISSION)
    else:
        bankroll -= stake
    peak = max(peak, bankroll)
    max_dd = max(max_dd, (peak-bankroll)/peak)
    trades.append({'won': row['won'], 'odds': row['decimal_odds'],
                   'edge': row['edge'], 'stake': stake,
                   'bankroll': bankroll, 'date': row['date'],
                   'horse': row['horse'], 'course': row['course'],
                   'model_prob': row['model_prob']})

rdf = pd.DataFrame(trades) if trades else pd.DataFrame()
if len(rdf) > 0:
    final = rdf['bankroll'].iloc[-1]
    wins = rdf['won'].sum()
    n = len(rdf)
    print(f"\n  Threshold (edge>10%, fixed 2%):")
    print(f"    Bets: {n:,} | Win: {wins/n*100:.1f}% | "
          f"Avg odds: {rdf['odds'].mean():.1f} | Avg edge: {rdf['edge'].mean():.2f}")
    print(f"    £{BANKROLL:,.0f} → £{final:,.0f} "
          f"(£{final-BANKROLL:+,.0f}, {(final/BANKROLL-1)*100:+.1f}%)")
    print(f"    Max DD: {max_dd*100:.1f}%")
    results_list.append(('Edge>10% fixed 2%', rdf))

# ── Market baseline ──
mkt = BANKROLL
for _, row in test_df.iterrows():
    if not (MIN_ODDS <= row['decimal_odds'] <= MAX_ODDS): continue
    # Market favourite per race
    if row['decimal_odds'] != test_df[test_df['race_id']==row['race_id']]['decimal_odds'].min():
        continue
    stake = mkt * 0.02
    if stake < 1: continue
    if row['won']:
        mkt += stake * (row['decimal_odds']-1) * (1-COMMISSION)
    else:
        mkt -= stake
print(f"\n  Market favourite (2%):")
print(f"    £{BANKROLL:,.0f} → £{mkt:,.0f} "
      f"(£{mkt-BANKROLL:+,.0f}, {(mkt/BANKROLL-1)*100:+.1f}%)")

# ── Summary ──
print(f"\n{'='*60}")
print(f"Total: {time.time()-t0:.0f}s")
print(f"{'='*60}")
