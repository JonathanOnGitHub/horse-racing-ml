#!/usr/bin/env python3
"""Efficient feature engineering — 2005-2015 only, optimised transforms."""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings, time, os
warnings.filterwarnings('ignore')
t0 = time.time()

DATA = Path.home() / '.cache/kagglehub/datasets/deltaromeo/horse-racing-results-ukireland-2015-2025/versions/118'
OUT = Path('/home/burley/horse-racing-ml')
COLS = ['date','course','race_id','type','class','dist','going','ran',
        'num','pos','draw','horse','age','sex','wgt','sp',
        'jockey','trainer','or','rpr','ts']

# ── Load 2005-2014 and 2015 ──
parts = []
for f in [
    DATA/'archive_2005-2014/archive_2005-2014/2005-2014.csv',
    DATA/'form_2015-present/form_2015-present/raceform.csv'
]:
    avail = [c for c in COLS if c in pd.read_csv(f, nrows=1).columns]
    parts.append(pd.read_csv(f, usecols=avail, low_memory=False))

df = pd.concat(parts, ignore_index=True)
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df = df.dropna(subset=['date']).sort_values('date').reset_index(drop=True)
print(f"Loaded {len(df):,} rows ({df['date'].dt.year.min()}–{df['date'].dt.year.max()}) [{time.time()-t0:.0f}s]")

# ── Parse ──
def parse_sp(s):
    if pd.isna(s): return np.nan
    s = str(s).strip().replace('F','').strip()
    try:
        if '/' in s: n,d=s.split('/'); return float(n)/float(d)+1.0
        return float(s)+1.0 if float(s)>=1 else np.nan
    except: return np.nan

def parse_wgt(w):
    if pd.isna(w): return np.nan
    try:
        if '-' in str(w): st,lb=str(w).split('-'); return int(st)*14+int(lb)
        return float(w)
    except: return np.nan

def parse_dist(d):
    if pd.isna(d): return np.nan
    d=str(d).replace('\u00bd','.5').replace('\u00bc','.25').replace('\u00be','.75')
    d=d.replace(' ','').replace('(','').replace(')','')
    y=0.0
    if 'm' in d:
        p=d.split('m')
        y+=float(p[0])*1760 if p[0] else 0
        rest=p[1] if len(p)>1 else ''
        if 'f' in rest:
            f=rest.replace('f','')
            y+=float(f)*220 if f else 0
    elif 'f' in d:
        f=d.replace('f','')
        y+=float(f)*220 if f else 0
    else:
        try: y=float(d)*220
        except: return np.nan
    return y

def is_winner(p):
    if pd.isna(p): return np.nan
    try: return 1.0 if int(float(p))==1 else 0.0
    except: return 0.0

print("Parsing...")
df['decimal_odds'] = df['sp'].apply(parse_sp)
df['wgt_lbs'] = df['wgt'].apply(parse_wgt)
df['dist_yards'] = df['dist'].apply(parse_dist)
df['won'] = df['pos'].apply(is_winner)
for c in ['or','rpr','ts','age','draw','num','ran']:
    df[c] = pd.to_numeric(df[c], errors='coerce')
df['draw'] = df['draw'].fillna(df['num'])
for c in ['ts','rpr','or']: df[c] = df[c].fillna(0)
print(f"  Done [{time.time()-t0:.0f}s]")

# ── Rolling form features (sorted by horse,date) ──
print("Rolling features...")
df = df.sort_values(['horse','date']).reset_index(drop=True)
h = df.groupby('horse')
df['form_3_win'] = h['won'].transform(lambda x: x.rolling(4,min_periods=1).mean().shift(1))
df['form_6_win'] = h['won'].transform(lambda x: x.rolling(7,min_periods=1).mean().shift(1))
df['days_since_last'] = h['date'].transform(lambda x: x.diff().dt.days)
df['rpr_ma3'] = h['rpr'].transform(lambda x: x.rolling(4,min_periods=1).mean().shift(1))
df['rpr_career_avg'] = h['rpr'].transform(lambda x: x.expanding().mean().shift(1))
df['rpr_trend'] = df['rpr_ma3'] - df['rpr_career_avg']
df['course_runs'] = df.groupby(['horse','course']).cumcount()
print(f"  Done [{time.time()-t0:.0f}s]")

# ── Jockey/trainer rates (optimised: transform, no manual loop) ──
print("Jockey/trainer rates...")
for col in ['jockey','trainer']:
    g = df.groupby(col)['won']
    cum_sum = g.cumsum()
    cum_cnt = g.cumcount() + 1
    df[f'{col}_win_rate'] = ((cum_sum - df['won']) / (cum_cnt - 1)).fillna(0)
    print(f"  {col} done [{time.time()-t0:.0f}s]")

# Fill
for c in ['form_3_win','form_6_win','days_since_last','rpr_ma3','rpr_career_avg',
          'rpr_trend','jockey_win_rate','trainer_win_rate','course_runs']:
    df[c] = df[c].replace([np.inf,-np.inf],np.nan).fillna(0)

# ── Encode ──
print("Encoding...")
from sklearn.preprocessing import LabelEncoder
for col in ['course','type','class','going','sex','horse','jockey','trainer']:
    le=LabelEncoder()
    df[col+'_enc'] = le.fit_transform(df[col].astype(str))

df['market_prob'] = 1.0 / df['decimal_odds'].replace(0,np.nan)
df['market_prob'] = df['market_prob'].replace([np.inf,-np.inf],np.nan)

# ── Save ──
df.to_pickle(OUT/'features.pkl')
mb = os.path.getsize(OUT/'features.pkl')/1e6
print(f"\nSaved features.pkl ({mb:.0f} MB) in {time.time()-t0:.0f}s")
print(f"Rows: {len(df):,}  Winners: {int(df['won'].sum()):,} ({df['won'].mean()*100:.2f}%)")
