#!/usr/bin/env python3
"""01_explore.py - Load, clean, and explore the horse racing dataset."""

import pandas as pd
import numpy as np
from pathlib import Path

DATA = Path.home() / ".cache/kagglehub/datasets/deltaromeo/horse-racing-results-ukireland-2015-2025/versions/118"
CSV = DATA / "form_2015-present/form_2015-present/raceform.csv"

df = pd.read_csv(CSV, low_memory=False)

print(f"=== Dataset Overview ===")
print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")
print()

# Column types
print(f"=== Column Types ===")
for col in df.columns:
    print(f"  {col:20s}  {str(df[col].dtype):10s}  nulls={df[col].isna().sum():>6}  unique={df[col].nunique()}")
print()

# Target: finishing position
print(f"=== Finishing Position (pos) ===")
print(df['pos'].value_counts().head(20).to_string())
print(f"  ... {df['pos'].nunique()} unique values")
non_finishers = df['pos'].isin(['PU', 'F', 'UR', 'BD', 'RO', 'SU', 'RR', 'NA'])
print(f"  Non-finishers (PU/F/UR/etc): {non_finishers.sum():,} ({non_finishers.mean()*100:.1f}%)")
print()

# SP (odds analysis)
print(f"=== Starting Prices (sp) ===")
sp_samples = df['sp'].dropna().sample(min(20, len(df)))
print(sp_samples.to_string())
print()

# Race types
print(f"=== Race Types ===")
print(df['type'].value_counts().to_string())
print()

# Going (track condition)
print(f"=== Going (Track Condition) ===")
print(df['going'].value_counts().head(20).to_string())
print()

# Check how many rows have usable odds and position
win_condition = df['pos'].apply(lambda x: str(x).isdigit()).astype(bool)
has_odds = df['sp'].notna()
print(f"=== Usability ===")
print(f"  Rows with numeric finish position: {win_condition.sum():,} ({win_condition.mean()*100:.1f}%)")
print(f"  Rows with valid SP: {has_odds.sum():,} ({has_odds.mean()*100:.1f}%)")
print(f"  Rows with both: {(win_condition & has_odds).sum():,} ({(win_condition & has_odds).mean()*100:.1f}%)")
print()

# Course distribution
print(f"=== Top Courses ===")
print(df['course'].value_counts().head(15).to_string())
print()

# Basic stats on numeric columns
num_cols = ['wgt', 'ran', 'age', 'or', 'rpr', 'ts', 'prize']
for col in num_cols:
    if col in df.columns:
        s = pd.to_numeric(df[col], errors='coerce')
        print(f"  {col:10s}: min={s.min():>8.1f}  median={s.median():>8.1f}  max={s.max():>8.1f}  null={s.isna().sum():>6}")
