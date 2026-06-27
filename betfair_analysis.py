#!/usr/bin/env python3
"""Betfair data — actual available columns analysis."""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings, time
warnings.filterwarnings('ignore')
t0 = time.time()

DATA = Path.home() / '.cache/kagglehub/datasets/deltaromeo/horse-racing-results-ukireland-2015-2025/versions/118' / 'betfair/betfair'
SEP = "=" * 60

p1 = pd.read_csv(DATA / 'betfair_mapping_2026_part_i.csv')
p2 = pd.read_csv(DATA / 'betfair_mapping_2026_part_ii.csv')
df = pd.concat([p1, p2], ignore_index=True)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['date','course','off','horse']).reset_index(drop=True)

# Check ALL columns for non-null and non-zero values
print(f"{'Column':20s} {'% non-null':12s} {'% > 0':10s} {'median':10s}")
print("-" * 55)
for c in df.columns:
    if df[c].dtype == 'object':
        nn = df[c].notna().mean() * 100
        print(f"{c:20s} {nn:>10.1f}%{'':12s} {'N/A':>8s}")
    else:
        num = pd.to_numeric(df[c], errors='coerce')
        nn = num.notna().mean() * 100
        gt0 = (num > 0).mean() * 100
        med = num.median()
        print(f"{c:20s} {nn:>10.1f}% {gt0:>10.1f}% {med:>10.2f}")

# What we actually have: bsp, pre_min, pre_max, ip_min, ip_max  
# Clean numeric versions
for c in ['bsp','pre_min','pre_max','ip_min','ip_max']:
    df[c] = pd.to_numeric(df[c], errors='coerce')

# ═══════════════════════════════════════════════════════════════
# STRATEGY: Pre-race volatility — can you trade the range?
# ═══════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("PRE-RACE VOLATILITY TRADING")
print("Back at pre_min, lay at pre_max = guaranteed profit if pre_min < pre_max")
print(SEP)

vol = df[df['pre_min'].notna() & df['pre_max'].notna()].copy()
print(f"Entries with pre-race prices: {len(vol):,}")

vol['swing'] = vol['pre_max'] / vol['pre_min']
# If pre_min == pre_max, no trade possible
tradable = vol[vol['pre_max'] > vol['pre_min']].copy()
print(f"Tradable (pre_max > pre_min): {len(tradable):,} ({len(tradable)/len(vol)*100:.1f}%)")

tradable['return_pct'] = (tradable['swing'] - 1) * 100

print(f"\nPer-horse:")
print(f"  Avg swing:      {tradable['swing'].mean():.2f}x")
print(f"  Median swing:   {tradable['swing'].median():.2f}x")
print(f"  Avg return:     {tradable['return_pct'].mean():.1f}%")
print(f"  Median return:  {tradable['return_pct'].median():.1f}%")

# Return distribution
for bound, label in [(20, '>20%'), (50, '>50%'), (100, '>100%')]:
    pct = (tradable['return_pct'] > bound).mean() * 100
    print(f"  Return {label}:          {pct:.1f}%")

# But this is mostly longshots swinging wildly. Let's check by odds band.
tradable['bsp_band'] = pd.cut(df.loc[tradable.index, 'bsp'],
                               bins=[0, 3, 5, 10, 30, 1000],
                               labels=['Fav (0-3)', 'Short (3-5)', 'Mid (5-10)', 
                                       'Long (10-30)', 'Extreme (30+)'])
print(f"\nBy BSP band:")
for band, g in tradable.groupby('bsp_band', observed=False):
    med_swing = g['swing'].median()
    med_ret = (med_swing - 1) * 100
    print(f"  {band:15s} (n={len(g):,}): median swing {med_swing:.2f}x ({med_ret:+.0f}%)")

# ═══════════════════════════════════════════════════════════════
# STRATEGY: BSP vs pre_min (did price firm or drift to the off?)
# ═══════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("BSP vs PRE-RACE RANGE (market at the off)")
print(SEP)

close = df[df['bsp'].notna() & df['pre_min'].notna() & df['pre_max'].notna()].copy()
print(f"Entries: {len(close):,}")

close['firmed'] = close['bsp'] < close['pre_min']  # BSP below the best pre-race price
close['drifted'] = close['bsp'] > close['pre_max']  # BSP above the worst pre-race price
close['inside'] = ~close['firmed'] & ~close['drifted']  # BSP inside pre-race range

print(f"  Price firmed at off:  {close['firmed'].sum():>6,} ({close['firmed'].mean()*100:.1f}%)")
print(f"  Price drifted:        {close['drifted'].sum():>6,} ({close['drifted'].mean()*100:.1f}%)")
print(f"  BSP inside range:     {close['inside'].sum():>6,} ({close['inside'].mean()*100:.1f}%)")

# ═══════════════════════════════════════════════════════════════
# STRATEGY: In-play volatility
# ═══════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("IN-PLAY VOLATILITY")
print(SEP)

ip = df[df['ip_min'].notna() & df['ip_max'].notna() & (df['ip_min'] > 0)].copy()
print(f"Entries with in-play data: {len(ip):,}")
if len(ip) > 0:
    ip['ip_range'] = ip['ip_max'] / ip['ip_min']
    print(f"  Median in-play range: {ip['ip_range'].median():.2f}x")
    print(f"  Avg in-play range:    {ip['ip_range'].mean():.2f}x")
    
    # Compare in-play range to pre-race range
    ip = ip[ip['pre_min'].notna() & ip['pre_max'].notna()]
    ip['pre_range'] = ip['pre_max'] / ip['pre_min']
    print(f"\n  In-play vs pre-race range (paired, n={len(ip):,}):")
    print(f"    Pre-race median:  {ip['pre_range'].median():.2f}x")
    print(f"    In-play median:   {ip['ip_range'].median():.2f}x")

# ═══════════════════════════════════════════════════════════════
# SIMULATION: Scalping pre-race range
# ═══════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("SIMULATION: Scalp pre-race range")
print("Back £10 at pre_min, lay £10 at pre_max per horse")
print("When pre_max > pre_min, guaranteed profit if executed")
print(SEP)

# Conservative: only trade where pre_min >= 2.0 and swing >= 20%
qualified = tradable[
    (df.loc[tradable.index, 'bsp'] >= 2.0) &
    (tradable['return_pct'] >= 20)
].copy()
print(f"Qualifying trades (BSP>=2.0, swing>=20%): {len(qualified):,}")

if len(qualified) > 0:
    # For each trade: back £10 at pre_min, lay at pre_max
    # Green book: back X at A, lay X*A/B at B -> profit = X*(A/B - 1)
    # Back £10: profit = 10 * (pre_max/pre_min - 1)
    qualified['profit_10'] = 10 * (qualified['swing'] - 1)
    total_profit = qualified['profit_10'].sum()
    total_bets = len(qualified)
    print(f"  Total gross profit:   £{total_profit:,.0f}")
    print(f"  Per-trade avg profit: £{total_profit/total_bets:.2f}")
    print(f"  Less 5% commission:   £{total_profit*0.95:,.0f}")
    print(f"  Total at risk:        £{total_bets * 10:,.0f}")
    print(f"  ROI (gross):          {total_profit/(total_bets*10)*100:.1f}%")
    print(f"  ROI (net 5% comm):    {total_profit*0.95/(total_bets*10)*100:.1f}%")
    
    # By BSP band
    print(f"\n  By BSP band:")
    for band, g in qualified.groupby('bsp_band', observed=False):
        pft = g['profit_10'].sum()
        n = len(g)
        print(f"    {band:15s} (n={n:>4,}): £{pft:>8,.0f} gross  £{pft/n:.2f}/trade  ROI {pft/(n*10)*100:.0f}%")

# ═══════════════════════════════════════════════════════════════
# END
# ═══════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print(f"Runtime: {time.time()-t0:.1f}s")
print(SEP)
