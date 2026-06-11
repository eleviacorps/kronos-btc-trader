"""
kronos_30m_backtest.py — 30-month historical backtest of all strategies.
Runs VWAP, EMA, ICT/SMC, TA on every 10-min candle across 30 months of BTC data.
Reports daily trade counts, PnL, win rates, and strategy rankings.
"""
import json, sys, os, time, argparse
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import ccxt

PROJECT_DIR = Path(__file__).parent
TIMEFRAME = '5m'
STEP_CANDLES = 2  # every 10 min
LOOKBACK = 100    # candles of context for TA/ICT

parser = argparse.ArgumentParser()
parser.add_argument('--months', type=int, default=30)
parser.add_argument('--save', action='store_true')
args = parser.parse_args()

# ── Fetch 30 months of BTC 5m data ──
print(f"Fetching BTC {TIMEFRAME} data for {args.months} months...")
exchange = ccxt.binance({'options': {'defaultType': 'spot'}, 'timeout': 30000})
exchange.load_markets()
all_candles = []
since = exchange.parse8601((datetime.utcnow() - timedelta(days=args.months*30)).isoformat())
batch = 0
while True:
    try:
        ohlcv = exchange.fetch_ohlcv('BTC/USDT', timeframe=TIMEFRAME, since=since, limit=1000)
        if not ohlcv: break
        all_candles.extend(ohlcv)
        since = ohlcv[-1][0] + 1
        batch += 1
        if batch % 10 == 0:
            print(f"  Fetched {len(all_candles):,} candles ({batch} batches)...")
        time.sleep(0.15)
        if len(all_candles) >= args.months * 30 * 24 * 12:
            break
    except Exception as e:
        print(f"  Fetch error: {e}, retrying...")
        time.sleep(2)

df = pd.DataFrame(all_candles, columns=['timestamp','open','high','low','close','volume'])
df['timestamps'] = pd.to_datetime(df['timestamp'], unit='ms')
df['amount'] = 0.0
df['date'] = df['timestamps'].dt.date
df = df.drop_duplicates(subset='timestamp').sort_values('timestamp').reset_index(drop=True)
print(f"\nLoaded {len(df):,} candles")
print(f"  Range: {df['timestamps'].iloc[0]} → {df['timestamps'].iloc[-1]}")
print(f"  Trading days: {df['date'].nunique()}")
print(f"  Price: ${df['low'].min():,.2f} - ${df['high'].max():,.2f}")

# ── Strategy Functions ──

def vwap_signal(ctx):
    if len(ctx) < 20: return 'HOLD', 0.0
    typical = (ctx['high'] + ctx['low'] + ctx['close']) / 3
    vwap = (typical * ctx['volume']).sum() / ctx['volume'].sum()
    price = float(ctx['close'].iloc[-1])
    dist = (price - vwap) / vwap * 100
    if dist > 0.1: return 'BUY', min(dist/0.5, 0.8)
    if dist < -0.1: return 'SELL', min(abs(dist)/0.5, 0.8)
    return 'HOLD', 0.0

def ema_signal(ctx, fast=9, slow=21):
    if len(ctx) < slow+1: return 'HOLD', 0.0
    c = ctx['close'].values
    ema_f = pd.Series(c).ewm(span=fast).mean().values[-1]
    ema_s = pd.Series(c).ewm(span=slow).mean().values[-1]
    pf = pd.Series(c[:-1]).ewm(span=fast).mean().values[-1]
    ps = pd.Series(c[:-1]).ewm(span=slow).mean().values[-1]
    sp = (ema_f - ema_s)/ema_s*100
    if pf <= ps and ema_f > ema_s: return 'BUY', min(abs(sp)*5, 0.8)
    if pf >= ps and ema_f < ema_s: return 'SELL', min(abs(sp)*5, 0.8)
    if ema_f > ema_s: return 'BUY', 0.2
    if ema_f < ema_s: return 'SELL', 0.2
    return 'HOLD', 0.0

def rsi_signal(ctx, period=14):
    if len(ctx) < period+1: return 'HOLD', 0.0
    c = ctx['close'].values[-period-1:]
    d = np.diff(c); g = np.where(d>0,d,0); l = np.where(d<0,-d,0)
    ag, al = np.mean(g), np.mean(l)
    if al == 0: return 'HOLD', 0.0
    rsi = 100 - 100/(1+ag/al)
    if rsi < 30: return 'BUY', min((30-rsi)/30, 1.0)
    if rsi > 70: return 'SELL', min((rsi-70)/30, 1.0)
    return 'HOLD', 0.0

def ict_ob_signal(ctx):
    if len(ctx) < 20: return 'HOLD', 0.0
    o,c,h,l = [ctx[x].values for x in ['open','close','high','low']]
    avg_body = np.mean(abs(c[-10:]-o[-10:]))
    for i in range(-3, -20, -1):
        br = abs(c[i]-o[i])/(avg_body+1e-10)
        if br > 1.8:
            pb = c[i-1]-o[i-1]
            if c[i] > o[i] and pb < 0 and abs(pb) > avg_body*0.5:
                return 'BUY', min(br/4, 0.8)
            if c[i] < o[i] and pb > 0 and pb > avg_body*0.5:
                return 'SELL', min(br/4, 0.8)
    return 'HOLD', 0.0

def ict_fvg_signal(ctx, lb=20):
    if len(ctx) < lb+3: return 'HOLD', 0.0
    h,l = ctx['high'].values[-lb:], ctx['low'].values[-lb:]
    for i in range(len(h)-3):
        if l[i+2] > h[i]: return 'BUY', min((l[i+2]-h[i])/h[i]*5, 1.0)
        if h[i+2] < l[i]: return 'SELL', min((l[i]-h[i+2])/l[i]*5, 1.0)
    return 'HOLD', 0.0

def ict_sweep_signal(ctx, lb=30):
    if len(ctx) < lb: return 'HOLD', 0.0
    h,l,c = [ctx[x].values[-lb:] for x in ['high','low','close']]
    ph, pl = max(h[:-3]), min(l[:-3])
    lh, ll, lc = max(h[-3:]), min(l[-3:]), c[-1]
    if lh > ph*1.0005 and lc < ph: return 'SELL', min((lh-ph)/ph*200, 0.9)
    if ll < pl*0.9995 and lc > pl: return 'BUY', min((pl-ll)/pl*200, 0.9)
    return 'HOLD', 0.0

def ict_disp_signal(ctx, lb=10):
    if len(ctx) < lb+1: return 'HOLD', 0.0
    c,o = ctx['close'].values, ctx['open'].values
    avg = np.mean(abs(c[-lb:-1]-o[-lb:-1]))
    r = abs(c[-1]-o[-1])/(avg+1e-10)
    if r > 2.0: return ('BUY' if c[-1] > o[-1] else 'SELL'), min(r/5, 0.9)
    return 'HOLD', 0.0

def combo_vote(signals, names, min_agree=2):
    votes = {'BUY': 0, 'SELL': 0}
    for n in names:
        if n in signals:
            s, _ = signals[n]
            if s in votes: votes[s] += 1
    if votes['BUY'] >= min_agree and votes['BUY'] > votes['SELL']: return 'BUY', votes['BUY']/len(names)
    if votes['SELL'] >= min_agree and votes['SELL'] > votes['BUY']: return 'SELL', votes['SELL']/len(names)
    return 'HOLD', 0.0

def combo_confirm(primary, secondaries, signals):
    if primary not in signals: return 'HOLD', 0.0
    ps, pc = signals[primary]
    if ps == 'HOLD': return 'HOLD', 0.0
    for sn in secondaries:
        if sn in signals and signals[sn][0] == ps: return ps, min(pc*1.2, 1.0)
    return 'HOLD', 0.0

# ── Register all strategies ──
ALL_STRATS = {
    'vwap': vwap_signal,
    'ema_cross': ema_signal,
    'rsi': rsi_signal,
    'ict_ob': ict_ob_signal,
    'ict_fvg': ict_fvg_signal,
    'ict_sweep': ict_sweep_signal,
    'ict_disp': ict_disp_signal,
}

# ── Evaluation loop ──
first_idx = LOOKBACK + 1
indices = list(range(first_idx, len(df) - 1, STEP_CANDLES))
print(f"\nEvaluating {len(indices):,} windows (every {STEP_CANDLES*5}min)...")

# Track per-strategy and daily
strategy_pnl = {}  # name -> list of (date, pnl_pct)
start_time = time.time()

for batch, idx in enumerate(indices):
    if batch % 5000 == 0:
        pct = batch / len(indices) * 100
        eta = (time.time()-start_time)/(batch+1)*(len(indices)-batch) if batch > 0 else 0
        print(f"  [{batch:,}/{len(indices):,}] {pct:.0f}% ETA: {eta:.0f}s")

    ctx = df.iloc[idx-LOOKBACK:idx]
    current_price = float(ctx['close'].iloc[-1])
    current_date = ctx['timestamps'].iloc[-1].date()
    
    # 20-min forward return
    fwd_idx = min(idx + 4, len(df) - 1)
    actual_return = ((df.iloc[fwd_idx]['close'] - current_price) / current_price) * 100
    
    # Compute signals
    signals = {}
    for sname, sfn in ALL_STRATS.items():
        try:
            sig, conf = sfn(ctx)
            signals[sname] = (sig, conf)
        except:
            signals[sname] = ('HOLD', 0.0)
    
    # Combos
    signals['combo_vwap_ema'] = combo_confirm('vwap', ['ema_cross'], signals)
    signals['combo_vwap_ict'] = combo_confirm('vwap', ['ict_ob', 'ict_fvg', 'ict_sweep'], signals)
    signals['combo_ta_vote'] = combo_vote(signals, ['vwap', 'ema_cross', 'rsi'], min_agree=2)
    signals['combo_smc_vote'] = combo_vote(signals, ['ict_ob', 'ict_fvg', 'ict_sweep', 'ict_disp'], min_agree=2)
    signals['combo_all_vote'] = combo_vote(signals, list(ALL_STRATS.keys()), min_agree=4)
    
    # Evaluate PnL
    for sname, (sig, conf) in signals.items():
        if sig in ('BUY', 'SELL'):
            pnl = actual_return if sig == 'BUY' else -actual_return
            strategy_pnl.setdefault(sname, []).append((current_date, pnl))

elapsed = time.time() - start_time
print(f"\nEvaluation complete: {elapsed:.0f}s ({len(indices):,} windows)")

# ── Aggregate results ──
print(f"\n{'='*75}")
print(f"  30-MONTH BACKTEST RESULTS")
print(f"{'='*75}")
print(f"  Period: {df['timestamps'].iloc[0].date()} → {df['timestamps'].iloc[-1].date()}")
print(f"  Windows: {len(indices):,} (every {STEP_CANDLES*5}min)")
print(f"  Trading days: {df['date'].nunique()}")
print(f"{'='*75}")

# Overall strategy ranking
print(f"\n{'─'*75}")
print(f"  OVERALL STRATEGY RANKING (30 months)")
print(f"{'─'*75}")
print(f"  {'Strategy':<22s} {'Trades':>8s} {'PnL%':>10s} {'Win%':>7s} {'Avg':>8s} {'Trades/Day':>11s}")
print(f"  {'-'*66}")

sorted_strats = sorted(strategy_pnl.items(), key=lambda x: sum(p for _, p in x[1]), reverse=True)
for sname, trades in sorted_strats:
    pnls = [p for _, p in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    wr = len(wins)/len(pnls)*100 if pnls else 0
    total_pnl = sum(pnls)
    avg_pnl = np.mean(pnls) if pnls else 0
    trades_per_day = len(trades) / df['date'].nunique()
    print(f"  {sname:<22s} {len(trades):>8,} {total_pnl:>+9.2f}% {wr:>6.1f}% {avg_pnl:>+7.2f}% {trades_per_day:>10.1f}")

# By-month breakdown for top 5
print(f"\n{'─'*75}")
print(f"  MONTHLY BREAKDOWN — TOP 5 STRATEGIES")
print(f"{'─'*75}")

top5_names = [s[0] for s in sorted_strats[:5]]
df['year_month'] = df['timestamps'].dt.to_period('M')
months = sorted(df['year_month'].unique())

# Header
header = f"  {'Month':<10s}"
for n in top5_names:
    header += f" {n[:12]:>13s}"
header += f" {'BuyHold':>10s}"
print(header)
print(f"  {'-'*75}")

for ym in months:
    month_trades = {}
    for sname in top5_names:
        month_trades[sname] = [p for d, p in strategy_pnl[sname] if pd.Timestamp(d).to_period('M') == ym]
    
    # Buy & hold for the month
    month_data = df[df['year_month'] == ym]
    buy_hold = ((month_data['close'].iloc[-1] - month_data['close'].iloc[0]) / month_data['close'].iloc[0]) * 100 if len(month_data) > 0 else 0
    
    line = f"  {str(ym):<10s}"
    for sname in top5_names:
        pnls = month_trades[sname]
        if pnls:
            line += f" {sum(pnls):>+12.2f}%"
        else:
            line += f" {'N/A':>12s}"
    line += f" {buy_hold:>+9.2f}%"
    print(line)

# Summary stats for top 3
print(f"\n{'─'*75}")
print(f"  TOP 3 STRATEGIES — DETAILED STATS")
print(f"{'─'*75}")
for sname, trades in sorted_strats[:3]:
    pnls = [p for _, p in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    wr = len(wins)/len(pnls)*100 if pnls else 0
    total_pnl = sum(pnls)
    
    # Monthly avg
    monthly_pnls = {}
    for d, p in trades:
        ym = pd.Timestamp(d).to_period('M')
        monthly_pnls.setdefault(ym, []).append(p)
    avg_monthly = np.mean([sum(v) for v in monthly_pnls.values()]) if monthly_pnls else 0
    
    # Max drawdown (simplified: consecutive losses)
    peak = 0
    max_dd = 0
    running = 0
    for p in pnls:
        running += p
        if running > peak: peak = running
        dd = peak - running
        if dd > max_dd: max_dd = dd
    
    print(f"\n  {sname}")
    print(f"    Total trades:     {len(trades):,}")
    print(f"    Total PnL:        ${total_pnl:+.2f} on $100")
    print(f"    Win rate:         {wr:.1f}%")
    print(f"    Avg win:          {np.mean(wins):+.2f}%" if wins else "    Avg win:          N/A")
    print(f"    Avg loss:         {np.mean(losses):+.2f}%" if losses else "    Avg loss:         N/A")
    print(f"    Avg monthly:      {avg_monthly:+.2f}%")
    print(f"    Max drawdown:     {max_dd:.2f}%")
    print(f"    Trades per day:   {len(trades)/df['date'].nunique():.1f}")
    print(f"    Profit factor:    {abs(sum(wins)/sum(losses)):.2f}" if losses and sum(losses) != 0 else "    Profit factor:    ∞")
    print(f"    Best month:       {max(sum(v) for v in monthly_pnls.values()):+.2f}%")
    print(f"    Worst month:      {min(sum(v) for v in monthly_pnls.values()):+.2f}%")

print(f"\n{'='*75}")
print(f"  BACKTEST COMPLETE")
print(f"{'='*75}")

# Save
if args.save:
    out = {
        'config': {'months': args.months, 'windows': len(indices), 'interval_min': STEP_CANDLES*5},
        'strategies': {sname: {'total_pnl_pct': round(float(sum(p for _, p in trades)), 2),
                                'trades': len(trades),
                                'win_rate': round(len([p for _, p in trades if p > 0])/len(trades)*100, 1)
                               } for sname, trades in strategy_pnl.items()}
    }
    json.dump(out, open(PROJECT_DIR/'bt_30m_results.json', 'w'), indent=2)
    print(f"  Results saved to bt_30m_results.json")
