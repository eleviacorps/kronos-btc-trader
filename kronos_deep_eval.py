"""
kronos_deep_eval.py — Comprehensive Kronos model prediction accuracy + strategy evaluation.
Tests the model's actual prediction quality, then finds optimal strategy parameters.
"""
import json, sys, os, time, argparse
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import torch
import ccxt

sys.path.insert(0, '.')
from model import Kronos, KronosTokenizer, KronosPredictor

PROJECT_DIR = Path(__file__).parent
TIMEFRAME = '5m'
CONTEXT = 400
PRED_LEN = 6
TOTAL_WINDOWS = 100
STEP = 6  # every 30 min

parser = argparse.ArgumentParser()
parser.add_argument('--windows', type=int, default=TOTAL_WINDOWS)
parser.add_argument('--step', type=int, default=STEP)
parser.add_argument('--save', action='store_true', help='Save detailed results')
args = parser.parse_args()

# ── Fetch Data ──
print("Fetching BTC/USDT 5m data...")
exchange = ccxt.binance({'options': {'defaultType': 'spot'}, 'timeout': 30000})
exchange.load_markets()
all_candles = []
since = exchange.parse8601((datetime.utcnow() - timedelta(days=14)).isoformat())
while True:
    ohlcv = exchange.fetch_ohlcv('BTC/USDT', timeframe=TIMEFRAME, since=since, limit=1000)
    if not ohlcv: break
    all_candles.extend(ohlcv)
    since = ohlcv[-1][0] + 1
    time.sleep(0.3)
    if len(all_candles) >= 4000: break
df = pd.DataFrame(all_candles, columns=['timestamp','open','high','low','close','volume'])
df['timestamps'] = pd.to_datetime(df['timestamp'], unit='ms')
df['amount'] = 0.0
df = df.drop_duplicates(subset='timestamp').sort_values('timestamp').reset_index(drop=True)
print(f"Loaded {len(df)} candles ({df['timestamps'].iloc[0]} → {df['timestamps'].iloc[-1]})")

# ── Load Model ──
print("Loading Kronos model...")
device = 'cuda:0'
tok = KronosTokenizer.from_pretrained('NeoQuasar/Kronos-Tokenizer-base').to(device)
model = Kronos.from_pretrained('NeoQuasar/Kronos-base').to(device)
predictor = KronosPredictor(model, tok, max_context=512)
print(f"Model loaded on {device}")

# ── Prediction points ──
first = CONTEXT + 1
step_idx = args.step
indices = list(range(first, len(df) - PRED_LEN - 1, step_idx))
print(f"Evaluating {len(indices)} windows (step={step_idx} candles, every {step_idx*5}min)")

# ── Storage ──
records = []
strategy_pnl = {}  # name -> list of trade pnl_pct

# ── Strategy Functions (mirror kronos_trader.py logic) ──
def antitrend_signal(net_change_pct, range_pct, extreme=0.10, min_range=0.15):
    if range_pct < min_range: return 'HOLD', 0.0
    if net_change_pct >= extreme: return 'SELL', min(net_change_pct/0.4, 0.85)
    if net_change_pct <= -extreme: return 'BUY', min(abs(net_change_pct)/0.4, 0.85)
    return 'HOLD', 0.0

def trend_momentum_signal(net_change_pct, range_pct, buy_th=0.15, sell_th=-0.15, min_range=0.25):
    if range_pct < min_range: return 'HOLD', 0.0
    if net_change_pct > buy_th: return 'BUY', min(net_change_pct/0.4, 1.0)
    if net_change_pct < sell_th: return 'SELL', min(abs(net_change_pct)/0.4, 1.0)
    return 'HOLD', 0.0

def momentum_slope_signal(closes, range_pct, slope_th=0.015, min_range=0.25):
    if range_pct < min_range: return 'HOLD', 0.0
    x = np.arange(len(closes))
    slope = np.polyfit(x, closes, 1)[0]
    slope_pct = (slope / closes[0]) * 100
    if slope_pct > slope_th: return 'BUY', min(slope_pct/0.08, 1.0)
    if slope_pct < -slope_th: return 'SELL', min(abs(slope_pct)/0.08, 1.0)
    return 'HOLD', 0.0

def candle_progression_signal(closes, range_pct, min_consensus=5, min_range=0.25):
    if range_pct < min_range: return 'HOLD', 0.0
    diffs = np.diff(closes)
    bulls = int(np.sum(diffs > 0))
    bears = int(np.sum(diffs < 0))
    total = len(diffs)
    if bulls >= min_consensus: return 'BUY', bulls/total
    if bears >= min_consensus: return 'SELL', bears/total
    return 'HOLD', 0.0

def range_extension_signal(pred_high, pred_low, current, range_pct, ext_pct=0.30, min_range=0.25):
    if range_pct < min_range: return 'HOLD', 0.0
    high_ext = ((pred_high - current) / current) * 100
    low_ext = ((pred_low - current) / current) * 100
    signals = []
    if high_ext > ext_pct: signals.append(('BUY', high_ext))
    if low_ext < -ext_pct: signals.append(('SELL', abs(low_ext)))
    if not signals: return 'HOLD', 0.0
    best = max(signals, key=lambda x: x[1])
    return best[0], min(best[1]/0.8, 1.0)

# ── Variants to test ──
VARIANTS = {
    'antitrend_agg':     {'fn': antitrend_signal, 'args': {'extreme': 0.04, 'min_range': 0.08}},
    'antitrend_std':     {'fn': antitrend_signal, 'args': {'extreme': 0.10, 'min_range': 0.15}},
    'antitrend_con':     {'fn': antitrend_signal, 'args': {'extreme': 0.20, 'min_range': 0.20}},
    'trend_momentum':    {'fn': trend_momentum_signal, 'args': {'buy_th': 0.15, 'sell_th': -0.15, 'min_range': 0.25}},
    'momentum_slope':    {'fn': momentum_slope_signal, 'args': {'slope_th': 0.015, 'min_range': 0.25}},
    'candle_prog':       {'fn': candle_progression_signal, 'args': {'min_consensus': 5, 'min_range': 0.25}},
    'range_ext':         {'fn': range_extension_signal, 'args': {'ext_pct': 0.30, 'min_range': 0.25}},
}

# ── Main evaluation loop ──
start_time = time.time()
print(f"\nEvaluating {len(indices)} windows...")

for batch, idx in enumerate(indices):
    if batch % 20 == 0:
        pct = batch / len(indices) * 100
        eta = (time.time()-start_time)/(batch+1)*(len(indices)-batch) if batch > 0 else 0
        print(f"  [{batch}/{len(indices)}] {pct:.0f}%  ETA: {eta:.0f}s")
    
    ctx = df.iloc[idx-CONTEXT:idx].copy()
    x_df = ctx[['open','high','low','close','volume','amount']]
    x_ts = ctx['timestamps']
    current_price = float(ctx['close'].iloc[-1])
    current_time = str(ctx['timestamps'].iloc[-1])
    
    y_ts = pd.Series(pd.date_range(
        start=x_ts.iloc[-1] + pd.Timedelta(minutes=5),
        periods=PRED_LEN, freq='5min'))
    
    actual = df.iloc[idx:idx+PRED_LEN]
    if len(actual) < PRED_LEN: continue
    actual_return = ((actual['close'].iloc[-1] - current_price) / current_price) * 100
    
    try:
        with torch.no_grad():
            pred_df = predictor.predict(
                df=x_df, x_timestamp=x_ts, y_timestamp=y_ts,
                pred_len=PRED_LEN, T=1.0, top_p=0.9,
                sample_count=5, verbose=False)
    except Exception as e:
        print(f"  ⚠️  Pred error at idx {idx}: {e}")
        continue
    
    pred_close = float(pred_df['close'].iloc[-1])
    net_change = ((pred_close - current_price) / current_price) * 100
    pred_range = ((pred_df['high'].max() - pred_df['low'].min()) / current_price) * 100
    pred_closes = pred_df['close'].values
    
    # Model accuracy record
    rec = {
        'idx': idx, 'time': current_time, 'price': current_price,
        'predicted_change': round(net_change, 3),
        'predicted_range': round(pred_range, 3),
        'actual_return': round(actual_return, 3),
        'direction_correct': (net_change > 0 and actual_return > 0) or (net_change < 0 and actual_return < 0),
    }
    records.append(rec)
    
    # Evaluate each strategy variant
    for vname, vdef in VARIANTS.items():
        fn = vdef['fn']
        vargs = vdef['args'].copy()
        if 'closes' in fn.__code__.co_varnames:
            sig, conf = fn(pred_closes, pred_range, **vargs)
        elif 'pred_high' in fn.__code__.co_varnames:
            sig, conf = fn(pred_df['high'].max(), pred_df['low'].min(), current_price, pred_range, **vargs)
        else:
            sig, conf = fn(net_change, pred_range, **vargs)
        
        if sig in ('BUY', 'SELL'):
            pnl = actual_return if sig == 'BUY' else -actual_return
            if vname not in strategy_pnl:
                strategy_pnl[vname] = []
            strategy_pnl[vname].append(pnl)

# ── Results ──
elapsed = time.time() - start_time
print(f"\n{'='*65}")
print(f"  KRONOS MODEL ACCURACY — {elapsed:.0f}s ({len(indices)} windows)")
print(f"{'='*65}")

# Direction accuracy
correct = sum(1 for r in records if r['direction_correct'])
total_pred = len(records)
direction_acc = correct / total_pred * 100 if total_pred else 0
print(f"\n📊 MODEL DIRECTION ACCURACY: {direction_acc:.1f}% ({correct}/{total_pred})")

# Accuracy by predicted magnitude
buckets = [('strong_bearish', -999, -0.15), ('bearish', -0.15, -0.05),
           ('neutral', -0.05, 0.05), ('bullish', 0.05, 0.15), ('strong_bullish', 0.15, 999)]
print(f"\n{'─'*60}")
print(f"  ACCURACY BY PREDICTION STRENGTH")
print(f"{'─'*60}")
print(f"  {'Bucket':<18s} {'Count':>6s} {'Correct':>8s} {'Acc%':>6s}")
print(f"  {'-'*40}")
for label, lo, hi in buckets:
    group = [r for r in records if lo <= r['predicted_change'] < hi]
    if group:
        c = sum(1 for r in group if r['direction_correct'])
        print(f"  {label:<18s} {len(group):>6d} {c:>8d} {c/len(group)*100:>5.1f}%")
    else:
        print(f"  {label:<18s} {'0':>6s} {'0':>8s} {'N/A':>6s}")

# Signal distribution
buy_signals = sum(1 for s in strategy_pnl.get('antitrend_std', []) if True)
print(f"\n{'─'*60}")
print(f"  STRATEGY PERFORMANCE")
print(f"{'─'*60}")
print(f"  {'Strategy':<22s} {'Trades':>7s} {'PnL%':>10s} {'Win%':>7s} {'AvgWin':>9s} {'AvgLoss':>9s}")
print(f"  {'-'*58}")
sorted_strats = sorted(strategy_pnl.items(), key=lambda x: np.sum(x[1]), reverse=True)
for vname, pnls in sorted_strats:
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    wr = len(wins)/len(pnls)*100 if pnls else 0
    avg_w = np.mean(wins) if wins else 0
    avg_l = np.mean(losses) if losses else 0
    print(f"  {vname:<22s} {len(pnls):>7d} {np.sum(pnls):>+9.2f}% {wr:>6.1f}% {avg_w:>+8.2f}% {avg_l:>+8.2f}%")

# Best variant recommendation
print(f"\n{'─'*60}")
top_strat = max(strategy_pnl.items(), key=lambda x: np.sum(x[1]))
print(f"  🏆 BEST STRATEGY: {top_strat[0]} ({np.sum(top_strat[1]):+.2f}% total)")

# Save
if args.save:
    out = {'accuracy': {'windows': total_pred, 'direction_accuracy_pct': round(direction_acc, 1), 'by_strength': {}},
           'strategies': {k: {'total_pnl_pct': round(float(np.sum(v)), 2), 'trades': len(v),
                              'win_rate': round(len([p for p in v if p>0])/len(v)*100, 1) if v else 0}
                         for k, v in strategy_pnl.items()}}
    path = PROJECT_DIR / 'deep_eval_results.json'
    json.dump(out, open(path, 'w'), indent=2)
    print(f"  Results saved to {path}")
    
print(f"{'='*65}")
print(f"  EVALUATION COMPLETE")
print(f"{'='*65}")
