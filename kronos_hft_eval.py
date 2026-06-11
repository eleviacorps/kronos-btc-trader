"""
kronos_hft_eval.py — HFT Scalp strategy evaluation (every 10 min).
Tests Kronos-based, ICT/SMC, and TA strategies + combinations to find the best scalp setup.
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
CONTEXT = 200
PRED_LEN = 4
TOTAL_WINDOWS = 400
STEP = 2

parser = argparse.ArgumentParser()
parser.add_argument('--windows', type=int, default=TOTAL_WINDOWS)
parser.add_argument('--save', action='store_true')
args = parser.parse_args()

print("Fetching BTC 5m data...")
exchange = ccxt.binance({'options': {'defaultType': 'spot'}, 'timeout': 30000})
exchange.load_markets()
all_candles = []
since = exchange.parse8601((datetime.utcnow() - timedelta(days=10)).isoformat())
while True:
    ohlcv = exchange.fetch_ohlcv('BTC/USDT', timeframe=TIMEFRAME, since=since, limit=1000)
    if not ohlcv: break
    all_candles.extend(ohlcv)
    since = ohlcv[-1][0] + 1
    time.sleep(0.3)
    if len(all_candles) >= 3000: break
df = pd.DataFrame(all_candles, columns=['timestamp','open','high','low','close','volume'])
df['timestamps'] = pd.to_datetime(df['timestamp'], unit='ms')
df['amount'] = 0.0
df = df.drop_duplicates(subset='timestamp').sort_values('timestamp').reset_index(drop=True)
print(f"Loaded {len(df)} candles ({df['timestamps'].iloc[0]} → {df['timestamps'].iloc[-1]})")

first = CONTEXT + 1
step_idx = STEP
all_indices = list(range(first, len(df) - PRED_LEN - 1, step_idx))
indices = all_indices[:args.windows]
print(f"Evaluating {len(indices)} HFT windows (every {step_idx*5}min)")

# ── Strategy Functions ──

# === KRONOS-BASED STRATEGIES (scalp params) ===
def k_range_ext(pred_high, pred_low, current, range_pct, ext=0.10, min_r=0.08):
    if range_pct < min_r: return 'HOLD', 0.0
    he = ((pred_high - current)/current)*100
    le = ((pred_low - current)/current)*100
    sigs = []
    if he > ext: sigs.append(('BUY', he))
    if le < -ext: sigs.append(('SELL', abs(le)))
    if not sigs: return 'HOLD', 0.0
    b = max(sigs, key=lambda x: x[1])
    return b[0], min(b[1]/0.8, 1.0)

def k_trend_mom(net, range_pct, bt=0.06, st=-0.06, mr=0.08):
    if range_pct < mr: return 'HOLD', 0.0
    if net > bt: return 'BUY', min(net/0.4, 1.0)
    if net < st: return 'SELL', min(abs(net)/0.4, 1.0)
    return 'HOLD', 0.0

def k_mom_slope(closes, range_pct, sl=0.008, mr=0.08):
    if range_pct < mr: return 'HOLD', 0.0
    x = np.arange(len(closes))
    slope = np.polyfit(x, closes, 1)[0]
    sp = (slope/closes[0])*100
    if sp > sl: return 'BUY', min(sp/0.08, 1.0)
    if sp < -sl: return 'SELL', min(abs(sp)/0.08, 1.0)
    return 'HOLD', 0.0

# === ICT / SMC STRATEGIES ===
def ict_ob(ctx_df, direction=None):
    """Order Block detection. Looks for last significant reversal candle.
    Returns 'BUY' if bullish OB, 'SELL' if bearish OB, 'HOLD' otherwise."""
    if len(ctx_df) < 20: return 'HOLD', 0.0
    closes = ctx_df['close'].values
    highs = ctx_df['high'].values
    lows = ctx_df['low'].values
    opens = ctx_df['open'].values
    bodies = abs(closes - opens)
    avg_body = np.mean(bodies[-10:])
    # Find last 3-candle displacement: a strong move -> check preceding candle
    for i in range(-3, -20, -1):
        body_ratio = bodies[i] / (avg_body + 1e-10)
        if body_ratio > 1.8:
            prev_body = closes[i-1] - opens[i-1]
            if closes[i] > opens[i]:  # Bullish displacement
                if prev_body < 0 and abs(prev_body) > avg_body * 0.5:
                    # Bearish candle before bullish breakout = bullish OB
                    return 'BUY', min(body_ratio/4.0, 0.8)
            else:  # Bearish displacement
                if prev_body > 0 and prev_body > avg_body * 0.5:
                    # Bullish candle before bearish breakout = bearish OB
                    return 'SELL', min(body_ratio/4.0, 0.8)
    return 'HOLD', 0.0

def ict_fvg(ctx_df, lookback=20):
    """Fair Value Gap: gap between candle 1's wick and candle 3's wick on other side."""
    if len(ctx_df) < lookback + 3: return 'HOLD', 0.0
    for i in range(-lookback, -2):
        c1_low, c1_high = ctx_df['low'].iloc[i], ctx_df['high'].iloc[i]
        c2_low, c2_high = ctx_df['low'].iloc[i+1], ctx_df['high'].iloc[i+1]
        c3_low, c3_high = ctx_df['low'].iloc[i+2], ctx_df['high'].iloc[i+2]
        # Bullish FVG: c3 low > c1 high (gap up after gap up)
        if c3_low > c1_high:
            gap_pct = (c3_low - c1_high) / c1_high * 100
            if gap_pct > 0.02:
                return 'BUY', min(gap_pct/0.2, 1.0)
        # Bearish FVG: c3 high < c1 low (gap down after gap down)
        if c3_high < c1_low:
            gap_pct = (c1_low - c3_high) / c1_low * 100
            if gap_pct > 0.02:
                return 'SELL', min(gap_pct/0.2, 1.0)
    return 'HOLD', 0.0

def ict_liquidity_sweep(ctx_df, lookback=30):
    """Liquidity sweep: price breaks recent swing high/low then reverses within 3 candles."""
    if len(ctx_df) < lookback: return 'HOLD', 0.0
    highs = ctx_df['high'].values[-lookback:]
    lows = ctx_df['low'].values[-lookback:]
    closes = ctx_df['close'].values[-lookback:]
    opens = ctx_df['open'].values[-lookback:]
    prev_high = max(highs[:-3])
    prev_low = min(lows[:-3])
    last_high = max(highs[-3:])
    last_low = min(lows[-3:])
    last_close = closes[-1]
    # Bearish sweep: price broke above prev high, closed back below
    if last_high > prev_high * 1.0005 and last_close < prev_high:
        sweep_pct = (last_high - prev_high) / prev_high * 100
        return 'SELL', min(sweep_pct * 2, 0.9)
    # Bullish sweep: price broke below prev low, closed back above
    if last_low < prev_low * 0.9995 and last_close > prev_low:
        sweep_pct = (prev_low - last_low) / prev_low * 100
        return 'BUY', min(sweep_pct * 2, 0.9)
    return 'HOLD', 0.0

def ict_displacement(ctx_df, lookback=10):
    """Displacement: strong momentum candle (body > 2x avg)."""
    if len(ctx_df) < lookback + 1: return 'HOLD', 0.0
    bodies = abs(ctx_df['close'].values - ctx_df['open'].values)
    avg_body = np.mean(bodies[-lookback:-1])
    last_body = bodies[-1]
    last_close = ctx_df['close'].iloc[-1]
    last_open = ctx_df['open'].iloc[-1]
    ratio = last_body / (avg_body + 1e-10)
    if ratio > 2.0:
        if last_close > last_open:
            return 'BUY', min(ratio / 5.0, 0.9)
        else:
            return 'SELL', min(ratio / 5.0, 0.9)
    return 'HOLD', 0.0

# === TECHNICAL INDICATOR STRATEGIES ===
def ta_rsi(ctx_df, period=14):
    """RSI overbought/oversold. Oversold < 30 → BUY, overbought > 70 → SELL."""
    if len(ctx_df) < period + 1: return 'HOLD', 0.0
    closes = ctx_df['close'].values[-period-1:]
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains)
    avg_loss = np.mean(losses)
    if avg_loss == 0: return 'HOLD', 0.0
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    if rsi < 30: return 'BUY', min((30 - rsi) / 30, 1.0)
    if rsi > 70: return 'SELL', min((rsi - 70) / 30, 1.0)
    return 'HOLD', 0.0

def ta_vwap(ctx_df):
    """VWAP: price above VWAP = bullish bias, below = bearish."""
    if len(ctx_df) < 20: return 'HOLD', 0.0
    typical = (ctx_df['high'] + ctx_df['low'] + ctx_df['close']) / 3
    vwap = (typical * ctx_df['volume']).sum() / ctx_df['volume'].sum()
    last_close = ctx_df['close'].iloc[-1]
    dist = (last_close - vwap) / vwap * 100
    if dist > 0.1: return 'BUY', min(dist/0.5, 0.8)
    if dist < -0.1: return 'SELL', min(abs(dist)/0.5, 0.8)
    return 'HOLD', 0.0

def ta_ema_cross(ctx_df, fast=9, slow=21):
    """EMA crossover: fast above slow → BUY, below → SELL."""
    if len(ctx_df) < slow + 1: return 'HOLD', 0.0
    closes = ctx_df['close'].values
    ema_f = pd.Series(closes).ewm(span=fast).mean().values[-1]
    ema_s = pd.Series(closes).ewm(span=slow).mean().values[-1]
    prev_f = pd.Series(closes[:-1]).ewm(span=fast).mean().values[-1]
    prev_s = pd.Series(closes[:-1]).ewm(span=slow).mean().values[-1]
    # Crossover just happened
    if prev_f <= prev_s and ema_f > ema_s:
        return 'BUY', min((ema_f - ema_s) / ema_s * 100, 0.8)
    if prev_f >= prev_s and ema_f < ema_s:
        return 'SELL', min((ema_s - ema_f) / ema_f * 100, 0.8)
    # Already crossed: bias
    if ema_f > ema_s: return 'BUY', 0.15
    if ema_f < ema_s: return 'SELL', 0.15
    return 'HOLD', 0.0

# === COMBINATION STRATEGIES ===
def combo_vote(results_dict, strategies, min_agreement=2):
    """Voting-based combo: take the signal that has majority agreement among listed strategies."""
    votes = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
    total_conf = {'BUY': 0.0, 'SELL': 0.0, 'HOLD': 0.0}
    for name in strategies:
        if name in results_dict:
            sig, conf = results_dict[name]
            votes[sig] = votes.get(sig, 0) + 1
            total_conf[sig] = total_conf.get(sig, 0) + conf
    if votes['BUY'] >= min_agreement and votes['BUY'] > votes['SELL']:
        avg_c = total_conf['BUY'] / votes['BUY']
        return 'BUY', avg_c
    if votes['SELL'] >= min_agreement and votes['SELL'] > votes['BUY']:
        avg_c = total_conf['SELL'] / votes['SELL']
        return 'SELL', avg_c
    return 'HOLD', 0.0

def combo_confirmation(primary_name, secondary_names, results_dict):
    """Primary signal requires at least one secondary to confirm."""
    if primary_name not in results_dict:
        return 'HOLD', 0.0
    p_sig, p_conf = results_dict[primary_name]
    if p_sig == 'HOLD':
        return 'HOLD', 0.0
    for sn in secondary_names:
        if sn in results_dict:
            s_sig, _ = results_dict[sn]
            if s_sig == p_sig:
                return p_sig, p_conf * 1.2  # boost confidence
    return 'HOLD', 0.0

# ── Load Kronos Model ──
print("Loading Kronos model...")
device = 'cuda:0'
tok = KronosTokenizer.from_pretrained('NeoQuasar/Kronos-Tokenizer-base').to(device)
model = Kronos.from_pretrained('NeoQuasar/Kronos-base').to(device)
predictor = KronosPredictor(model, tok, max_context=512)
print(f"Model loaded on {device}")

# ── Strategy registry ──
ALL_STRATS = {
    'k_range_ext': {'fn': k_range_ext, 'type': 'kronos'},
    'k_trend_mom': {'fn': k_trend_mom, 'type': 'kronos'},
    'k_mom_slope': {'fn': k_mom_slope, 'type': 'kronos'},
    'ict_ob': {'fn': ict_ob, 'type': 'price'},
    'ict_fvg': {'fn': ict_fvg, 'type': 'price'},
    'ict_liquidity': {'fn': ict_liquidity_sweep, 'type': 'price'},
    'ict_displacement': {'fn': ict_displacement, 'type': 'price'},
    'ta_rsi': {'fn': ta_rsi, 'type': 'price'},
    'ta_vwap': {'fn': ta_vwap, 'type': 'price'},
    'ta_ema_cross': {'fn': ta_ema_cross, 'type': 'price'},
}

# ── Storage ──
records = []
strategy_pnl = {}
combo_pnl = {}

start_time = time.time()
print(f"\nEvaluating {len(indices)} HFT windows...")

for batch, idx in enumerate(indices):
    if batch % 30 == 0:
        pct = batch / len(indices) * 100
        eta = (time.time()-start_time)/(batch+1)*(len(indices)-batch) if batch > 0 else 0
        print(f"  [{batch}/{len(indices)}] {pct:.0f}% ETA: {eta:.0f}s")

    ctx = df.iloc[idx-CONTEXT:idx].copy()
    x_df = ctx[['open','high','low','close','volume','amount']]
    x_ts = ctx['timestamps']
    current_price = float(ctx['close'].iloc[-1])
    current_time = str(ctx['timestamps'].iloc[-1])

    y_ts = pd.Series(pd.date_range(start=x_ts.iloc[-1]+pd.Timedelta(minutes=5),
                                    periods=PRED_LEN, freq='5min'))
    actual = df.iloc[idx:idx+PRED_LEN]
    if len(actual) < PRED_LEN: continue
    actual_return = ((actual['close'].iloc[-1] - current_price) / current_price) * 100

    # Kronos prediction
    try:
        with torch.no_grad():
            pred_df = predictor.predict(df=x_df, x_timestamp=x_ts, y_timestamp=y_ts,
                                         pred_len=PRED_LEN, T=1.0, top_p=0.9,
                                         sample_count=5, verbose=False)
    except Exception as e:
        print(f"  ⚠️  Pred error at idx {idx}: {e}")
        continue

    pred_close = float(pred_df['close'].iloc[-1])
    net_change = ((pred_close - current_price) / current_price) * 100
    pred_range = ((pred_df['high'].max() - pred_df['low'].min()) / current_price) * 100
    pred_closes = pred_df['close'].values

    # Compute all strategy signals
    signals = {}

    # Kronos-based
    sig, conf = k_range_ext(pred_df['high'].max(), pred_df['low'].min(), current_price, pred_range)
    signals['k_range_ext'] = (sig, conf)
    sig, conf = k_trend_mom(net_change, pred_range)
    signals['k_trend_mom'] = (sig, conf)
    sig, conf = k_mom_slope(pred_closes, pred_range)
    signals['k_mom_slope'] = (sig, conf)

    # ICT/SMC (price-based, from context)
    sig, conf = ict_ob(ctx)
    signals['ict_ob'] = (sig, conf)
    sig, conf = ict_fvg(ctx)
    signals['ict_fvg'] = (sig, conf)
    sig, conf = ict_liquidity_sweep(ctx)
    signals['ict_liquidity'] = (sig, conf)
    sig, conf = ict_displacement(ctx)
    signals['ict_displacement'] = (sig, conf)

    # TA Indicators
    sig, conf = ta_rsi(ctx)
    signals['ta_rsi'] = (sig, conf)
    sig, conf = ta_vwap(ctx)
    signals['ta_vwap'] = (sig, conf)
    sig, conf = ta_ema_cross(ctx)
    signals['ta_ema_cross'] = (sig, conf)

    # Combos
    # Combo 1: Kronos trend + ICT confirmation
    signals['combo_kronos_ict'] = combo_confirmation('k_trend_mom', ['ict_ob', 'ict_fvg', 'ict_liquidity'], signals)
    # Combo 2: Range ext + displacement
    signals['combo_range_displace'] = combo_confirmation('k_range_ext', ['ict_displacement'], signals)
    # Combo 3: All-price vote (ICT + TA)
    signals['combo_price_vote'] = combo_vote(signals, ['ict_ob', 'ict_fvg', 'ict_liquidity', 'ict_displacement', 'ta_rsi', 'ta_vwap'], min_agreement=3)
    # Combo 4: Kronos + EMA (trend filter)
    signals['combo_kronos_ema'] = combo_confirmation('k_trend_mom', ['ta_ema_cross'], signals)
    # Combo 5: Range ext + liquidity sweep (best price action signals)
    signals['combo_range_sweep'] = combo_confirmation('k_range_ext', ['ict_liquidity', 'ict_displacement'], signals)
    # Combo 6: All-in vote (all strategies)
    signals['combo_all_vote'] = combo_vote(signals, list(ALL_STRATS.keys()), min_agreement=4)

    # Record
    rec = {'idx': idx, 'time': current_time, 'price': current_price,
           'net_change': round(net_change, 3), 'pred_range': round(pred_range, 3),
           'actual_return': round(actual_return, 3)}
    records.append(rec)

    # Evaluate PnL for each strategy
    for sname, (sig, conf) in signals.items():
        if sig in ('BUY', 'SELL'):
            pnl = actual_return if sig == 'BUY' else -actual_return
            strategy_pnl.setdefault(sname, []).append(pnl)

# ── Results ──
elapsed = time.time() - start_time
print(f"\n{'='*70}")
print(f"  HFT SCALP EVALUATION — {elapsed:.0f}s ({len(indices)} windows @ 10min)")
print(f"{'='*70}")

# Model direction accuracy
correct = sum(1 for r in records if (r['net_change'] > 0 and r['actual_return'] > 0) or (r['net_change'] < 0 and r['actual_return'] < 0))
total_pred = len(records)
dir_acc = correct / total_pred * 100 if total_pred else 0
print(f"\n📊 KRONOS DIRECTION ACCURACY (HFT scalp): {dir_acc:.1f}% ({correct}/{total_pred})")

# Per-strategy ranking
print(f"\n{'─'*70}")
print(f"  STRATEGY RANKINGS (sorted by total PnL%)")
print(f"{'─'*70}")
print(f"  {'Strategy':<24s} {'Trades':>7s} {'PnL%':>10s} {'Win%':>7s} {'AvgWin':>9s} {'AvgLoss':>9s}")
print(f"  {'-'*66}")

sorted_strats = sorted(strategy_pnl.items(), key=lambda x: np.sum(x[1]), reverse=True)
for sname, pnls in sorted_strats:
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    wr = len(wins)/len(pnls)*100 if pnls else 0
    avg_w = np.mean(wins) if wins else 0
    avg_l = np.mean(losses) if losses else 0
    print(f"  {sname:<24s} {len(pnls):>7d} {np.sum(pnls):>+9.2f}% {wr:>6.1f}% {avg_w:>+8.2f}% {avg_l:>+8.2f}%")

# Best standalone and combo
print(f"\n{'─'*70}")
kronos_only = {k: v for k, v in strategy_pnl.items() if k.startswith('k_')}
price_only = {k: v for k, v in strategy_pnl.items() if k.startswith('ict_') or k.startswith('ta_')}
combos = {k: v for k, v in strategy_pnl.items() if k.startswith('combo_')}

if kronos_only:
    best_k = max(kronos_only.items(), key=lambda x: np.sum(x[1]))
    print(f"  🥇 Best Kronos-only:     {best_k[0]} ({np.sum(best_k[1]):+.2f}%, {len(best_k[1])} trades)")
if price_only:
    best_p = max(price_only.items(), key=lambda x: np.sum(x[1]))
    print(f"  🥇 Best ICT/TA-only:     {best_p[0]} ({np.sum(best_p[1]):+.2f}%, {len(best_p[1])} trades)")
if combos:
    best_c = max(combos.items(), key=lambda x: np.sum(x[1]))
    print(f"  🥇 Best Combo:           {best_c[0]} ({np.sum(best_c[1]):+.2f}%, {len(best_c[1])} trades)")

# Overall best
best_all = max(strategy_pnl.items(), key=lambda x: np.sum(x[1]))
print(f"\n  🏆 OVERALL BEST:          {best_all[0]} ({np.sum(best_all[1]):+.2f}% total)")

# Save
if args.save:
    out = {
        'config': {'windows': total_pred, 'interval_min': step_idx*5, 'params': 'scalp'},
        'model_accuracy': {'direction_pct': round(dir_acc, 1), 'correct': correct, 'total': total_pred},
        'strategies': {k: {'total_pnl_pct': round(float(np.sum(v)), 2), 'trades': len(v),
                           'win_rate': round(len([p for p in v if p>0])/len(v)*100, 1) if v else 0}
                       for k, v in strategy_pnl.items()}
    }
    json.dump(out, open(PROJECT_DIR/'hft_eval_results.json', 'w'), indent=2)
    print(f"  Results saved to hft_eval_results.json")

print(f"{'='*70}")
print(f"  EVALUATION COMPLETE")
print(f"{'='*70}")
