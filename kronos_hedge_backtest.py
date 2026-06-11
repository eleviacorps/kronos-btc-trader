"""
kronos_hedge_backtest.py — Dual-TF hedge strategy.
5m Kronos for entry, 30m Kronos for SL rescue.
Tests if higher-TF confirmation saves losing trades.
"""
import json, sys, os, time, argparse
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import torch, ccxt

sys.path.insert(0, '.')
from model import Kronos, KronosTokenizer, KronosPredictor

PROJECT_DIR = Path(__file__).parent
parser = argparse.ArgumentParser()
parser.add_argument('--windows', type=int, default=300)
parser.add_argument('--save', action='store_true')
args = parser.parse_args()

print("=== KRONOS HEDGE BACKTEST ===")
print("5m entry + 30m SL rescue — checking if higher-TF model confirms trades")

# 1. Fetch 5m data
print("\nFetching 5m BTC data...")
ex = ccxt.binance({'options': {'defaultType': 'spot'}, 'timeout': 30000})
ex.load_markets()
all_5m = []
since = ex.parse8601((datetime.utcnow() - timedelta(days=60)).isoformat())
while True:
    o = ex.fetch_ohlcv('BTC/USDT', timeframe='5m', since=since, limit=1000)
    if not o: break
    all_5m.extend(o); since = o[-1][0]+1; time.sleep(0.2)
    if len(all_5m) >= 18000: break
df5 = pd.DataFrame(all_5m, columns=['ts','o','h','l','c','v'])
df5['t'] = pd.to_datetime(df5['ts'], unit='ms')
df5 = df5.drop_duplicates('ts').sort_values('ts').reset_index(drop=True)
print(f"  5m: {len(df5)} candles ({df5['t'].iloc[0]} → {df5['t'].iloc[-1]})")

# 2. Fetch 30m data (start earlier to cover 5m range + lookback)
print("Fetching 30m BTC data (90d)...")
all_30m = []
since = ex.parse8601((datetime.utcnow() - timedelta(days=90)).isoformat())
while True:
    o = ex.fetch_ohlcv('BTC/USDT', timeframe='30m', since=since, limit=500)
    if not o: break
    all_30m.extend(o); since = o[-1][0]+1; time.sleep(0.2)
    if len(all_30m) >= 3000: break
df30 = pd.DataFrame(all_30m, columns=['ts','o','h','l','c','v'])
df30['t'] = pd.to_datetime(df30['ts'], unit='ms')
df30 = df30.drop_duplicates('ts').sort_values('ts').reset_index(drop=True)
print(f"  30m: {len(df30)} candles ({df30['t'].iloc[0]} → {df30['t'].iloc[-1]})")

# 3. Load Kronos model
print("\nLoading Kronos on CUDA...")
device = 'cuda:0'
tok = KronosTokenizer.from_pretrained('NeoQuasar/Kronos-Tokenizer-base').to(device)
model = Kronos.from_pretrained('NeoQuasar/Kronos-base').to(device)
predictor = KronosPredictor(model, tok, max_context=512)
print("Model loaded")

# 4. Helper: Kronos prediction on a dataframe (generic for any TF)
def kronos_pred(df, lookback=200, pred_len=4, tf_minutes=5):
    if len(df) < lookback + 1:
        if len(df) < lookback: return None, 0, 0
    use_lookback = min(lookback, len(df) - 1)
    x = df.iloc[-use_lookback:][['o','h','l','c','v']].copy()
    x.columns = ['open','high','low','close','volume']
    x['amount'] = 0.0
    ts = df['t'].iloc[-use_lookback:].reset_index(drop=True)
    y_ts = pd.Series(pd.date_range(start=ts.iloc[-1]+pd.Timedelta(minutes=tf_minutes),
                                     periods=pred_len, freq=f'{tf_minutes}min'))
    try:
        with torch.no_grad():
            p = predictor.predict(df=x, x_timestamp=ts, y_timestamp=y_ts,
                                   pred_len=pred_len, T=1.0, top_p=0.9, sample_count=5, verbose=False)
        pc = float(p['close'].iloc[-1])
        cp = float(df['c'].iloc[-1])
        net = ((pc-cp)/cp)*100
        pr = ((p['high'].max()-p['low'].min())/cp)*100
        return net, pr, pc
    except Exception as e:
        print(f"  ⚠️  kronos_pred error: {e}")
        return None, 0, 0

# 5. Entry signal (Kronos range_ext on 5m)
def entry_signal_5m(pred_net, pred_range):
    """Kronos range_ext scalp entry — same params as live system."""
    if pred_net is None: return 'HOLD', 0
    if pred_range < 0.08: return 'HOLD', 0
    # Range extension: trade breakout direction when Kronos predicts wide range
    if pred_net > 0.04: return 'BUY', min(pred_net/0.2, 0.85)
    if pred_net < -0.04: return 'SELL', min(abs(pred_net)/0.2, 0.85)
    return 'HOLD', 0

# 6. Rescue check on 30m TF
def rescue_check_30m(pred_net_30m, original_side, min_conf=0.10):
    """Check if 30m Kronos confirms original trade direction.
    If yes, the SL might be noise — hedge and wait for recovery."""
    if pred_net_30m is None: return False
    # Original BUY: 30m should predict bullish (price will recover)
    if original_side == 'BUY' and pred_net_30m > min_conf: return True
    # Original SELL: 30m should predict bearish (price will continue down)
    if original_side == 'SELL' and pred_net_30m < -min_conf: return True
    return False

# 7. Main backtest loop
print(f"\n{'='*60}")
print(f"BACKTEST: {args.windows} windows")
print(f"{'='*60}")

TP_PCT = 0.3; SL_PCT = 0.2
HEDGE_SL_TRIGGER = 0.15  # Check hedge at -0.15% (before -0.2% SL)
HEDGE_TIMESTOP_MIN = 120  # Max min to hold after hedge

LOOKBACK_5M = 200; LOOKBACK_30M = 40
indices = list(range(LOOKBACK_5M + 5, min(len(df5) - 5, args.windows + LOOKBACK_5M + 5), 2))

trades = []
saved_by_hedge = 0
sl_hit_count = 0
total_hedge_checks = 0
start_time = time.time()

for batch, idx in enumerate(indices):
    if batch % 30 == 0:
        pct = batch/len(indices)*100
        eta = (time.time()-start_time)/(batch+1)*(len(indices)-batch) if batch > 0 else 0
        print(f"  [{batch}/{len(indices)}] {pct:.0f}% ETA: {eta:.0f}s")

    ctx5 = df5.iloc[idx-LOOKBACK_5M:idx]
    price = float(ctx5['c'].iloc[-1])
    ts = ctx5['t'].iloc[-1]

    # Get 5m Kronos prediction
    net5, rng5, _ = kronos_pred(ctx5, lookback=LOOKBACK_5M, pred_len=4, tf_minutes=5)
    sig, conf = entry_signal_5m(net5, rng5)
    if sig == 'HOLD': continue

    # Simulate trade
    side = sig; entry = price
    tp_price = entry * (1 + TP_PCT/100) if side == 'BUY' else entry * (1 - TP_PCT/100)
    sl_price = entry * (1 - SL_PCT/100) if side == 'BUY' else entry * (1 + SL_PCT/100)
    hedge_activated = False
    hedge_ts = None
    outcome = 'SL'
    exit_price = entry

    # Walk forward candle by candle (max 24 candles = 2 hours)
    for offset in range(1, 25):
        fwd_idx = idx + offset
        if fwd_idx >= len(df5): break
        cp = float(df5.iloc[fwd_idx]['c'])

        # Check TP
        if (side == 'BUY' and cp >= tp_price) or (side == 'SELL' and cp <= tp_price):
            outcome = 'TP'; exit_price = tp_price; break

        # Check SL (or hedged SL)
        sl_active = sl_price
        if hedge_activated:
            # After hedge, widened SL or use trailing
            sl_active = entry * (1 - SL_PCT*2/100) if side == 'BUY' else entry * (1 + SL_PCT*2/100)

        if (side == 'BUY' and cp <= sl_active) or (side == 'SELL' and cp >= sl_active):
            if not hedge_activated:
                # Price approaching SL — check hedge
                # Find matching 30m candle
                ts_cp = df5.iloc[fwd_idx]['t']
                match_30 = df30[df30['t'] <= ts_cp]
                total_hedge_checks += 1
                has_ctx = len(match_30) > LOOKBACK_30M
                print(f"  [HEDGE #{total_hedge_checks}] side={side} entry={entry:.0f} cp={cp:.0f} has_30m_ctx={has_ctx} ctx_len={len(match_30)}", flush=True)
                if has_ctx:
                    ctx30 = match_30.iloc[-LOOKBACK_30M:]
                    net30, rng30, _ = kronos_pred(ctx30, lookback=LOOKBACK_30M, pred_len=2, tf_minutes=30)
                    will_rescue = rescue_check_30m(net30, side)
                    print(f"  [HEDGE #{total_hedge_checks}] side={side} entry={entry:.0f} cp={cp:.0f} 30m_net={net30} 30m_rng={rng30:.3f}% rescue={will_rescue}", flush=True)
                    if will_rescue:
                        # Hedge activated — widen SL, keep position
                        hedge_activated = True
                        hedge_ts = ts_cp
                        continue
            # Either hedge didn't trigger, or post-hedge SL hit
            outcome = 'HEDGED_SL' if hedge_activated else 'SL'
            exit_price = cp
            break

    if hedge_activated: saved_by_hedge += 1
    if outcome == 'SL': sl_hit_count += 1

    pnl = ((exit_price-entry)/entry)*100
    if side == 'SELL': pnl = -pnl
    trades.append({'entry': round(entry,2), 'side': side,
                   'exit': round(exit_price,2), 'pnl_pct': round(pnl,2),
                   'outcome': outcome, 'hedged': hedge_activated,
                   'time': str(ts)})

# 8. Results
elapsed = time.time() - start_time
print(f"\n{'='*60}")
print(f"  HEDGE BACKTEST — {elapsed:.0f}s ({len(trades)} trades)")
print(f"{'='*60}")

total_pnl = sum(t['pnl_pct'] for t in trades)
wins = [t for t in trades if t['pnl_pct'] > 0]
losses = [t for t in trades if t['pnl_pct'] < 0]
wr = len(wins)/len(trades)*100 if trades else 0

print(f"\n📊 OVERALL")
print(f"  Trades:        {len(trades)}")
print(f"  Total PnL:     {total_pnl:+.2f}%")
print(f"  Win rate:      {wr:.1f}%")
print(f"  Avg win:       {np.mean([t['pnl_pct'] for t in wins]):+.2f}%" if wins else "  Avg win:       N/A")
print(f"  Avg loss:      {np.mean([t['pnl_pct'] for t in losses]):+.2f}%" if losses else "  Avg loss:      N/A")

print(f"\n🛡️  HEDGE STATS")
print(f"  Hedge checks:    {total_hedge_checks}")
print(f"  Hedges activated: {saved_by_hedge}")
hedge_saved = [t for t in trades if t['hedged']]
hedge_pnl = sum(t['pnl_pct'] for t in hedge_saved) if hedge_saved else 0
print(f"  Hedge trade PnL: {hedge_pnl:+.2f}%")

normal_sl = [t for t in trades if t['outcome'] == 'SL']
hedged_sl = [t for t in trades if t['outcome'] == 'HEDGED_SL']
print(f"  Normal SL losses:  {len(normal_sl)}")
print(f"  Hedged SL losses:  {len(hedged_sl)}")
print(f"  TP wins:           {len([t for t in trades if t['outcome'] == 'TP'])}")

# Before/after hedge impact
without_hedge_pnl = sum(t['pnl_pct'] for t in trades if not t['hedged'])
for t in hedge_saved:
    # What would this trade have been without hedge?
    if t['pnl_pct'] > 0:  # Saved by hedge
        without_hedge_pnl -= t['pnl_pct']  # subtract the gain (was a loss before hedge)
        without_hedge_pnl += -SL_PCT  # add back the -0.2% loss it would have been
print(f"\n  PnL without hedge: {without_hedge_pnl:+.2f}%")
print(f"  PnL with hedge:    {total_pnl:+.2f}%")
print(f"  Hedge improvement: {total_pnl - without_hedge_pnl:+.2f}%")

if args.save:
    json.dump({'trades': trades, 'summary': {
        'total_trades': len(trades), 'total_pnl': round(total_pnl,2),
        'win_rate': round(wr,1), 'hedges_activated': saved_by_hedge,
        'pnl_without_hedge': round(without_hedge_pnl,2),
        'hedge_improvement': round(total_pnl - without_hedge_pnl,2)}},
        open(PROJECT_DIR/'hedge_results.json','w'), indent=2)
    print(f"\n  Saved to hedge_results.json")
print(f"{'='*60}")
