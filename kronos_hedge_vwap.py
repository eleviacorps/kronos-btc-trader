"""
kronos_hedge_vwap.py — VWAP-on-30m rescue hedge backtest.
5m Kronos entries, 30m VWAP rescue checks (instead of Kronos rescue).
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

print("=== VWAP RESCUE HEDGE BACKTEST ===")
print("5m Kronos entry + 30m VWAP rescue (VWAP has 62.5% WR on 5m)")

# 1. Fetch data
print("\nFetching 5m BTC data (10 days)...")
ex = ccxt.binance({'options': {'defaultType': 'spot'}, 'timeout': 30000})
ex.load_markets()
all_5m = []
since = ex.parse8601((datetime.utcnow() - timedelta(days=14)).isoformat())
while True:
    o = ex.fetch_ohlcv('BTC/USDT', '5m', since=since, limit=1000)
    if not o: break
    all_5m.extend(o); since = o[-1][0]+1; time.sleep(0.2)
    if len(all_5m) >= 4000: break
df5 = pd.DataFrame(all_5m, columns=['ts','o','h','l','c','v'])
df5['t'] = pd.to_datetime(df5['ts'], unit='ms')
df5 = df5.drop_duplicates('ts').sort_values('ts').reset_index(drop=True)
print(f"  5m: {len(df5)} candles ({df5['t'].iloc[0]} → {df5['t'].iloc[-1]})")

print("Fetching 30m BTC data (30 days to cover lookback)...")
all_30m = []
since = ex.parse8601((datetime.utcnow() - timedelta(days=60)).isoformat())
while True:
    o = ex.fetch_ohlcv('BTC/USDT', '30m', since=since, limit=500)
    if not o: break
    all_30m.extend(o); since = o[-1][0]+1; time.sleep(0.2)
    if len(all_30m) >= 2000: break
df30 = pd.DataFrame(all_30m, columns=['ts','o','h','l','c','v'])
df30['t'] = pd.to_datetime(df30['ts'], unit='ms')
df30 = df30.drop_duplicates('ts').sort_values('ts').reset_index(drop=True)
print(f"  30m: {len(df30)} candles ({df30['t'].iloc[0]} → {df30['t'].iloc[-1]})")

# 2. Load Kronos
print("\nLoading Kronos on CUDA...")
device = 'cuda:0'
tok = KronosTokenizer.from_pretrained('NeoQuasar/Kronos-Tokenizer-base').to(device)
model = Kronos.from_pretrained('NeoQuasar/Kronos-base').to(device)
predictor = KronosPredictor(model, tok, max_context=512)
print("Model loaded")

# 3. Kronos prediction helper
def kronos_pred(df, lb=200, pl=4, tmin=5):
    if len(df) < lb: return None, 0, 0
    ulb = min(lb, len(df)-1)
    x = df.iloc[-ulb:][['o','h','l','c','v']].copy()
    x.columns = ['open','high','low','close','volume']
    x['amount'] = 0.0
    ts = df['t'].iloc[-ulb:].reset_index(drop=True)
    yt = pd.Series(pd.date_range(start=ts.iloc[-1]+pd.Timedelta(minutes=tmin), periods=pl, freq=f'{tmin}min'))
    try:
        with torch.no_grad():
            p = predictor.predict(df=x, x_timestamp=ts, y_timestamp=yt, pred_len=pl, T=1.0, top_p=0.9, sample_count=5, verbose=False)
        pc = float(p['close'].iloc[-1]); cp = float(df['c'].iloc[-1])
        return ((pc-cp)/cp)*100, ((p['high'].max()-p['low'].min())/cp)*100, pc
    except: return None, 0, 0

# 4. VWAP on 30m context (rescue check)
def vwap_signal_30m(ctx_30m):
    """Returns direction bias from 30m VWAP. Used as rescue confirmation."""
    if len(ctx_30m) < 20: return 'HOLD', 0
    typical = (ctx_30m['h'] + ctx_30m['l'] + ctx_30m['c']) / 3
    vwap = (typical * ctx_30m['v']).sum() / ctx_30m['v'].sum()
    price = float(ctx_30m['c'].iloc[-1])
    dist = (price - vwap) / vwap * 100
    # Strong VWAP bias = confirmation of direction
    if dist > 0.15: return 'BUY', min(dist/0.5, 0.8)
    if dist < -0.15: return 'SELL', min(abs(dist)/0.5, 0.8)
    return 'HOLD', 0

# 5. Entry signal (Kronos range_ext)
def entry_signal(net, rng):
    if net is None or rng < 0.08: return 'HOLD', 0
    if net > 0.04: return 'BUY', min(net/0.2, 0.85)
    if net < -0.04: return 'SELL', min(abs(net)/0.2, 0.85)
    return 'HOLD', 0

# 6. Main loop
print(f"\n{'='*60}")
print(f"BACKTEST: {args.windows} windows")
print(f"{'='*60}")

TP, SL = 0.3, 0.2
indices = list(range(205, min(len(df5)-5, args.windows+205), 2))
trades = []; hedges_activated = 0; total_checks = 0
start = time.time()

for batch, idx in enumerate(indices):
    if batch % 30 == 0:
        eta = (time.time()-start)/(batch+1)*(len(indices)-batch) if batch > 0 else 0
        print(f"  [{batch}/{len(indices)}] {batch/len(indices)*100:.0f}% ETA: {eta:.0f}s")

    ctx = df5.iloc[idx-200:idx]
    price = float(ctx['c'].iloc[-1])
    net5, rng5, _ = kronos_pred(ctx, lb=200, pl=4, tmin=5)
    sig, _ = entry_signal(net5, rng5)
    if sig == 'HOLD': continue

    side = sig; entry = price
    tp_p = entry * (1+TP/100) if side == 'BUY' else entry * (1-TP/100)
    sl_p = entry * (1-SL/100) if side == 'BUY' else entry * (1+SL/100)

    hedge = False; outcome = 'SL'; exit_p = entry
    for off in range(1, 25):
        fi = idx+off
        if fi >= len(df5): break
        cp = float(df5.iloc[fi]['c'])
        if (side == 'BUY' and cp >= tp_p) or (side == 'SELL' and cp <= tp_p):
            outcome = 'TP'; exit_p = tp_p; break
        sl_a = sl_p * (1-SL/100) if hedge else sl_p
        if (side == 'BUY' and cp <= sl_a) or (side == 'SELL' and cp >= sl_a):
            if not hedge:
                total_checks += 1
                ts_cp = df5.iloc[fi]['t']
                m30 = df30[df30['t'] <= ts_cp]
                if len(m30) > 40:
                    ctx30 = m30.iloc[-40:]
                    vwap_sig, vwap_conf = vwap_signal_30m(ctx30)
                    # Hedge if 30m VWAP confirms original direction
                    will_rescue = (side == 'BUY' and vwap_sig == 'BUY' and vwap_conf > 0.3) or \
                                  (side == 'SELL' and vwap_sig == 'SELL' and vwap_conf > 0.3)
                    print(f"  [HEDGE] side={side} entry={entry:.0f} cp={cp:.0f} 30m_vwap_sig={vwap_sig}({vwap_conf:.2f}) rescue={will_rescue}", flush=True)
                    if will_rescue:
                        hedge = True; hedges_activated += 1; continue
            outcome = 'HEDGED_SL' if hedge else 'SL'; exit_p = cp; break

    pnl = ((exit_p-entry)/entry)*100
    if side == 'SELL': pnl = -pnl
    trades.append({'entry': round(entry,2), 'side': side, 'exit': round(exit_p,2),
                   'pnl_pct': round(pnl,2), 'outcome': outcome, 'hedged': hedge})

elapsed = time.time()-start
print(f"\n{'='*60}")
print(f"  VWAP RESCUE HEDGE — {elapsed:.0f}s ({len(trades)} trades)")
print(f"{'='*60}")

tp = sum(t['pnl_pct'] for t in trades)
w = [t for t in trades if t['pnl_pct'] > 0]; l = [t for t in trades if t['pnl_pct'] < 0]
wr = len(w)/len(trades)*100 if trades else 0
print(f"\n📊 OVERALL: {len(trades)} trades | PnL: {tp:+.2f}% | WR: {wr:.1f}%")
if w: print(f"  Avg win: {np.mean([t['pnl_pct'] for t in w]):+.2f}%")
if l: print(f"  Avg loss: {np.mean([t['pnl_pct'] for t in l]):+.2f}%")

hs = [t for t in trades if t['hedged']]
print(f"\n🛡️  HEDGE: {total_checks} checks, {hedges_activated} activations")
print(f"  Hedge trades PnL: {sum(t['pnl_pct'] for t in hs):+.2f}%" if hs else "  No hedges")
print(f"  Normal SL: {len([t for t in trades if t['outcome']=='SL'])}")
print(f"  Hedged SL: {len([t for t in trades if t['outcome']=='HEDGED_SL'])}")
print(f"  TP wins: {len([t for t in trades if t['outcome']=='TP'])}")

woh = sum(t['pnl_pct'] for t in trades if not t['hedged'])
for t in hs:
    if t['pnl_pct'] > 0: woh -= t['pnl_pct']; woh += -SL
print(f"  PnL without hedge: {woh:+.2f}%")
print(f"  PnL with hedge:    {tp:+.2f}%")
print(f"  Hedge improvement: {tp-woh:+.2f}%")

if args.save:
    json.dump({'trades':trades,'summary':{'total':len(trades),'pnl':round(tp,2),'wr':round(wr,1),
               'hedges':hedges_activated,'without':round(woh,2),'improvement':round(tp-woh,2)}},
              open(PROJECT_DIR/'hedge_vwap_results.json','w'), indent=2)
    print(f"\n  Saved to hedge_vwap_results.json")
print(f"{'='*60}")
