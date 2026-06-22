"""Debug Kronos predictions on gold data."""
import sys, os, time
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
import torch, ccxt
from model import Kronos, KronosTokenizer, KronosPredictor

print("Fetching XAU/USDT:USDT...", flush=True)
ex = ccxt.binance({'options': {'defaultType': 'future'}, 'timeout': 15000})
ex.load_markets()
o = ex.fetch_ohlcv('XAU/USDT:USDT', '5m', limit=300)
df = pd.DataFrame(o, columns=['ts', 'o', 'h', 'l', 'c', 'v'])
df['t'] = pd.to_datetime(df['ts'], unit='ms')

print(f"Loaded {len(df)} candles, price range: ${df['c'].min():.2f} - ${df['c'].max():.2f}", flush=True)

print("Loading Kronos...", flush=True)
device = 'cuda:0'
tok = KronosTokenizer.from_pretrained('NeoQuasar/Kronos-Tokenizer-base').to(device)
model = Kronos.from_pretrained('NeoQuasar/Kronos-base').to(device)
predictor = KronosPredictor(model, tok, max_context=512)
print("Loaded", flush=True)

# Run predictions and compare with actual
print("\nidx | cur_pr | pred_cl | net%  | rng%  | k_dir | actual% | k_correct", flush=True)
print("-" * 75, flush=True)

correct = 0
total = 0
strong_correct = 0
strong_total = 0

for offset in range(min(100, (len(df)-210)//2)):
    idx = 205 + offset * 2
    if idx + 5 >= len(df):
        break

    ctx = df.iloc[idx-200:idx].copy()
    x = ctx[['o', 'h', 'l', 'c', 'v']].copy()
    x.columns = ['open', 'high', 'low', 'close', 'volume']
    x['amount'] = 0.0
    ts = ctx['t'].reset_index(drop=True)
    yt = pd.Series(pd.date_range(start=ts.iloc[-1]+pd.Timedelta(minutes=5), periods=4, freq='5min'))

    with torch.no_grad():
        p = predictor.predict(df=x, x_timestamp=ts, y_timestamp=yt,
                               pred_len=4, T=1.0, top_p=0.9,
                               sample_count=5, verbose=False)

    pc = float(p['close'].iloc[-1])
    cp = float(ctx['c'].iloc[-1])
    net = ((pc - cp) / cp) * 100
    rng = ((p['high'].max() - p['low'].min()) / cp) * 100

    # Actual move in next 4 candles
    actual_cp = float(df.iloc[idx+4]['c']) if idx+4 < len(df) else cp
    actual_move = ((actual_cp - cp) / cp) * 100

    k_dir = 'UP' if net > 0.04 else ('DOWN' if net < -0.04 else 'NEUT')
    a_dir = 'UP' if actual_move > 0.04 else ('DOWN' if actual_move < -0.04 else 'NEUT')

    is_correct = (k_dir == a_dir and k_dir != 'NEUT')
    if is_correct:
        correct += 1
    if k_dir != 'NEUT':
        total += 1

    # Strong signals only
    if abs(net) > 0.08:
        strong_total += 1
        if k_dir == a_dir:
            strong_correct += 1

    if offset < 20 or (is_correct and abs(net) > 0.08):
        print(f"{idx:4d} | ${cp:>6.2f} | ${pc:>6.2f} | {net:+.3f}% | {rng:.3f}% | {k_dir:>4s} | {actual_move:+.3f}% | {'✓' if is_correct else '✗'}", flush=True)

print(f"\n--- Summary ({total} non-neutral predictions) ---", flush=True)
print(f"Direction accuracy: {correct}/{total} = {correct/total*100:.1f}%" if total else "No predictions", flush=True)
print(f"Strong (>0.08%) accuracy: {strong_correct}/{strong_total} = {strong_correct/strong_total*100:.1f}%" if strong_total else "No strong preds", flush=True)

# Show Kronos output structure
print(f"\n--- Kronos output structure ---", flush=True)
print(f"p type: {type(p)}", flush=True)
print(f"p columns: {list(p.columns) if hasattr(p, 'columns') else 'N/A'}", flush=True)
if hasattr(p, 'columns'):
    print(f"p shape: {p.shape}", flush=True)
    print(f"p head:\n{p.head()}", flush=True)
