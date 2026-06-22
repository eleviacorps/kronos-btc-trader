"""Test Koronos antitrend and different thresholds on gold."""
import sys, os, time
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
import torch, ccxt
from model import Kronos, KronosTokenizer, KronosPredictor

print("Fetching gold data...", flush=True)
ex = ccxt.binance({'options':{'defaultType':'future'},'timeout':15000})
ex.load_markets()
o = ex.fetch_ohlcv('XAU/USDT:USDT','5m',limit=500)
df = pd.DataFrame(o, columns=['ts','o','h','l','c','v'])
df['t'] = pd.to_datetime(df['ts'],unit='ms')
print(f"  {len(df)} candles, price: ${df['c'].iloc[-1]:.2f}", flush=True)

print("Loading Kronos...", flush=True)
device = 'cuda:0'
tok = KronosTokenizer.from_pretrained('NeoQuasar/Kronos-Tokenizer-base').to(device)
model = Kronos.from_pretrained('NeoQuasar/Kronos-base').to(device)
predictor = KronosPredictor(model, tok, max_context=512)
print("  Done", flush=True)

# Test different approaches
def get_kronos_pred(df, idx, lookback=200, pred_len=4):
    """Get Kronos prediction for window starting at idx."""
    ctx = df.iloc[idx-lookback:idx].copy()
    x = ctx[['o','h','l','c','v']].copy()
    x.columns = ['open','high','low','close','volume']; x['amount']=0.0
    ts = ctx['t'].reset_index(drop=True)
    yt = pd.Series(pd.date_range(start=ts.iloc[-1]+pd.Timedelta(minutes=5), periods=pred_len, freq='5min'))
    with torch.no_grad():
        p = predictor.predict(df=x, x_timestamp=ts, y_timestamp=yt,
                               pred_len=pred_len, T=1.0, top_p=0.9,
                               sample_count=5, verbose=False)
    pc = float(p['close'].iloc[-1]); cp = float(ctx['c'].iloc[-1])
    return ((pc-cp)/cp)*100, ((p['high'].max()-p['low'].min())/cp)*100

print("\n=== Testing Kronos direction vs ANTITREND ===", flush=True)

# For each window: compare Kronos pred direction vs actual, and antitrend direction vs actual
Kronos_direct_correct = 0
Kronos_direct_total = 0
antitrend_correct = 0
antitrend_total = 0

LOOKBACK = 200
results = []

for idx in range(LOOKBACK + 5, min(len(df) - 10, 350), 2):
    net, rng = get_kronos_pred(df, idx, LOOKBACK, 4)
    cp = float(df.iloc[idx-1]['c'])
    actual_4 = float(df.iloc[idx+4]['c']) if idx+4 < len(df) else cp
    actual_move = ((actual_4 - cp) / cp) * 100

    # Kronos direct: if net > threshold, BUY. If < negative threshold, SELL
    # Antitrend: if net > threshold, SELL (short the predicted rally). If < negative, BUY

    for threshold in [0.02, 0.04, 0.06, 0.08, 0.10]:
        k_dir = 'UP' if net > threshold else ('DOWN' if net < -threshold else 'NEUT')
        a_dir = 'UP' if actual_move > 0.02 else ('DOWN' if actual_move < -0.02 else 'NEUT')

        # Kronos direct
        if k_dir == a_dir and k_dir != 'NEUT':
            pass  # correct

        # Antitrend (opposite)
        at_dir = 'DOWN' if net > threshold else ('UP' if net < -threshold else 'NEUT')
        if at_dir == a_dir and at_dir != 'NEUT':
            pass  # correct

        # Track for summary
        if k_dir != 'NEUT':
            Kronos_direct_total += 1
            if k_dir == a_dir:
                Kronos_direct_correct += 1

        if at_dir != 'NEUT':
            antitrend_total += 1
            if at_dir == a_dir:
                antitrend_correct += 1

    if idx < 220:  # Show first few
        results.append((idx, cp, net, actual_move, rng))

# Print sample
print(f"{'idx':>4} | {'price':>7} | {'net%':>8} | {'actual%':>8} | {'rng%':>6} | {'k_dir':>5} | {'a_dir':>5} | {'anti':>5}", flush=True)
print("-"*65, flush=True)
for idx, cp, net, act, rng in results:
    kd = 'UP' if net > 0.04 else ('DOWN' if net < -0.04 else 'NEUT')
    ad = 'UP' if act > 0.02 else ('DOWN' if act < -0.02 else 'NEUT')
    anti = 'UP' if net < -0.04 else ('DOWN' if net > 0.04 else 'NEUT')
    anti_match = '✓' if anti == ad else '✗'
    kd_match = '✓' if kd == ad else '✗'
    print(f"{idx:4d} | ${cp:>5.2f} | {net:>+7.3f}% | {act:>+7.3f}% | {rng:>.3f}% | {kd:>4s}{kd_match} | {ad:>4s} | {anti:>4s}{anti_match}", flush=True)

print(f"\n--- Summary ---", flush=True)
# Need to recalculate properly - let me just do it clean
print("\nDetailed threshold analysis:", flush=True)
for threshold in [0.02, 0.04, 0.06, 0.08, 0.10]:
    k_c = 0; k_t = 0; a_c = 0; a_t = 0
    for idx in range(LOOKBACK + 5, min(len(df) - 10, 350), 2):
        net, rng = get_kronos_pred(df, idx, LOOKBACK, 4)
        cp = float(df.iloc[idx-1]['c'])
        actual_4 = float(df.iloc[idx+4]['c']) if idx+4 < len(df) else cp
        actual_move = ((actual_4 - cp) / cp) * 100

        k_dir = 1 if net > threshold else (-1 if net < -threshold else 0)
        a_dir = 1 if actual_move > 0.02 else (-1 if actual_move < -0.02 else 0)

        # Direct
        if k_dir != 0 and k_dir == a_dir: k_c += 1
        if k_dir != 0: k_t += 1

        # Antitrend
        at_dir = -k_dir  # opposite
        if at_dir != 0 and at_dir == a_dir: a_c += 1
        if at_dir != 0: a_t += 1

    k_acc = k_c/k_t*100 if k_t else 0
    a_acc = a_c/a_t*100 if a_t else 0
    print(f"  Threshold {threshold:>5.2f}: Direct={k_c}/{k_t}={k_acc:.0f}%  Antitrend={a_c}/{a_t}={a_acc:.0f}%", flush=True)

# Also test with longer prediction horizon
print("\n=== Testing different prediction lengths ===", flush=True)
for pred_len in [2, 4, 6, 8]:
    k_c = 0; k_t = 0; a_c = 0; a_t = 0
    for idx in range(LOOKBACK + 5, min(len(df) - pred_len - 5, 320), 2):
        net, rng = get_kronos_pred(df, idx, LOOKBACK, pred_len)
        cp = float(df.iloc[idx-1]['c'])
        actual_final = float(df.iloc[idx+pred_len-1]['c']) if idx+pred_len-1 < len(df) else cp
        actual_move = ((actual_final - cp) / cp) * 100

        k_dir = 1 if net > 0.04 else (-1 if net < -0.04 else 0)
        a_dir = 1 if actual_move > 0.02 else (-1 if actual_move < -0.02 else 0)

        if k_dir != 0 and k_dir == a_dir: k_c += 1
        if k_dir != 0: k_t += 1

        at_dir = -k_dir
        if at_dir != 0 and at_dir == a_dir: a_c += 1
        if at_dir != 0: a_t += 1

    k_acc = k_c/k_t*100 if k_t else 0
    a_acc = a_c/a_t*100 if a_t else 0
    print(f"  pred_len={pred_len}: Direct={k_c}/{k_t}={k_acc:.0f}%  Antitrend={a_c}/{a_t}={a_acc:.0f}%", flush=True)
