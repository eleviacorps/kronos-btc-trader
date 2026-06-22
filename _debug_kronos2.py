"""Debug Kronos on REAL XAU/USD spot vs Binance futures."""
import sys, os, time
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
import torch
from model import Kronos, KronosTokenizer, KronosPredictor

print("Testing Kronos on different data sources", flush=True)

# 1. Try Yahoo Finance spot gold
print("\n1. Fetching XAU/USD from Yahoo Finance...", flush=True)
import yfinance as yf
try:
    raw = yf.download('XAUUSD=X', period='2d', interval='5m', progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df_yf = raw.reset_index()
    df_yf.columns = [c.lower() for c in df_yf.columns]
    # Map columns
    if 'date' in df_yf.columns:
        df_yf.rename(columns={'date': 't'}, inplace=True)
    df_yf['t'] = pd.to_datetime(df_yf['t'])
    df_yf.rename(columns={'open': 'o', 'high': 'h', 'low': 'l', 'close': 'c', 'volume': 'v'}, inplace=True)
    print(f"  Yahoo: {len(df_yf)} candles, price: ${df_yf['c'].iloc[-1]:.2f}", flush=True)
except Exception as e:
    print(f"  Yahoo failed: {e}", flush=True)
    df_yf = None

# 2. Try GC=F (Gold Futures)
print("\n2. Fetching GC=F from Yahoo Finance...", flush=True)
try:
    raw2 = yf.download('GC=F', period='2d', interval='5m', progress=False)
    if isinstance(raw2.columns, pd.MultiIndex):
        raw2.columns = raw2.columns.get_level_values(0)
    df_gc = raw2.reset_index()
    df_gc.columns = [c.lower() for c in df_gc.columns]
    df_gc['t'] = pd.to_datetime(df_gc['date' if 'date' in df_gc.columns else 'datetime'])
    df_gc.rename(columns={'open': 'o', 'high': 'h', 'low': 'l', 'close': 'c', 'volume': 'v'}, inplace=True)
    print(f"  GC=F: {len(df_gc)} candles, price: ${df_gc['c'].iloc[-1]:.2f}", flush=True)
except Exception as e:
    print(f"  GC=F failed: {e}", flush=True)
    df_gc = None

# 3. Binance spot XAU/USDT
print("\n3. Fetching XAU/USDT spot from Binance...", flush=True)
import ccxt
ex = ccxt.binance({'options': {'defaultType': 'spot'}, 'timeout': 15000})
ex.load_markets()
try:
    o3 = ex.fetch_ohlcv('XAU/USDT', '5m', limit=200)
    if o3 and len(o3) > 0:
        df_spot = pd.DataFrame(o3, columns=['ts','o','h','l','c','v'])
        df_spot['t'] = pd.to_datetime(df_spot['ts'], unit='ms')
        print(f"  Binance spot XAU/USDT: {len(df_spot)} candles, price: ${df_spot['c'].iloc[-1]:.2f}", flush=True)
    else:
        df_spot = None
        print("  Binance spot XAU/USDT: no data", flush=True)
except Exception as e:
    print(f"  Binance spot failed: {e}", flush=True)
    df_spot = None

# 4. Binance futures XAU/USDT:USDT
print("\n4. Fetching XAU/USDT:USDT futures from Binance...", flush=True)
exf = ccxt.binance({'options': {'defaultType': 'future'}, 'timeout': 15000})
exf.load_markets()
o4 = exf.fetch_ohlcv('XAU/USDT:USDT', '5m', limit=200)
df_fut = pd.DataFrame(o4, columns=['ts','o','h','l','c','v'])
df_fut['t'] = pd.to_datetime(df_fut['ts'], unit='ms')
print(f"  Binance futures XAU/USDT:USDT: {len(df_fut)} candles, price: ${df_fut['c'].iloc[-1]:.2f}", flush=True)

# Now test Kronos on each source
print("\n" + "="*60, flush=True)
print("Loading Kronos...", flush=True)
device = 'cuda:0'
tok = KronosTokenizer.from_pretrained('NeoQuasar/Kronos-Tokenizer-base').to(device)
model = Kronos.from_pretrained('NeoQuasar/Kronos-base').to(device)
predictor = KronosPredictor(model, tok, max_context=512)
print("Loaded\n", flush=True)

def test_kronos(df, name, lookback=200):
    """Test Kronos direction accuracy on a dataframe."""
    if df is None or len(df) < lookback + 10:
        print(f"  {name}: insufficient data", flush=True)
        return

    correct = 0
    total = 0
    predictions = []

    for idx in range(lookback + 1, min(len(df) - 5, lookback + 50), 2):
        ctx = df.iloc[idx-lookback:idx].copy()
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

        actual_cp = float(df.iloc[idx+4]['c']) if idx+4 < len(df) else cp
        actual_move = ((actual_cp - cp) / cp) * 100

        k_dir = 1 if net > 0 else -1
        a_dir = 1 if actual_move > 0 else -1

        if k_dir == a_dir and abs(net) > 0.01:
            correct += 1
        if abs(net) > 0.01:
            total += 1

        predictions.append((net, actual_move))

    acc = correct / total * 100 if total else 0
    print(f"  {name}: {correct}/{total} = {acc:.1f}% accurate", flush=True)
    
    # Show stats
    nets = [p[0] for p in predictions]
    acts = [p[1] for p in predictions]
    avg_pred = sum(nets) / len(nets) if nets else 0
    corr_coef = np.corrcoef(nets, acts)[0,1] if len(nets) > 1 else 0
    print(f"    Avg pred: {avg_pred:.4f}% | Corr: {corr_coef:.3f}", flush=True)

# Test each source
if df_yf is not None:
    test_kronos(df_yf, "XAU/USD (Yahoo spot)")
if df_gc is not None:
    test_kronos(df_gc, "GC=F (Yahoo futures)")
if df_spot is not None:
    test_kronos(df_spot, "XAU/USDT (Binance spot)")
test_kronos(df_fut, "XAU/USDT:USDT (Binance futures)")

# Also test BTC for comparison
print("\n5. Fetching BTC/USDT for comparison...", flush=True)
exs = ccxt.binance({'timeout':15000})
exs.load_markets()
ob = exs.fetch_ohlcv('BTC/USDT', '5m', limit=300)
df_btc = pd.DataFrame(ob, columns=['ts','o','h','l','c','v'])
df_btc['t'] = pd.to_datetime(df_btc['ts'], unit='ms')
print(f"  BTC/USDT: {len(df_btc)} candles, price: ${df_btc['c'].iloc[-1]:.2f}", flush=True)
test_kronos(df_btc, "BTC/USDT (Binance spot)")
