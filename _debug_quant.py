"""Debug HMM regimes and HTF values."""
import numpy as np, pandas as pd, ccxt, time
from datetime import datetime, timedelta
from collections import Counter
import sys
sys.path.insert(0, 'D:/Programming/AiProjects/Kronos(reviewing)/Kronos')

ex = ccxt.binance({'options':{'defaultType':'spot'},'timeout':30000})
ex.load_markets()

print("Fetching 5m data...")
all_c = []
since = ex.parse8601((datetime.utcnow() - timedelta(days=18)).isoformat())
while len(all_c) < 5000:
    o = ex.fetch_ohlcv('BTC/USDT', '5m', since=since, limit=1000)
    if not o: break
    all_c.extend(o)
    since = o[-1][0] + 1
    time.sleep(0.1)
df = pd.DataFrame(all_c, columns=['ts','o','h','l','c','v'])
rets = np.diff(np.log(df['c'].values))
print(f"  {len(rets)} returns, std={np.std(rets)*100:.3f}%")

print("\nHMM Regime Profiles:")
from quant_models.hmm_regime import HMMRegimeDetector
hmm = HMMRegimeDetector(n_regimes=4, n_iter=100)
hmm.fit(rets)
for r in range(4):
    p = hmm.profile(r)
    print(f"  Regime {r}: label={p.get('label','?')}, mean={p.get('mean',0)*100:.4f}%, std={p.get('std',0)*100:.4f}%")

print("\nLast 500 window predictions:")
last_states = []
for i in range(500, len(rets), 2):
    r = hmm.predict(rets[i-50:i])
    last_states.append(r)
cnt = Counter(last_states)
for r, c in sorted(cnt.items()):
    print(f"  Regime {r} ({hmm.regime_label(r)}): {c} ({c/5:.1f}%)")

print("\nFetching 1h data...")
since_1h = ex.parse8601((datetime.utcnow() - timedelta(days=60)).isoformat())
all_1h = []
while len(all_1h) < 500:
    o = ex.fetch_ohlcv('BTC/USDT', '1h', since=since_1h, limit=500)
    if not o: break
    all_1h.extend(o)
    since_1h = o[-1][0] + 1
    time.sleep(0.1)
df_1h = pd.DataFrame(all_1h, columns=['ts','o','h','l','c','v'])
df_1h['ema50'] = df_1h['c'].ewm(span=50, adjust=False).mean()

price = float(df_1h['c'].iloc[-1])
ema = float(df_1h['ema50'].iloc[-1])
ratio = (price - ema) / ema * 100
print(f"Latest price={price:.2f}, EMA50={ema:.2f}, diff={ratio:+.2f}%")

if ratio > 0.5:
    print("  → HTF bias = BULLISH")
elif ratio < -0.5:
    print("  → HTF bias = BEARISH")
else:
    print("  → HTF bias = NEUTRAL")

# Also show EMA50 from last 24h
last_24 = df_1h.tail(24)
print(f"\nLast 24h prices: {last_24['c'].min():.0f} - {last_24['c'].max():.0f}")
print(f"Last 24h EMA50 range: {last_24['ema50'].min():.0f} - {last_24['ema50'].max():.0f}")
