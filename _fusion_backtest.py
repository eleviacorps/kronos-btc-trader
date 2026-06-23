#!/usr/bin/env python3
"""fusion_backtest_15m.py — Full backtest: Kronos + TimesFM + Fusion on 15m BTC."""
import sys, os, json, warnings, time, math
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

PROJECT_DIR = Path(r"D:\Programming\AiProjects\Kronos(reviewing)\Kronos")
sys.path.insert(0, str(PROJECT_DIR))
import ccxt, torch
from model import Kronos, KronosTokenizer, KronosPredictor

print("="*70)
print("KRONOS + TIMESFM + FUSION — 10 DAY BACKTEST (15m)")
print("="*70)

# ── 1. DATA ──
print("\n[1] Fetching 10 days BTC 15m...")
ex = ccxt.binance({'timeout':30000})
ohlcv = ex.fetch_ohlcv('BTC/USDT','15m',limit=960)
close = np.array([c[4] for c in ohlcv], dtype=float)
high = np.array([c[2] for c in ohlcv], dtype=float)
low = np.array([c[3] for c in ohlcv], dtype=float)
times = np.array([c[0] for c in ohlcv])
print(f"  {len(ohlcv)} candles, ${close[0]:.0f} -> ${close[-1]:.0f}")

# ── 2. MODELS ──
print("\n[2] Loading models...")
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
t0 = time.time()
tok = KronosTokenizer.from_pretrained('NeoQuasar/Kronos-Tokenizer-base').to(device)
model = Kronos.from_pretrained('NeoQuasar/Kronos-base').to(device)
predictor = KronosPredictor(model, tok, max_context=512, device=device)
print(f"  Kronos on {device} ({time.time()-t0:.1f}s)")
try:
    import timesfm as _tf
    torch.set_float32_matmul_precision('high')
    t0 = time.time()
    tf_model = _tf.TimesFM_2p5_200M_torch.from_pretrained('google/timesfm-2.5-200m-pytorch')
    tf_model.compile(_tf.ForecastConfig(max_context=512,max_horizon=64,
        normalize_inputs=True,use_continuous_quantile_head=True,
        force_flip_invariance=True,infer_is_positive=True,fix_quantile_crossing=True))
    HAS_TF = True
    print(f"  TimesFM 2.5 200M ({time.time()-t0:.1f}s)")
except Exception as e:
    HAS_TF = False
    print(f"  TimesFM unavailable: {e}")

# ── 3. INDICATORS ──
def calc_atr(h, l, c, p=14):
    tr = np.maximum(h[1:]-l[1:], np.maximum(abs(h[1:]-c[:-1]), abs(l[1:]-c[:-1])))
    a = np.full(len(c), 0.2)
    for i in range(p, len(c)): a[i] = np.mean(tr[i-p:i])
    return a / c * 100

def calc_rsi(c, p=14):
    r = np.full(len(c), 50.0)
    if len(c) <= p: return r
    d = np.diff(c); g = np.where(d>0,d,0); l_ = np.where(d<0,-d,0)
    for i in range(p, len(c)):
        ag = np.mean(g[i-p:i]); al = np.mean(l_[i-p:i])
        r[i] = 100 - 100/(1+ag/al) if al else 100
    return r

atr = calc_atr(high, low, close)
rsi = calc_rsi(close)

# ── 4. BACKTEST ──
MIN_C = 200
TRADES = []
in_pos = False
side = ''
ep = 0
ei = 0

print(f"\n[3] Running backtest ({len(close)} candles)...")
t0 = time.time()

for i in range(MIN_C, len(close)):
    # Exit check
    if in_pos:
        bars = i - ei
        ae = atr[ei]
        tp = ep * (1+1.5*ae/100) if side=='BUY' else ep*(1-1.5*ae/100)
        sl = ep * (1-0.7*ae/100) if side=='BUY' else ep*(1+0.7*ae/100)
        tp1 = ep*(1+0.5*ae/100) if side=='BUY' else ep*(1-0.5*ae/100)
        tp2 = ep*(1+1.0*ae/100) if side=='BUY' else ep*(1-1.0*ae/100)
        reason=None; xp=None
        if side=='BUY':
            if close[i]>=tp: reason,xp='TP',tp
            elif close[i]<=sl: reason,xp='SL',sl
            elif close[i]>=tp2: reason,xp='TP2',tp2
            elif close[i]>=tp1: reason,xp='TP1',tp1
        else:
            if close[i]<=tp: reason,xp='TP',tp
            elif close[i]>=sl: reason,xp='SL',sl
            elif close[i]<=tp2: reason,xp='TP2',tp2
            elif close[i]<=tp1: reason,xp='TP1',tp1
        if not reason and bars>=8: reason,xp='TS',close[i]
        if reason:
            pp = (xp-ep)/ep*100
            if side=='SELL': pp=-pp
            pnl = pp/100*1000*200
            TRADES.append({'side':side,'reason':reason,'pnl':round(pnl,2),
                'entry':round(ep,2),'exit':round(xp,2),
                'time':str(datetime.fromtimestamp(times[i]/1000))[:16]})
            in_pos=False
        continue

    # Entry — run signals every 4th candle
    if i % 4 != 0:
        continue

    ctx = ohlcv[max(0,i-200):i]
    if len(ctx) < 50: continue

    df = pd.DataFrame(ctx, columns=['ts','o','h','l','c','v'])
    df.columns = ['timestamp','open','high','low','close','volume']
    df['amount'] = 0.0
    cts = pd.to_datetime(df['timestamp'], unit='ms')
    yt = pd.Series(pd.date_range(start=cts.iloc[-1]+pd.Timedelta(minutes=15), periods=4, freq='15min'))

    # Kronos
    net = 0.0
    kronos_dir = 'NEUTRAL'
    kronos_conf = 0.0
    try:
        with torch.no_grad():
            p = predictor.predict(df[['open','high','low','close','volume','amount']],
                x_timestamp=cts, y_timestamp=yt, pred_len=4, T=1.0, top_p=0.9, sample_count=5, verbose=False)
        pc = float(p['close'].iloc[-1])
        net = (pc - close[i-1]) / close[i-1] * 100
        if net > 0.1: kronos_dir='BULLISH'
        elif net < -0.1: kronos_dir='BEARISH'
        kronos_conf = min(abs(net)/0.3, 1.0) if abs(net) > 0.04 else 0.0
    except:
        pass

    # Antitrend
    if abs(net) >= 0.08:
        base_dir = 'SELL' if net > 0.10 else 'BUY' if net < -0.10 else 'HOLD'
        base_conf = min(abs(net)/0.3, 1.0) if base_dir != 'HOLD' else 0.0
    else:
        base_dir, base_conf = 'HOLD', 0.0

    # TimesFM
    tf_dir, tf_conf = 'NEUTRAL', 0.0
    if HAS_TF:
        try:
            _p = np.array(df['close'].values, dtype=np.float64)
            _pt, _q = tf_model.forecast(horizon=8, inputs=[_p])
            _fe = float(_pt[0,-1])
            _ch = (_fe - close[i-1]) / close[i-1] * 100
            _uc = (_q[0,-1,9] - _q[0,-1,1]) / _fe * 100
            if _uc <= 1.5:
                if _ch > 0.05: tf_dir, tf_conf = 'BULLISH', min(abs(_ch)/0.3, 1.0)
                elif _ch < -0.05: tf_dir, tf_conf = 'BEARISH', min(abs(_ch)/0.3, 1.0)
        except:
            pass

    # Fusion BMA
    signals = []
    if base_dir=='BUY': signals.append((1, base_conf))
    elif base_dir=='SELL': signals.append((-1, base_conf))
    if kronos_dir=='BULLISH': signals.append((1, kronos_conf))
    elif kronos_dir=='BEARISH': signals.append((-1, kronos_conf))
    if tf_dir=='BULLISH': signals.append((1, tf_conf))
    elif tf_dir=='BEARISH': signals.append((-1, tf_conf))
    if not signals: continue

    tw = sum(s[1] for s in signals)
    if tw==0: continue
    vote = sum(s[0]*s[1] for s in signals)/tw
    conf = tw/len(signals)
    if conf < 0.3: continue
    if vote > 0.15: side='BUY'
    elif vote < -0.15: side='SELL'
    else: continue

    in_pos=True; ep=close[i-1]; ei=i

elapsed = time.time()-t0
print(f"  Backtest: {elapsed:.0f}s")

# ── 5. RESULTS ──
import pandas as pd
print("\n"+ "="*70)
print("RESULTS — 10 DAY BACKTEST (15m BTC)")
print("="*70)
total = len(TRADES)
if total==0: print("NO TRADES"); sys.exit(0)
wins=[t for t in TRADES if t['pnl']>0]
losses=[t for t in TRADES if t['pnl']<=0]
wr=len(wins)/total*100; tp=sum(t['pnl'] for t in TRADES)
aw=np.mean([t['pnl'] for t in wins]) if wins else 0
al=abs(np.mean([t['pnl'] for t in losses])) if losses else 0
rr=aw/al if al else 0
buys=[t for t in TRADES if t['side']=='BUY']
sells=[t for t in TRADES if t['side']=='SELL']
print(f"  Trades: {total} | WR: {wr:.1f}% | PnL: ${tp:+.0f} | R:R {rr:.2f}:1")
if buys: print(f"  BUY: {len(buys)} ({len([t for t in buys if t['pnl']>0])/len(buys)*100:.0f}%)")
if sells: print(f"  SELL: {len(sells)} ({len([t for t in sells if t['pnl']>0])/len(sells)*100:.0f}%)")
reasons={}
for t in TRADES: reasons[t['reason']]=reasons.get(t['reason'],0)+1
print(f"  Exits: {reasons}")
print(f"\n  Top 10:")
for t in sorted(TRADES,key=lambda x:abs(x['pnl']),reverse=True)[:10]:
    print(f"    {t['time']} {t['side']:5s} {t['reason']:5s} ${t['pnl']:>+7.2f}")
print("="*70)
