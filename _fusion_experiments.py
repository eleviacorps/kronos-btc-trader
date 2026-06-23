#!/usr/bin/env python3
"""fusion_experiments.py — v3"""
import sys, warnings, numpy as np
warnings.filterwarnings('ignore')
from pathlib import Path
PROJECT_DIR = Path(r"D:\Programming\AiProjects\Kronos(reviewing)\Kronos")
sys.path.insert(0, str(PROJECT_DIR))
import ccxt

print("[1] Fetching 7 days BTC 5m...")
ex = ccxt.binance({'timeout': 30000})
ohlcv = ex.fetch_ohlcv('BTC/USDT', '5m', limit=2016)
close = np.array([c[4] for c in ohlcv], dtype=float)
high = np.array([c[2] for c in ohlcv], dtype=float)
low = np.array([c[3] for c in ohlcv], dtype=float)
print(f"  {len(close)} candles, ${close[0]:.0f} → ${close[-1]:.0f}")

def atr_pct(h, l, c, p=14):
    tr = np.maximum(h[1:]-l[1:], np.maximum(abs(h[1:]-c[:-1]), abs(l[1:]-c[:-1])))
    a = np.full(len(c), 0.2)
    for i in range(p, len(c)): a[i] = np.mean(tr[i-p:i])
    return a / c * 100

def rsi(c, p=14):
    if len(c) <= p: return np.full(len(c), 50.0)
    d = np.diff(c); g = np.where(d>0,d,0); l = np.where(d<0,-d,0)
    r = np.full(len(c), 50.0)
    for i in range(p, len(c)):
        ag = np.mean(g[i-p:i]); al = np.mean(l[i-p:i])
        r[i] = 100 - 100/(1+ag/al) if al else 100
    return r

atr = atr_pct(high, low, close)
rsi_v = rsi(close)

def run(name, conf=0.3, rlo=25, rhi=75, adx=False, asym=False, scale=False):
    trades = []; inp=False; side=''; ep=0; ei=0
    for i in range(200, len(close)):
        if inp:
            ae = atr[ei]
            if asym and side=='BUY': tp_p, sl_p = 2.0*ae, 0.7*ae
            elif asym and side=='SELL': tp_p, sl_p = 1.2*ae, 0.7*ae
            else: tp_p, sl_p = 1.5*ae, 0.7*ae
            tp = ep*(1+tp_p/100) if side=='BUY' else ep*(1-tp_p/100)
            sp = ep*(1-sl_p/100) if side=='BUY' else ep*(1+sl_p/100)
            reason=None; xp=None
            if scale:
                t1 = ep*(1+0.5*ae/100) if side=='BUY' else ep*(1-0.5*ae/100)
                t2 = ep*(1+1.0*ae/100) if side=='BUY' else ep*(1-1.0*ae/100)
                if side=='BUY':
                    if close[i]>=tp: reason,xp='TP',tp
                    elif close[i]<=sp: reason,xp='SL',sp
                    elif close[i]>=t2: reason,xp='TP2',t2
                    elif close[i]>=t1: reason,xp='TP1',t1
                else:
                    if close[i]<=tp: reason,xp='TP',tp
                    elif close[i]>=sp: reason,xp='SL',sp
                    elif close[i]<=t2: reason,xp='TP2',t2
                    elif close[i]<=t1: reason,xp='TP1',t1
            else:
                if side=='BUY':
                    if close[i]>=tp: reason,xp='TP',tp
                    elif close[i]<=sp: reason,xp='SL',sp
                else:
                    if close[i]<=tp: reason,xp='TP',tp
                    elif close[i]>=sp: reason,xp='SL',sp
            if not reason and i-ei>=12: reason,xp='TS',close[i]
            if reason:
                pp = (xp-ep)/ep*100
                if side=='SELL': pp=-pp
                pnl = pp/100*1000*200
                trades.append((side, pnl))
                inp=False
        if not inp:
            if rsi_v[i] < rlo:
                c = min(1.0, (rlo-rsi_v[i])/rlo*0.8+0.6)
                if c>=conf: inp,side,ep,ei=True,'BUY',close[i],i
            elif rsi_v[i] > rhi:
                c = min(1.0, (rsi_v[i]-rhi)/(100-rhi)*0.8+0.6)
                if c>=conf: inp,side,ep,ei=True,'SELL',close[i],i
    if not trades: return name,0,0,0,0,0,0,0
    n=len(trades); w=[t for t in trades if t[1]>0]; l=[t for t in trades if t[1]<=0]
    wr=len(w)/n*100; aw=np.mean([t[1] for t in w]) if w else 0; al=abs(np.mean([t[1] for t in l])) if l else 0
    tp=sum(t[1] for t in trades); rr=aw/al if al else 0
    b=[t for t in trades if t[0]=='BUY']; s=[t for t in trades if t[0]=='SELL']
    bwr=len([t for t in b if t[1]>0])/len(b)*100 if b else 0
    swr=len([t for t in s if t[1]>0])/len(s)*100 if s else 0
    return name,n,round(wr,1),round(tp),round(aw,2),round(al,2),round(rr,2),round(bwr,1),round(swr,1)

exps = [
    ("1. BASELINE (RSI 25/75, conf 0.3)",{}),
    ("2. Asymmetric TP/SL (BUY 2x, SELL 1.2x)",dict(asym=True)),
    ("3. Conf threshold 0.5",dict(conf=0.5)),
    ("4. Scale-out exits (0.5x/1.0x ATR)",dict(scale=True)),
    ("5. RSI 30/70 (wider range)",dict(rlo=30,rhi=70)),
    ("6. RSI 20/80 (narrower range)",dict(rlo=20,rhi=80)),
    ("7. Conf 0.4 + Asym",dict(conf=0.4,asym=True)),
    ("8. Conf 0.5 + Asym",dict(conf=0.5,asym=True)),
    ("9. Asym + Scale-out",dict(asym=True,scale=True)),
    ("10. Asym + Scale + Conf 0.4",dict(asym=True,scale=True,conf=0.4)),
    ("11. RSI 30/70 + Asym + Scale",dict(rlo=30,rhi=70,asym=True,scale=True)),
    ("12. RSI 30/70 + Asym + Conf 0.4",dict(rlo=30,rhi=70,asym=True,conf=0.4)),
    ("13. RSI 20/80 + Asym",dict(rlo=20,rhi=80,asym=True)),
    ("14. Scale-out only (baseline RSI)",dict(scale=True,conf=0.4)),
]

print("\nResults (7 days BTC 5m, $1k cap, 200x):")
print(f"{'#':>2} {'Strategy':<40} {'Trades':>6} {'WR%':>5} {'PnL':>6} {'R:R':>5} {'BUY%':>5} {'SELL%':>5}")
print("-"*80)
for i,(name,kw) in enumerate(exps,1):
    r=run(name,**kw)
    if r[1]==0: print(f" {i:2d} {name:<40} {'NONE':>6}")
    else: print(f" {i:2d} {name:<40} {r[1]:5d}  {r[2]:5.1f}% ${r[3]:>+4d} {r[6]:.2f}:1 {r[7]:4.0f}% {r[8]:4.0f}%")
