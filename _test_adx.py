"""Test ADX implementations against each other and TradingView-like values."""
import sys, os
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
import ccxt

print("Fetching XAU/USDT:USDT data...", flush=True)
ex = ccxt.binance({'options': {'defaultType': 'future'}, 'timeout': 30000})
ex.load_markets()
o = ex.fetch_ohlcv('XAU/USDT:USDT', '5m', limit=200)
df = pd.DataFrame(o, columns=['ts','o','h','l','c','v'])
df['t'] = pd.to_datetime(df['ts'], unit='ms')

def wilder_rma(series, period):
    return series.ewm(alpha=1/period, adjust=False).mean()

def calc_adx_wilder(ctx, period=14):
    h=ctx['h'].values; l=ctx['l'].values; c=ctx['c'].values; n=len(h)
    tr=np.zeros(n); pdm=np.zeros(n); ndm=np.zeros(n)
    for i in range(1,n):
        tr[i]=max(h[i]-l[i],abs(h[i]-c[i-1]),abs(l[i]-c[i-1]))
        up=h[i]-h[i-1]; dn=l[i-1]-l[i]
        if up>dn and up>0: pdm[i]=up
        if dn>up and dn>0: ndm[i]=dn
    tr_s=pd.Series(tr).ewm(alpha=1/period,adjust=False).mean()
    pdm_s=pd.Series(pdm).ewm(alpha=1/period,adjust=False).mean()
    ndm_s=pd.Series(ndm).ewm(alpha=1/period,adjust=False).mean()
    pdi=100*pdm_s/tr_s.replace(0,np.nan)
    ndi=100*ndm_s/tr_s.replace(0,np.nan)
    dx=100*abs(pdi-ndi)/(pdi+ndi).replace(0,np.nan)
    adx=dx.ewm(alpha=1/period,adjust=False).mean()
    return float(adx.iloc[-1]), float(pdi.iloc[-1]), float(ndi.iloc[-1])

def calc_adx_sma(ctx, period=14):
    h=ctx['h'].values; l=ctx['l'].values; c=ctx['c'].values; n=len(h)
    tr=np.zeros(n); pdm=np.zeros(n); ndm=np.zeros(n)
    for i in range(1,n):
        tr[i]=max(h[i]-l[i],abs(h[i]-c[i-1]),abs(l[i]-c[i-1]))
        up=h[i]-h[i-1]; dn=l[i-1]-l[i]
        if up>dn and up>0: pdm[i]=up
        if dn>up and dn>0: ndm[i]=dn
    tr_s=pd.Series(tr).rolling(period).mean()
    pdm_s=pd.Series(pdm).rolling(period).mean()
    ndm_s=pd.Series(ndm).rolling(period).mean()
    pdi=100*pdm_s/tr_s.replace(0,np.nan)
    ndi=100*ndm_s/tr_s.replace(0,np.nan)
    dx=100*abs(pdi-ndi)/(pdi+ndi).replace(0,np.nan)
    adx=dx.rolling(period).mean()
    return float(adx.iloc[-1]), float(pdi.iloc[-1]), float(ndi.iloc[-1])

# V2 ADX (the one that got +17.85%)
def calc_adx_v2(ctx, period=14):
    h,l,c=ctx['h'].values,ctx['l'].values,ctx['c'].values; n=len(h)
    tr=np.zeros(n); pdm=np.zeros(n); ndm=np.zeros(n)
    for i in range(1,n):
        tr[i]=max(h[i]-l[i],abs(h[i]-c[i-1]),abs(l[i]-c[i-1]))
        up=h[i]-h[i-1]; dn=l[i-1]-l[i]
        if up>dn and up>0: pdm[i]=up
        if dn>up and dn>0: ndm[i]=dn
    ta=pd.Series(tr).rolling(period).mean().values
    pdi=100*pd.Series(pdm).rolling(period).mean()/np.maximum(ta,1e-10)
    ndi=100*pd.Series(ndm).rolling(period).mean()/np.maximum(ta,1e-10)
    dx=100*abs(pdi-ndi)/np.maximum(pdi+ndi,1e-10)
    return float(dx.rolling(period).mean().iloc[-1]), float(pdi.iloc[-1]), float(ndi.iloc[-1])

print("=== ADX Comparison (14-period, last 200 5m XAU/USDT candles ===")
a1, p1, n1 = calc_adx_wilder(df, 14)
print(f"Wilder's RMA: ADX={a1:.3f}  +DI={p1:.3f}  -DI={n1:.3f}  Trend={'UP' if p1>n1 else 'DOWN'}  ADX>25={a1>25}")

a2, p2, n2 = calc_adx_sma(df, 14)
print(f"SMA:          ADX={a2:.3f}  +DI={p2:.3f}  -DI={n2:.3f}  Trend={'UP' if p2>n2 else 'DOWN'}  ADX>25={a2>25}")

a3, p3, n3 = calc_adx_v2(df, 14)
print(f"V2 (hybrid):  ADX={a3:.3f}  +DI={p3:.3f}  -DI={n3:.3f}  Trend={'UP' if p3>n3 else 'DOWN'}  ADX>25={a3>25}")

# Also check VWAP signal
print("\n=== VWAP Trend signal on last candle ===")
typical = (df['h'] + df['l'] + df['c']) / 3
vwap_100 = (typical * df['v']).rolling(100).sum() / df['v'].rolling(100).sum()
price = float(df['c'].iloc[-1])
v100 = float(vwap_100.iloc[-1])
dist = (price - v100) / v100 * 100
print(f"Price={price:.2f} VWAP(100)={v100:.2f} Dist={dist:.3f}%")
print(f"Signal: {'BUY' if dist > 0.1 else 'SELL' if dist < -0.1 else 'HOLD'}")

# VWAP on last 20
vwap_20 = (typical * df['v']).rolling(20).sum() / df['v'].rolling(20).sum()
v20 = float(vwap_20.iloc[-1])
dist20 = (price - v20) / v20 * 100
print(f"VWAP(20)={v20:.2f} Dist={dist20:.3f}%")
