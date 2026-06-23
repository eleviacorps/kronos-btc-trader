#!/usr/bin/env python3
"""Compare with/without TimesFM on same data."""
import sys, json, warnings, time
from pathlib import Path
from datetime import datetime
import numpy as np
warnings.filterwarnings('ignore')

PROJECT_DIR = Path(r"D:\Programming\AiProjects\Kronos(reviewing)\Kronos")
sys.path.insert(0, str(PROJECT_DIR))
import ccxt, torch, pandas as pd
from model import Kronos, KronosTokenizer, KronosPredictor

# Data
ex = ccxt.binance({'timeout':30000})
ohlcv = ex.fetch_ohlcv('BTC/USDT','15m',limit=960)
close = np.array([c[4] for c in ohlcv], dtype=float)
high = np.array([c[2] for c in ohlcv], dtype=float)
low = np.array([c[3] for c in ohlcv], dtype=float)
times = np.array([c[0] for c in ohlcv])

# Models
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
tok = KronosTokenizer.from_pretrained('NeoQuasar/Kronos-Tokenizer-base').to(device)
model = Kronos.from_pretrained('NeoQuasar/Kronos-base').to(device)
predictor = KronosPredictor(model, tok, max_context=512, device=device)
try:
    import timesfm as _tf
    torch.set_float32_matmul_precision('high')
    tf_model = _tf.TimesFM_2p5_200M_torch.from_pretrained('google/timesfm-2.5-200m-pytorch')
    tf_model.compile(_tf.ForecastConfig(max_context=512,max_horizon=64,normalize_inputs=True,
        use_continuous_quantile_head=True,force_flip_invariance=True,infer_is_positive=True,fix_quantile_crossing=True))
    HAS_TF = True
except:
    HAS_TF = False

def run_backtest(use_tf):
    TRADES = []; ip=False; side=''; ep=0; ei=0
    for i in range(200, len(close)):
        if ip:
            ae = max(0.05, (np.mean(np.maximum(high[i-14:i]-low[i-14:i], np.maximum(abs(high[i-14:i]-close[i-15:i-1]), abs(low[i-14:i]-close[i-15:i-1])))) if i>14 else 0.2)/close[i-1]*100)
            tp=ep*(1+1.5*ae/100) if side=='BUY' else ep*(1-1.5*ae/100)
            sl=ep*(1-0.7*ae/100) if side=='BUY' else ep*(1+0.7*ae/100)
            t1=ep*(1+0.5*ae/100) if side=='BUY' else ep*(1-0.5*ae/100)
            t2=ep*(1+1.0*ae/100) if side=='BUY' else ep*(1-1.0*ae/100)
            r=None; x=None
            if side=='BUY':
                if close[i]>=tp: r,x='TP',tp
                elif close[i]<=sl: r,x='SL',sl
                elif close[i]>=t2: r,x='TP2',t2
                elif close[i]>=t1: r,x='TP1',t1
            else:
                if close[i]<=tp: r,x='TP',tp
                elif close[i]>=sl: r,x='SL',sl
                elif close[i]<=t2: r,x='TP2',t2
                elif close[i]<=t1: r,x='TP1',t1
            if not r and i-ei>=8: r,x='TS',close[i]
            if r:
                pp=(x-ep)/ep*100
                if side=='SELL': pp=-pp
                TRADES.append({'pnl':round(pp/100*1000*200,2)})
                ip=False
            continue
        if i%4!=0: continue
        ctx=ohlcv[max(0,i-200):i]
        if len(ctx)<50: continue
        df=pd.DataFrame(ctx,columns=['ts','o','h','l','c','v'])
        df.columns=['timestamp','open','high','low','close','volume']
        df['amount']=0.0
        cts=pd.to_datetime(df['timestamp'],unit='ms')
        yt=pd.Series(pd.date_range(start=cts.iloc[-1]+pd.Timedelta(minutes=15),periods=4,freq='15min'))
        net=0.0; kd='NEUTRAL'; kc=0.0
        try:
            with torch.no_grad():
                p=predictor.predict(df[['open','high','low','close','volume','amount']],x_timestamp=cts,y_timestamp=yt,pred_len=4,T=1.0,top_p=0.9,sample_count=5,verbose=False)
            pc=float(p['close'].iloc[-1]); net=(pc-close[i-1])/close[i-1]*100
            if net>0.1: kd='BULLISH'
            elif net<-0.1: kd='BEARISH'
            kc=min(abs(net)/0.3,1.0) if abs(net)>0.04 else 0.0
        except: pass
        bd='HOLD'; bc=0.0
        if abs(net)>=0.08:
            bd='SELL' if net>0.10 else 'BUY' if net<-0.10 else 'HOLD'
            bc=min(abs(net)/0.3,1.0) if bd!='HOLD' else 0.0
        td='NEUTRAL'; tc=0.0
        if use_tf and HAS_TF:
            try:
                _p=np.array(df['close'].values,dtype=np.float64)
                _pt,_q=tf_model.forecast(horizon=8,inputs=[_p])
                _fe=float(_pt[0,-1]); _ch=(_fe-close[i-1])/close[i-1]*100
                _uc=(_q[0,-1,9]-_q[0,-1,1])/_fe*100
                if _uc<=1.5:
                    if _ch>0.05: td,tc='BULLISH',min(abs(_ch)/0.3,1.0)
                    elif _ch<-0.05: td,tc='BEARISH',min(abs(_ch)/0.3,1.0)
            except: pass
        sigs=[]
        if bd=='BUY': sigs.append((1,bc))
        elif bd=='SELL': sigs.append((-1,bc))
        if kd=='BULLISH': sigs.append((1,kc))
        elif kd=='BEARISH': sigs.append((-1,kc))
        if td=='BULLISH': sigs.append((1,tc))
        elif td=='BEARISH': sigs.append((-1,tc))
        if not sigs: continue
        tw=sum(s[1] for s in sigs)
        if tw==0: continue
        vote=sum(s[0]*s[1] for s in sigs)/tw; conf=tw/len(sigs)
        if conf<0.3: continue
        if vote>0.15: side='BUY'
        elif vote<-0.15: side='SELL'
        else: continue
        ip=True; ep=close[i-1]; ei=i
    return TRADES

print("Running WITHOUT TimesFM...")
t0=time.time(); r1=run_backtest(False)
w1=len([t for t in r1 if t['pnl']>0]); p1=sum(t['pnl'] for t in r1)
print(f"  {len(r1)} trades, WR: {w1/len(r1)*100:.1f}%, PnL: ${p1:+.0f} ({time.time()-t0:.0f}s)")

print("Running WITH TimesFM...")
t0=time.time(); r2=run_backtest(True)
w2=len([t for t in r2 if t['pnl']>0]); p2=sum(t['pnl'] for t in r2)
print(f"  {len(r2)} trades, WR: {w2/len(r2)*100:.1f}%, PnL: ${p2:+.0f} ({time.time()-t0:.0f}s)")

delta_p = p2 - p1
delta_w = (w2/len(r2)*100 - w1/len(r1)*100) if r2 and r1 else 0
print(f"\nTimesFM impact: PnL ${delta_p:+.0f}, WR {delta_w:+.1f}%")
