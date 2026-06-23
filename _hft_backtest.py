#!/usr/bin/env python3
"""HFT backtest — 5m BTC, every 2nd candle, Kronos + TimesFM + Fusion."""
import sys, warnings, time, numpy as np
warnings.filterwarnings('ignore')
from pathlib import Path; from datetime import datetime
import sys
PROJECT_DIR = Path(r"D:\Programming\AiProjects\Kronos(reviewing)\Kronos")
sys.path.insert(0, str(PROJECT_DIR))
import ccxt, torch, pandas as pd
from model import Kronos, KronosTokenizer, KronosPredictor

print("="*70)
print("HFT BACKTEST — 5m BTC, Kronos+TimesFM+Fusion")
print("="*70)

# ── DATA ──
ex = ccxt.binance({'timeout':30000})
ohlcv = ex.fetch_ohlcv('BTC/USDT','5m',limit=2016)  # 7 days
close = np.array([c[4] for c in ohlcv], dtype=float)
high = np.array([c[2] for c in ohlcv], dtype=float)
low = np.array([c[3] for c in ohlcv], dtype=float)
times = np.array([c[0] for c in ohlcv])
print(f"Data: {len(ohlcv)} candles, ${close[0]:.0f} -> ${close[-1]:.0f}")

# ── MODELS ──
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
t0=time.time()
tok = KronosTokenizer.from_pretrained('NeoQuasar/Kronos-Tokenizer-base').to(device)
model = Kronos.from_pretrained('NeoQuasar/Kronos-base').to(device)
predictor = KronosPredictor(model, tok, max_context=512, device=device)
print(f"Kronos: {time.time()-t0:.1f}s")
try:
    import timesfm as _tf
    torch.set_float32_matmul_precision('high')
    t0=time.time()
    tf_model = _tf.TimesFM_2p5_200M_torch.from_pretrained('google/timesfm-2.5-200m-pytorch')
    tf_model.compile(_tf.ForecastConfig(max_context=512,max_horizon=64,normalize_inputs=True,
        use_continuous_quantile_head=True,force_flip_invariance=True,infer_is_positive=True,fix_quantile_crossing=True))
    HAS_TF=True; print(f"TimesFM: {time.time()-t0:.1f}s")
except: HAS_TF=False; print("No TimesFM")

# ── INDICATORS ──
def ATR(h,l,c,p=14):
    tr=np.maximum(h[1:]-l[1:],np.maximum(abs(h[1:]-c[:-1]),abs(l[1:]-c[:-1])))
    a=np.full(len(c),0.2)
    for i in range(p,len(c)): a[i]=np.mean(tr[i-p:i])
    return a/c*100
atr=ATR(high,low,close)

# ── BACKTEST ──
print("\nRunning HFT backtest...")
TRADES=[]; ip=False; sd=''; ep=0; ei=0; t0=time.time()

# Configs — HFT style
CONF_TH = 0.3
ENTRY_STEP = 2       # every 2nd candle = 10min
TIMESTOP = 12        # 12 candles = 60min
TP_MULT = 1.5
SL_MULT = 0.7
SCALE = True         # scale-out at 0.5x and 1.0x ATR

for i in range(200, len(close)):
    if ip:
        ae = atr[ei]
        if sd=='BUY':
            tp=ep*(1+TP_MULT*ae/100); sl=ep*(1-SL_MULT*ae/100)
            t1=ep*(1+0.5*ae/100) if SCALE else tp
            t2=ep*(1+1.0*ae/100) if SCALE else None
        else:
            tp=ep*(1-TP_MULT*ae/100); sl=ep*(1+SL_MULT*ae/100)
            t1=ep*(1-0.5*ae/100) if SCALE else tp
            t2=ep*(1-1.0*ae/100) if SCALE else None
        r=None; x=None
        if sd=='BUY':
            if close[i]>=tp: r,x='TP',tp
            elif close[i]<=sl: r,x='SL',sl
            elif SCALE and close[i]>=t2: r,x='TP2',t2
            elif SCALE and close[i]>=t1: r,x='TP1',t1
        else:
            if close[i]<=tp: r,x='TP',tp
            elif close[i]>=sl: r,x='SL',sl
            elif SCALE and close[i]<=t2: r,x='TP2',t2
            elif SCALE and close[i]<=t1: r,x='TP1',t1
        if not r and i-ei>=TIMESTOP: r,x='TS',close[i]
        if r:
            pp=(x-ep)/ep*100
            if sd=='SELL': pp=-pp
            TRADES.append({'s':sd,'r':r,'p':round(pp/100*1000*200,2),
                'e':round(ep,2),'x':round(x,2),'t':str(datetime.fromtimestamp(times[i]/1000))[:16]})
            ip=False
        continue
    if i%ENTRY_STEP!=0: continue

    ctx=ohlcv[max(0,i-200):i]
    if len(ctx)<50: continue
    df=pd.DataFrame(ctx,columns=['ts','o','h','l','c','v'])
    df.columns=['timestamp','open','high','low','close','volume']; df['amount']=0.0
    cts=pd.to_datetime(df['timestamp'],unit='ms')
    yt=pd.Series(pd.date_range(start=cts.iloc[-1]+pd.Timedelta(minutes=5),periods=4,freq='5min'))

    net=0.0; kd='N'; kc=0.0
    try:
        with torch.no_grad():
            p=predictor.predict(df[['open','high','low','close','volume','amount']],x_timestamp=cts,y_timestamp=yt,pred_len=4,T=1.0,top_p=0.9,sample_count=5,verbose=False)
        pc=float(p['close'].iloc[-1]); net=(pc-close[i-1])/close[i-1]*100
        if net>0.1: kd='B'
        elif net<-0.1: kd='S'
        kc=min(abs(net)/0.3,1.0) if abs(net)>0.04 else 0.0
    except: pass

    bd='H'; bc=0.0
    if abs(net)>=0.08:
        bd='S' if net>0.10 else 'B' if net<-0.10 else 'H'
        bc=min(abs(net)/0.3,1.0) if bd!='H' else 0.0

    td='N'; tc=0.0
    if HAS_TF:
        try:
            _p=np.array(df['close'].values,dtype=np.float64)
            _pt,_q=tf_model.forecast(horizon=8,inputs=[_p])
            _fe=float(_pt[0,-1]); _ch=(_fe-close[i-1])/close[i-1]*100
            _uc=(_q[0,-1,9]-_q[0,-1,1])/_fe*100
            if _uc<=1.5:
                if _ch>0.05: td,tc='B',min(abs(_ch)/0.3,1.0)
                elif _ch<-0.05: td,tc='S',min(abs(_ch)/0.3,1.0)
        except: pass

    sigs=[]
    if bd=='B': sigs.append((1,bc))
    elif bd=='S': sigs.append((-1,bc))
    if kd=='B': sigs.append((1,kc))
    elif kd=='S': sigs.append((-1,kc))
    if td=='B': sigs.append((1,tc))
    elif td=='S': sigs.append((-1,tc))
    if not sigs: continue
    tw=sum(s[1] for s in sigs)
    if tw==0: continue
    vote=sum(s[0]*s[1] for s in sigs)/tw; conf=tw/len(sigs)
    if conf<CONF_TH: continue
    if vote>0.15: sd='BUY'
    elif vote<-0.15: sd='SELL'
    else: continue
    ip=True; ep=close[i-1]; ei=i

print(f"Time: {time.time()-t0:.0f}s")

# ── RESULTS ──
n=len(TRADES)
if n==0: print("NO TRADES"); sys.exit(0)
w=[t for t in TRADES if t['p']>0]; l=[t for t in TRADES if t['p']<=0]
wr=len(w)/n*100; tp=sum(t['p'] for t in TRADES)
aw=np.mean([t['p'] for t in w]) if w else 0
al=abs(np.mean([t['p'] for t in l])) if l else 0
rr=aw/al if al else 0
buys=[t for t in TRADES if t['s']=='BUY']; sells=[t for t in TRADES if t['s']=='SELL']

print("\n"+ "="*70)
print("RESULTS — HFT BACKTEST (5m BTC, 7 days)")
print("="*70)
print(f"  Trades:  {n}")
print(f"  WR:      {wr:.1f}%")
print(f"  PnL:     ${tp:+.0f}")
print(f"  Avg win: +${aw:.2f}")
print(f"  Avg loss: -${al:.2f}")
print(f"  R:R:     {rr:.2f}:1")
if buys: print(f"  BUY:     {len(buys)} ({len([t for t in buys if t['p']>0])/len(buys)*100:.0f}%)")
if sells: print(f"  SELL:    {len(sells)} ({len([t for t in sells if t['p']>0])/len(sells)*100:.0f}%)")
reasons={}
for t in TRADES: reasons[t['r']]=reasons.get(t['r'],0)+1
print(f"  Exits:   {dict(sorted(reasons.items()))}")

# By confidence threshold
for ct in [0.3, 0.4, 0.5]:
    ft=[t for t in TRADES if t.get('c',0)>=ct]  # we don't store conf per trade
    pass

print(f"\n  Best 5:")
for t in sorted(TRADES,key=lambda x:abs(x['p']),reverse=True)[:5]:
    print(f"    {t['t']} {t['s']:5s} {t['r']:5s} ${t['p']:>+8.2f}")
print(f"  Worst 5:")
for t in sorted(TRADES,key=lambda x:x['p'])[:5]:
    print(f"    {t['t']} {t['s']:5s} {t['r']:5s} ${t['p']:>+8.2f}")
print("="*70)
