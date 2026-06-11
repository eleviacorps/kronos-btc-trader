"""
kronos_kronos_sim.py — Portfolio sim using ACTUAL Kronos model for entries.
10 days, 0.5 BTC, 1:200 lev, TP/SL trigger fills.
Entry: Kronos range_ext prediction on 5m (like kronos_hft_eval.py k_range_ext).
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
parser.add_argument('--windows', type=int, default=1500, help='Number of 10-min windows')
parser.add_argument('--capital', type=float, default=5000)
parser.add_argument('--size', type=float, default=0.5)
parser.add_argument('--save', action='store_true')
args = parser.parse_args()

CAPITAL = args.capital; POS_SIZE = args.size; LEV = 200
TP_PCT = 0.3; SL_PCT = 0.2

print(f"=== KRONOS MODEL PORTFOLIO SIM ===")
print(f"Capital: ${CAPITAL} | Size: {POS_SIZE} BTC | Lev: 1:{LEV}")

# Fetch data
print("\nFetching 5m BTC data...")
ex = ccxt.binance({'options':{'defaultType':'spot'},'timeout':30000})
ex.load_markets()
all_c = []
since = ex.parse8601((datetime.utcnow()-timedelta(days=14)).isoformat())
while True:
    o = ex.fetch_ohlcv('BTC/USDT','5m',since=since,limit=1000)
    if not o: break
    all_c.extend(o); since=o[-1][0]+1; time.sleep(0.2)
    if len(all_c) >= 5000: break
df = pd.DataFrame(all_c, columns=['ts','o','h','l','c','v'])
df['t'] = pd.to_datetime(df['ts'],unit='ms')
df = df.drop_duplicates('ts').sort_values('ts').reset_index(drop=True)
print(f"Loaded {len(df)} candles ({df['t'].iloc[0]} → {df['t'].iloc[-1]})")

# Load Kronos
print("Loading Kronos on CUDA...")
device = 'cuda:0'
tok = KronosTokenizer.from_pretrained('NeoQuasar/Kronos-Tokenizer-base').to(device)
model = Kronos.from_pretrained('NeoQuasar/Kronos-base').to(device)
predictor = KronosPredictor(model, tok, max_context=512)
print("Model loaded")

# Kronos prediction
def kronos_pred(df5, lb=200, pl=4):
    if len(df5) < lb: return None, 0
    use_lb = min(lb, len(df5))
    x = df5.iloc[-use_lb:][['o','h','l','c','v']].copy()
    x.columns = ['open','high','low','close','volume']; x['amount']=0.0
    ts = df5['t'].iloc[-use_lb:].reset_index(drop=True)
    yt = pd.Series(pd.date_range(start=ts.iloc[-1]+pd.Timedelta(minutes=5), periods=pl, freq='5min'))
    try:
        with torch.no_grad():
            p = predictor.predict(df=x, x_timestamp=ts, y_timestamp=yt, pred_len=pl, T=1.0, top_p=0.9, sample_count=5, verbose=False)
        pc = float(p['close'].iloc[-1]); cp = float(df5['c'].iloc[-1])
        net = ((pc-cp)/cp)*100
        rng = ((p['high'].max()-p['low'].min())/cp)*100
        # Debug: print first few predictions
        if np.random.random() < 0.02:  # ~2% sampling
            print(f"  [PRED] net={net:+.4f}% rng={rng:.4f}% price=${cp:.0f}", flush=True)
        return net, rng
    except Exception as e:
        print(f"  [PRED ERROR] {e}", flush=True)
        return None, 0

# Entry: range_ext (best Kronos-only in HFT eval)
def entry_sig(net, rng):
    if net is None or rng < 0.08: return 'HOLD', 0
    if net > 0.04: return 'BUY', min(net/0.2, 0.85)
    if net < -0.04: return 'SELL', min(abs(net)/0.2, 0.85)
    return 'HOLD', 0

# Position simulation (same logic as portfolio_sim)
def sim_trade(side, entry, idx, mc=24):
    tp = entry*(1+TP_PCT/100) if side=='BUY' else entry*(1-TP_PCT/100)
    sl = entry*(1-SL_PCT/100) if side=='BUY' else entry*(1+SL_PCT/100)
    outcome='EXPIRY'; ex_p=entry
    for o in range(1, mc+1):
        fi=idx+o
        if fi>=len(df): break
        cp=float(df.iloc[fi]['c'])
        if (side=='BUY' and cp>=tp) or (side=='SELL' and cp<=tp): ex_p=tp; outcome='TP'; break
        if (side=='BUY' and cp<=sl) or (side=='SELL' and cp>=sl): ex_p=sl; outcome='SL'; break
    if outcome=='EXPIRY': ex_p=float(df.iloc[min(idx+mc,len(df)-1)]['c'])
    pnl_pct=((ex_p-entry)/entry)*100
    if side=='SELL': pnl_pct=-pnl_pct
    pnl=pnl_pct/100*POS_SIZE*entry
    return round(pnl,2), outcome

# Main loop
LOOKBACK=200
indices=list(range(LOOKBACK+5, min(len(df)-5, args.windows+LOOKBACK+5), 2))
print(f"\nSimulating {len(indices)} windows...")

balance=CAPITAL; trades=[]; start=time.time()
for batch, idx in enumerate(indices):
    if batch%100==0:
        eta=(time.time()-start)/(batch+1)*(len(indices)-batch) if batch>0 else 0
        print(f"  [{batch}/{len(indices)}] {batch/len(indices)*100:.0f}% ETA: {eta:.0f}s")

    ctx=df.iloc[idx-LOOKBACK:idx]
    price=float(ctx['c'].iloc[-1])
    net,rng=kronos_pred(ctx)
    sig,_=entry_sig(net,rng)
    if sig=='HOLD': continue

    margin=POS_SIZE*price/LEV
    if margin>balance: continue
    balance-=margin
    pnl,outcome=sim_trade(sig,price,idx)
    balance+=margin+pnl
    trades.append({'entry':round(price,2),'side':sig,'pnl':pnl,'outcome':outcome,'bal':round(balance,2)})
    if balance<=0: break

elapsed=time.time()-start
print(f"\n{'='*55}")
print(f"  KRONOS MODEL PORTFOLIO — {elapsed:.0f}s")
print(f"{'='*55}")

pnls=[t['pnl'] for t in trades]; wins=[p for p in pnls if p>0]; losses=[p for p in pnls if p<0]
wr=len(wins)/len(pnls)*100 if pnls else 0; total=balance-CAPITAL
tp_count=len([t for t in trades if t['outcome']=='TP'])
sl_count=len([t for t in trades if t['outcome']=='SL'])
exp_count=len([t for t in trades if t['outcome']=='EXPIRY'])

print(f"  Final equity:      ${balance:.2f}")
print(f"  Total PnL:         ${total:+.2f} ({total/CAPITAL*100:+.2f}%)")
print(f"  Total trades:      {len(trades)}")
print(f"  Win rate:          {wr:.1f}%")
print(f"  Avg win:           ${np.mean(wins):+.2f}" if wins else "")
print(f"  Avg loss:          ${np.mean(losses):+.2f}" if losses else "")
print(f"  Profit factor:     {abs(sum(wins)/sum(losses)):.2f}" if losses else "")
print(f"  TP/SL/Expiry:      {tp_count}/{sl_count}/{exp_count}")
print(f"\n  Last 5 trades:")
for t in trades[-5:]:
    icon='🟢' if t['pnl']>0 else '🔴'
    print(f"    {icon} {t['side']:4s} ${t['entry']:,.0f} → ${t['pnl']:+7.2f} ({t['outcome']}) bal=${t['bal']:.2f}")

if args.save:
    json.dump({'trades':trades,'final':balance,'total_pnl':round(total,2)}, open(PROJECT_DIR/'kronos_sim_results.json','w'), indent=2)
    print(f"\nSaved to kronos_sim_results.json")
print(f"{'='*55}")
