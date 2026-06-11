"""
kronos_live_backtest.py — 10-day backtest using EXACT live bot strategy.
Imports real ScalpStrategyEnsemble from kronos_trader.py with VWAP/EMA/RSI consensus.
Same flow as kronos_scalper.py: predict → ensemble → consensus → execute.
"""
import json, sys, os, time, argparse
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import torch, ccxt

sys.path.insert(0, '.')
import kronos_trader as kt
from model import Kronos, KronosTokenizer, KronosPredictor

PROJECT_DIR = Path(__file__).parent
parser = argparse.ArgumentParser()
parser.add_argument('--windows', type=int, default=1500)
parser.add_argument('--capital', type=float, default=5000)
parser.add_argument('--size', type=float, default=0.75)
parser.add_argument('--save', action='store_true')
args = parser.parse_args()

CAPITAL = args.capital; POS_SIZE = args.size; LEV = 200

print(f"=== LIVE BOT BACKTEST ===")
print(f"Using exact ScalpStrategyEnsemble from kronos_trader.py")
print(f"Capital: ${CAPITAL} | Size: {POS_SIZE} BTC | Lev: 1:{LEV}")

print("\nFetching BTC 5m data...")
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

print("Loading Kronos + ensemble...")
device = 'cuda:0'
tok = KronosTokenizer.from_pretrained('NeoQuasar/Kronos-Tokenizer-base').to(device)
model = Kronos.from_pretrained('NeoQuasar/Kronos-base').to(device)
predictor = KronosPredictor(model, tok, max_context=512)

# Use the exact live ScalpStrategyEnsemble
ensemble = kt.ScalpStrategyEnsemble()
print("Ensemble loaded with strategies:", [s.name for s in ensemble.strategies])

# ── Position simulation (same as portfolio_sim, trigger fills) ──
TP_PCT = 0.3; SL_PCT = 0.2

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

# ── Main loop ──
LOOKBACK = 400  # kronos_trader.py default
indices = list(range(LOOKBACK+5, min(len(df)-5, args.windows+LOOKBACK+5), 2))
print(f"\nSimulating {len(indices)} windows (exact live bot pipeline)...")

balance=CAPITAL; trades=[]; start=time.time()
for batch, idx in enumerate(indices):
    if batch%50==0:
        eta=(time.time()-start)/(batch+1)*(len(indices)-batch) if batch>0 else 0
        print(f"  [{batch}/{len(indices)}] {batch/len(indices)*100:.0f}% ETA: {eta:.0f}s  bal=${balance:.0f}")

    ctx = df.iloc[idx-LOOKBACK:idx].copy()
    x_df = ctx[['o','h','l','c','v']].copy()
    x_df.columns = ['open','high','low','close','volume']
    x_df['amount'] = 0.0
    x_ts = ctx['t']
    price = float(ctx['c'].iloc[-1])

    # Kronos prediction
    y_ts = pd.Series(pd.date_range(start=x_ts.iloc[-1]+pd.Timedelta(minutes=5), periods=kt.PRED_LEN, freq='5min'))
    try:
        with torch.no_grad():
            pred_df = predictor.predict(df=x_df, x_timestamp=x_ts, y_timestamp=y_ts,
                                         pred_len=kt.PRED_LEN, T=1.0, top_p=0.9,
                                         sample_count=5, verbose=False)
    except Exception as e:
        continue

    # Build KronosPrediction
    pred = kt.KronosPrediction(
        current_close=price,
        current_time=str(ctx['t'].iloc[-1]),
        pred_df=pred_df,
        direction="BULLISH" if (float(pred_df['close'].iloc[-1])/price-1)*100 > 0.1 else "BEARISH" if (float(pred_df['close'].iloc[-1])/price-1)*100 < -0.1 else "NEUTRAL",
        net_change_pct=((float(pred_df['close'].iloc[-1])-price)/price)*100,
        range_pct=((pred_df['high'].max()-pred_df['low'].min())/price)*100,
        volatility=((pred_df['high'].max()-pred_df['low'].min())/price)*100,
        candle_changes=[],
    )

    # Run ensemble + consensus on properly-named dataframe
    results = ensemble.analyze(pred, x_df)
    consensus = ensemble.consensus(results)

    # Debug: show ALL strategy states for first 20 windows
    all_sigs = [f"{r.name}={r.signal.value}({r.confidence:.3f})" for r in results]
    if batch < 20 or any(r.signal != kt.Signal.HOLD for r in results):
        print(f"  [W{batch:3d}] ${price:,.0f} all={all_sigs} → consensus={consensus.signal.value}({consensus.confidence:.3f})", flush=True)

    if consensus.signal == kt.Signal.HOLD or consensus.confidence < 0.25:
        continue

    side = 'buy' if consensus.signal in (kt.Signal.BUY, kt.Signal.STRONG_BUY) else 'sell'
    margin = POS_SIZE * price / LEV
    if margin > balance: continue

    balance -= margin
    pnl, outcome = sim_trade(side, price, idx)
    balance += margin + pnl
    trades.append({'entry':round(price,2),'side':side,'pnl':pnl,'outcome':outcome,
                   'conf':round(consensus.confidence,3),
                   'bal':round(balance,2)})
    if balance <= 0: break

elapsed = time.time()-start
print(f"\n{'='*55}")
print(f"  LIVE BOT BACKTEST — {elapsed:.0f}s")
print(f"{'='*55}")

pnls=[t['pnl'] for t in trades]; wins=[p for p in pnls if p>0]; losses=[p for p in pnls if p<0]
wr=len(wins)/len(pnls)*100 if pnls else 0; total=balance-CAPITAL
tp_c=len([t for t in trades if t['outcome']=='TP'])
sl_c=len([t for t in trades if t['outcome']=='SL'])
exp_c=len([t for t in trades if t['outcome']=='EXPIRY'])

print(f"  Final equity:      ${balance:.2f}")
print(f"  Total PnL:         ${total:+.2f} ({total/CAPITAL*100:+.2f}%)")
print(f"  Total trades:      {len(trades)}")
print(f"  Win rate:          {wr:.1f}%")
if wins: print(f"  Avg win:           ${np.mean(wins):+.2f}")
if losses: print(f"  Avg loss:          ${np.mean(losses):+.2f}")
print(f"  Profit factor:     {abs(sum(wins)/sum(losses)):.2f}" if losses else "")
print(f"  TP/SL/Expiry:      {tp_c}/{sl_c}/{exp_c}")
print(f"\n  Last 5 trades:")
for t in trades[-5:]:
    icon='🟢' if t['pnl']>0 else '🔴'
    print(f"    {icon} {t['side']:4s} ${t['entry']:,.0f} c={t['conf']:.2f} → ${t['pnl']:+7.2f} ({t['outcome']}) bal=${t['bal']:.2f}")

if args.save:
    json.dump({'trades':trades,'final':balance,'pnl':round(total,2)}, open(PROJECT_DIR/'live_bot_results.json','w'), indent=2)
    print(f"\nSaved to live_bot_results.json")
print(f"{'='*55}")
