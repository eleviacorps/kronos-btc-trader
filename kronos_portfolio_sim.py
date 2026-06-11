"""
kronos_portfolio_sim.py — $1K paper trading simulation.
Simulates actual position management: 0.75 BTC, 1:200 lev, TP/SL trigger fills.
Bot: VWAP signals. Agent: VWAP + EMA combo. 30 months of data.
"""
import json, sys, os, time, argparse
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import ccxt

PROJECT_DIR = Path(__file__).parent
parser = argparse.ArgumentParser()
parser.add_argument('--save', action='store_true')
parser.add_argument('--capital', type=float, default=1000.0, help='Starting capital (default: 1000)')
parser.add_argument('--size', type=float, default=0.75, help='BTC position size (default: 0.75)')
args = parser.parse_args()

LEV = 200; POS_SIZE = args.size; CAPITAL = args.capital
TP_PCT = 0.3; SL_PCT = 0.2

print(f"=== PORTFOLIO SIMULATION: ${CAPITAL:.0f} paper trading ===")
print(f"Bot: VWAP signals | Agent: VWAP+EMA combo | 1:200 lev, {POS_SIZE} BTC")

print("\nFetching 30 months BTC 5m data...")
exchange = ccxt.binance({'options':{'defaultType':'spot'},'timeout':30000})
exchange.load_markets()
all_c = []
since = exchange.parse8601((datetime.utcnow()-timedelta(days=900)).isoformat())
while True:
    o = exchange.fetch_ohlcv('BTC/USDT','5m',since=since,limit=1000)
    if not o: break
    all_c.extend(o); since=o[-1][0]+1; time.sleep(0.15)
    if len(all_c) >= 260000: break
df = pd.DataFrame(all_c, columns=['ts','o','h','l','c','v'])
df['t'] = pd.to_datetime(df['ts'],unit='ms')
df['date'] = df['t'].dt.date
df = df.drop_duplicates('ts').sort_values('ts').reset_index(drop=True)
print(f"Loaded {len(df):,} candles ({df['t'].iloc[0]} → {df['t'].iloc[-1]})")
print(f"Trading days: {df['date'].nunique()}")

# ── Signal functions ──

def vwap_signal(ctx):
    if len(ctx) < 20: return 'HOLD', 0.0
    typical = (ctx['h'] + ctx['l'] + ctx['c']) / 3
    vwap = (typical * ctx['v']).sum() / ctx['v'].sum()
    price = float(ctx['c'].iloc[-1])
    dist = (price - vwap) / vwap * 100
    if dist > 0.1: return 'BUY', min(dist/0.5, 0.8)
    if dist < -0.1: return 'SELL', min(abs(dist)/0.5, 0.8)
    return 'HOLD', 0.0

def ema_signal(ctx, fast=9, slow=21):
    if len(ctx) < slow+1: return 'HOLD', 0.0
    c = ctx['c'].values
    ef = pd.Series(c).ewm(span=fast).mean().values[-1]
    es = pd.Series(c).ewm(span=slow).mean().values[-1]
    if ef > es: return 'BUY', min((ef-es)/es*100*5, 0.8)
    return 'SELL', 0.2

# ── Position simulation ──

def simulate_trade(side, entry, df_start, max_candles=24):
    """Simulate a position with TP/SL. Returns (pnl_usd, exit_price, outcome)."""
    tp = entry * (1 + TP_PCT/100) if side == 'BUY' else entry * (1 - TP_PCT/100)
    sl = entry * (1 - SL_PCT/100) if side == 'BUY' else entry * (1 + SL_PCT/100)
    exit_price = entry
    outcome = 'EXPIRY'
    for off in range(1, max_candles+1):
        fi = df_start + off
        if fi >= len(df): break
        cp = float(df.iloc[fi]['c'])
        if (side == 'BUY' and cp >= tp) or (side == 'SELL' and cp <= tp):
            exit_price = tp; outcome = 'TP'; break
        if (side == 'BUY' and cp <= sl) or (side == 'SELL' and cp >= sl):
            exit_price = sl; outcome = 'SL'; break
    # For expiry: close at last candle's close (no forced SL)
    if outcome == 'EXPIRY':
        last_idx = min(df_start + max_candles, len(df) - 1)
        exit_price = float(df.iloc[last_idx]['c'])
    pnl_pct = ((exit_price - entry)/entry)*100
    if side == 'SELL': pnl_pct = -pnl_pct
    pnl_usd = pnl_pct/100 * POS_SIZE * entry  # position value * pct
    if outcome == 'TP': pnl_usd = TP_PCT/100 * POS_SIZE * entry
    if outcome == 'SL': pnl_usd = -(SL_PCT/100) * POS_SIZE * entry
    return pnl_usd, exit_price, outcome

# ── Simulate both strategies ──

LOOKBACK = 100
indices = list(range(LOOKBACK+1, len(df)-5, 2))

def run_strategy(name, use_ema_filter=False):
    balance = CAPITAL
    equity_curve = [(df['t'].iloc[LOOKBACK], balance)]
    trades = []
    for batch, idx in enumerate(indices):
        ctx = df.iloc[idx-LOOKBACK:idx]
        price = float(ctx['c'].iloc[-1])
        sig, conf = vwap_signal(ctx)
        if sig == 'HOLD': continue
        if use_ema_filter:
            e_sig, e_conf = ema_signal(ctx)
            if e_sig != sig: continue
            if conf < 0.3: continue
        margin_needed = POS_SIZE * price / LEV
        if margin_needed > balance: continue
        balance -= margin_needed
        pnl, exit_p, outcome = simulate_trade(sig, price, idx)
        balance += margin_needed + pnl
        trades.append({'entry':round(price,2),'side':sig,'pnl':round(pnl,2),
                       'outcome':outcome,'balance_after':round(balance,2),
                       'time':str(ctx['t'].iloc[-1])[:16]})
        if batch % 10000 == 0:
            equity_curve.append((ctx['t'].iloc[-1], balance))
    equity_curve.append((df['t'].iloc[-1], balance))
    return trades, equity_curve, balance

print(f"\nSimulating {len(indices):,} windows over 30 months...")

start = time.time()
bot_trades, bot_eq, bot_final = run_strategy('Bot', use_ema_filter=False)
elapsed = time.time()-start
print(f"Bot done: {elapsed:.0f}s — {len(bot_trades)} trades, final ${bot_final:.2f}")

start = time.time()
agent_trades, agent_eq, agent_final = run_strategy('Agent', use_ema_filter=True)
elapsed = time.time()-start
print(f"Agent done: {elapsed:.0f}s — {len(agent_trades)} trades, final ${agent_final:.2f}")

# ── Results ──

def report(name, trades, eq_curve, final):
    pnls = [t['pnl'] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    wr = len(wins)/len(pnls)*100 if pnls else 0
    total_pnl = final - CAPITAL
    peak = CAPITAL; max_dd = 0
    for _, bal in eq_curve:
        if bal > peak: peak = bal
        dd = (peak - bal)/peak*100
        if dd > max_dd: max_dd = dd

    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    print(f"  Starting capital:  ${CAPITAL:.2f}")
    print(f"  Final equity:      ${final:.2f}")
    print(f"  Total PnL:         ${total_pnl:+.2f} ({total_pnl/CAPITAL*100:+.2f}%)")
    print(f"  Total trades:      {len(trades):,}")
    print(f"  Win rate:          {wr:.1f}%")
    print(f"  Avg win:           ${np.mean(wins):+.2f}" if wins else "  Avg win:           N/A")
    print(f"  Avg loss:          ${np.mean(losses):+.2f}" if losses else "  Avg loss:          N/A")
    print(f"  Profit factor:     {abs(sum(wins)/sum(losses)):.2f}" if losses else "  Profit factor:     ∞")
    print(f"  Max drawdown:      {max_dd:.1f}%")
    
    trades_df = pd.DataFrame(trades)
    trades_df['month'] = pd.to_datetime([t['time'] for t in trades]).to_period('M')
    monthly = trades_df.groupby('month')['pnl'].sum()
    print(f"  Best month:        ${monthly.max():+.2f}")
    print(f"  Worst month:       ${monthly.min():+.2f}")
    print(f"  Profitable months: {(monthly>0).sum()}/{len(monthly)}")
    
    print(f"\n  Last 5 trades:")
    for t in trades[-5:]:
        icon = '🟢' if t['pnl'] > 0 else '🔴'
        print(f"    {icon} {t['time']} {t['side']:4s} ${t['entry']:,.0f} → ${t['pnl']:+7.2f} ({t['outcome']}) bal=${t['balance_after']:.2f}")

report("BOT (VWAP only)", bot_trades, bot_eq, bot_final)
report("AGENT (VWAP+EMA combo)", agent_trades, agent_eq, agent_final)

print(f"\n{'='*55}")
print(f"  COMPARISON")
print(f"{'='*55}")
bot_wr = len([t for t in bot_trades if t['pnl']>0])/len(bot_trades)*100 if bot_trades else 0
agent_wr = len([t for t in agent_trades if t['pnl']>0])/len(agent_trades)*100 if agent_trades else 0
print(f"  {'':>15s} {'Bot':>12s} {'Agent':>12s}")
print(f"  {'Final equity:':>15s} ${bot_final:>9.2f} ${agent_final:>9.2f}")
print(f"  {'Return:':>15s} {bot_final/CAPITAL*100-100:>+11.2f}% {agent_final/CAPITAL*100-100:>+11.2f}%")
print(f"  {'Trades:':>15s} {len(bot_trades):>12,} {len(agent_trades):>12,}")
print(f"  {'Win rate:':>15s} {bot_wr:>11.1f}% {agent_wr:>11.1f}%")

if args.save:
    out = {'bot':{'trades':bot_trades, 'final':bot_final},
           'agent':{'trades':agent_trades, 'final':agent_final}}
    json.dump(out, open(PROJECT_DIR/'portfolio_sim_results.json','w'), indent=2)
    print(f"\nResults saved to portfolio_sim_results.json")
print(f"{'='*55}")
