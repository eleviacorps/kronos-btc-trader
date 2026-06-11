#!/usr/bin/env python3
"""
kronos_hc_collector.py — Data collector for Hermes-judged High Conviction trading.
Collects TP/SL status, account status, and VWAP/EMA/RSI signals for Hermes to review.
Designed for agent-mode cron.
"""
import json, os, subprocess, sys, time
from pathlib import Path
from datetime import datetime
import numpy as np
import ccxt
import pandas as pd

NO_WINDOW = 0x08000000 if os.name == 'nt' else 0
PROJECT_DIR = Path(r"D:\Programming\AiProjects\Kronos(reviewing)\Kronos")
VENV_PYTHON = PROJECT_DIR / ".venv" / "Scripts" / "python.exe"
HC_LEDGER = "paper_trades_hc_agent.json"
os.chdir(PROJECT_DIR)

def run(cmd, timeout=60):
    result = subprocess.run(
        [str(VENV_PYTHON)] + cmd.split(),
        capture_output=True, text=True, timeout=timeout,
        cwd=PROJECT_DIR, creationflags=NO_WINDOW)
    return result.stdout + result.stderr

print(f"{'='*60}")
print(f"  KRONOS HC DATA COLLECTOR (Agent Edition)")
print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'='*60}")

# 1. TP/SL check — close positions at trigger price
print(f"\n{'─'*40}")
print("  [1] TP/SL CHECK (HC Agent)")
print(f"{'─'*40}")
tpsl_out = run(f"kronos_exec.py --scalp --check-tp-sl --ledger {HC_LEDGER}")
print(tpsl_out.strip() if tpsl_out.strip() else "  (no positions closed)")

# 2. Status — balance, positions, recent trades
print(f"\n{'─'*40}")
print("  [2] ACCOUNT STATUS (HC Agent)")
print(f"{'─'*40}")
print(run(f"kronos_exec.py --scalp --status --ledger {HC_LEDGER}"))

# 3. Compute HC signals from live data
print(f"\n{'─'*40}")
print("  [3] HIGH CONVICTION SIGNALS")
print(f"{'─'*40}")
try:
    ex = ccxt.binance({'options': {'defaultType': 'spot'}, 'timeout': 15000})
    ohlcv = ex.fetch_ohlcv("BTC/USDT", timeframe="5m", limit=50)
    df = pd.DataFrame(ohlcv, columns=["timestamp","open","high","low","close","volume"])
    price = float(df['close'].iloc[-1])

    typical = (df['high'] + df['low'] + df['close']) / 3
    vwap = (typical * df['volume']).sum() / df['volume'].sum()
    vwap_dist = (price - vwap) / vwap * 100

    c = df['close'].values
    ema9 = pd.Series(c).ewm(span=9).mean().values[-1]
    ema21 = pd.Series(c).ewm(span=21).mean().values[-1]
    ema_bull = ema9 > ema21

    deltas = np.diff(c[-(15):] if len(c) > 15 else c)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains)
    avg_loss = np.mean(losses) if np.mean(losses) > 0 else 0.001
    rs = avg_gain / avg_loss
    rsi = 100 - 100 / (1 + rs)

    print(f"  BTC/USDT:    ${price:,.2f}")
    print(f"  VWAP:        ${vwap:,.2f}  (distance: {vwap_dist:+.3f}%)")
    print(f"  EMA9:        ${ema9:,.2f}")
    print(f"  EMA21:       ${ema21:,.2f}")
    print(f"  EMA Trend:   {'📈 BULLISH (EMA9 > EMA21)' if ema_bull else '📉 BEARISH (EMA9 < EMA21)'}")
    print(f"  RSI (14):    {rsi:.1f}  ({'🟢 Oversold' if rsi < 30 else '🔴 Overbought' if rsi > 70 else '⚪ Neutral'})")

    # Entry logic
    hc_signal = "HOLD"
    hc_conf = 0.0
    hc_reason = ""
    if vwap_dist > 0.2 and ema_bull:
        hc_signal = "BUY"
        hc_conf = min(vwap_dist / 0.6, 0.9)
        hc_reason = "VWAP +0.2% + EMA bullish"
    elif vwap_dist < -0.2 and not ema_bull:
        hc_signal = "SELL"
        hc_conf = min(abs(vwap_dist) / 0.6, 0.9)
        hc_reason = "VWAP -0.2% + EMA bearish"
    elif rsi < 25 and vwap_dist > 0.1:
        hc_signal = "BUY"
        hc_conf = min((30 - rsi) / 30 * 1.2, 1.0)
        hc_reason = f"RSI {rsi:.0f} oversold + VWAP positive"
    elif rsi > 75 and vwap_dist < -0.1:
        hc_signal = "SELL"
        hc_conf = min((rsi - 70) / 30 * 1.2, 1.0)
        hc_reason = f"RSI {rsi:.0f} overbought + VWAP negative"

    print(f"  ─────────────────────────────────────")
    print(f"  HC Decision:   {hc_signal} (conf={hc_conf:.3f})")
    print(f"  HC Reason:     {hc_reason}")
    print(f"  HC Thresholds: VWAP ±0.2% + EMA agree  |  RSI <25/>75 + VWAP ±0.1%")
    print(f"  Position:      0.375 BTC (half scalp)  |  TP: +0.30%  SL: -0.20%")
    print(f"  ─────────────────────────────────────")

except Exception as e:
    print(f"  ⚠️  Signal computation error: {e}")

print(f"\n{'='*60}")
print(f"  END DATA — Hermes, review the above and execute if HC signal is clear.")
print(f"{'='*60}")
