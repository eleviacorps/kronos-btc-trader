#!/usr/bin/env python3
"""
kronos_hc_scalper.py — High Conviction scalping.
Only trades when STRONG agreement between VWAP, EMA, RSI.
Uses paper_trades_hc.json (separate from bot and agent).

Entry rules:
  1. VWAP 0.2% threshold AND EMA cross direction agree → trade
  2. RSI extreme (<25 or >75) AND VWAP 0.1% agrees → trade
Position: 0.375 BTC (half of normal), same TP/SL as scalp.
"""
import json, os, subprocess, sys, re
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import ccxt
import pandas as pd

NO_WINDOW = 0x08000000 if os.name == 'nt' else 0
PROJECT_DIR = Path(r"D:\Programming\AiProjects\Kronos(reviewing)\Kronos")
VENV_PYTHON = PROJECT_DIR / ".venv" / "Scripts" / "python.exe"
HC_LEDGER = "paper_trades_hc.json"
COOLDOWN_MIN = 5

os.chdir(PROJECT_DIR)

def get_btc_price():
    ex = ccxt.binance({'options': {'defaultType': 'spot'}, 'timeout': 15000})
    return ex.fetch_ticker("BTC/USDT")["last"]

def load_hc_ledger():
    path = PROJECT_DIR / HC_LEDGER
    if path.exists():
        data = json.load(path.open())
        data.setdefault("balance", 1000)
        data.setdefault("positions", [])
        data.setdefault("trades", [])
        data.setdefault("total_pnl", 0.0)
        data.setdefault("cooldown", {})
        data.setdefault("initial_capital", 1000)
        return data
    return {"balance": 1000, "positions": [], "trades": [], "total_pnl": 0.0,
            "cooldown": {}, "initial_capital": 1000}

def save_hc_ledger(ledger):
    (PROJECT_DIR / HC_LEDGER).write_text(json.dumps(ledger, indent=2))

def check_tp_sl_hc():
    """Inline TP/SL check for HC positions (uses trigger price fill)."""
    ledger = load_hc_ledger()
    price = get_btc_price()
    closed = []
    remaining = []
    for pos in ledger["positions"]:
        hit = False
        reason = None
        if pos["side"] == "buy":
            ret = (price - pos["entry_price"]) / pos["entry_price"]
        else:
            ret = (pos["entry_price"] - price) / pos["entry_price"]

        if ret * 100 >= pos["tp_pct"]:
            hit, reason = True, "TP"
        elif ret * 100 <= -pos["sl_pct"]:
            hit, reason = True, "SL"

        if hit:
            if reason == "SL":
                if pos["side"] == "buy":
                    close_price = pos["entry_price"] * (1 - pos["sl_pct"] / 100)
                else:
                    close_price = pos["entry_price"] * (1 + pos["sl_pct"] / 100)
            elif reason == "TP":
                if pos["side"] == "buy":
                    close_price = pos["entry_price"] * (1 + pos["tp_pct"] / 100)
                else:
                    close_price = pos["entry_price"] * (1 - pos["tp_pct"] / 100)
            else:
                close_price = price

            margin = pos["size"] * pos["entry_price"] / 200
            if pos["side"] == "buy":
                pnl_usdt = (close_price - pos["entry_price"]) * pos["size"]
            else:
                pnl_usdt = (pos["entry_price"] - close_price) * pos["size"]

            ledger["balance"] += margin + pnl_usdt
            trade = {"timestamp": datetime.now().isoformat(), "side": pos["side"],
                     "close_reason": reason, "entry_price": round(pos["entry_price"], 2),
                     "exit_price": round(close_price, 2), "size_btc": pos["size"],
                     "leverage": 200, "pnl_usdt": round(pnl_usdt, 2),
                     "pnl_pct": round(ret * 100, 2),
                     "balance_after": round(ledger["balance"], 2)}
            ledger["trades"].append(trade)
            closed.append(trade)
            print(f"  [HC] {'🟢' if reason=='TP' else '🔴'} CLOSED {pos['side'].upper()} {reason} — ${pnl_usdt:+.2f}")
        else:
            remaining.append(pos)

    ledger["positions"] = remaining
    if closed:
        total_val = ledger["balance"]
        ledger["total_pnl"] = round(total_val - ledger.get("initial_capital", 1000), 2)
        save_hc_ledger(ledger)
    return closed

def check_cooldown_hc(side):
    ledger = load_hc_ledger()
    now = datetime.now()
    if side in ledger.get("cooldown", {}):
        last = datetime.fromisoformat(ledger["cooldown"][side])
        if (now - last).total_seconds() / 60 < COOLDOWN_MIN:
            print(f"  [HC] ⏳ Cooldown active for {side.upper()}")
            return True
    return False

def analyze_strategies(ctx_df):
    """Compute VWAP, EMA, RSI signals from price context."""
    results = {}
    price = float(ctx_df['close'].iloc[-1])

    # VWAP
    typical = (ctx_df['high'] + ctx_df['low'] + ctx_df['close']) / 3
    vwap = (typical * ctx_df['volume']).sum() / ctx_df['volume'].sum()
    vwap_dist = (price - vwap) / vwap * 100

    # EMA 9/21
    c = ctx_df['close'].values
    ema9 = pd.Series(c).ewm(span=9).mean().values[-1]
    ema21 = pd.Series(c).ewm(span=21).mean().values[-1]

    # RSI
    deltas = np.diff(c[-(15):] if len(c) > 15 else c)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    rs = np.mean(gains) / (np.mean(losses) + 1e-10)
    rsi = 100 - 100 / (1 + rs) if np.mean(losses) > 0 else 50

    return {'price': price, 'vwap_dist': vwap_dist, 'ema9': ema9, 'ema21': ema21,
            'rsi': rsi, 'vwap': vwap}

def entry_decision(sigs):
    """Return ('BUY'|'SELL'|'HOLD', confidence) based on combo rules."""
    vd = sigs['vwap_dist']
    rsi = sigs['rsi']
    ema_bull = sigs['ema9'] > sigs['ema21']

    # Rule 1: VWAP 0.2% + EMA agreement
    if vd > 0.2 and ema_bull:
        return 'BUY', min(vd / 0.6, 0.9)
    if vd < -0.2 and not ema_bull:
        return 'SELL', min(abs(vd) / 0.6, 0.9)

    # Rule 2: RSI extreme + VWAP agreement
    if rsi < 25 and vd > 0.1:
        return 'BUY', min((30 - rsi) / 30 * 1.2, 1.0)
    if rsi > 75 and vd < -0.1:
        return 'SELL', min((rsi - 70) / 30 * 1.2, 1.0)

    return 'HOLD', 0.0

def main():
    cycle_start = datetime.now()
    print(f"{'='*55}")
    print(f"  KRONOS HIGH CONVICTION SCALPER — {cycle_start.strftime('%H:%M:%S')}")
    print(f"{'='*55}")

    # 1. Check TP/SL
    print("\n[1] TP/SL check...")
    tpsl = check_tp_sl_hc()
    if not tpsl:
        print("  (no positions closed)")

    # 2. Load ledger
    ledger = load_hc_ledger()
    has_position = len(ledger["positions"]) > 0
    print(f"\n[2] HC Balance: ${ledger['balance']:,.2f} | Positions: {len(ledger['positions'])}")

    if has_position:
        print(f"\n[3] Position open — letting it run to TP/SL.")
        return

    # 3. Fetch market data + compute signals
    print("\n[3] Computing HC signals...")
    ex = ccxt.binance({'options': {'defaultType': 'spot'}, 'timeout': 15000})
    ohlcv = ex.fetch_ohlcv("BTC/USDT", timeframe="5m", limit=50)
    df = pd.DataFrame(ohlcv, columns=["timestamp","open","high","low","close","volume"])
    sigs = analyze_strategies(df)

    print(f"  BTC: ${sigs['price']:,.2f}")
    print(f"  VWAP: ${sigs['vwap']:,.2f} (dist: {sigs['vwap_dist']:+.2f}%)")
    print(f"  EMA9: ${sigs['ema9']:,.0f} | EMA21: ${sigs['ema21']:,.0f} | {'BULL' if sigs['ema9'] > sigs['ema21'] else 'BEAR'}")
    print(f"  RSI: {sigs['rsi']:.1f}")

    # 4. Decision
    decision, conf = entry_decision(sigs)
    print(f"\n[4] Decision: {decision} (conf={conf:.3f})")

    if decision == "HOLD":
        print("  No high-conviction setup detected.")
        return

    side = decision.lower()
    if check_cooldown_hc(side):
        print(f"  Skipped by cooldown.")
        return

    # 5. Execute (0.375 BTC half-size)
    size = 0.375
    price = sigs['price']
    margin = size * price / 200
    if margin > ledger["balance"]:
        print(f"  ❌ Insufficient margin: have ${ledger['balance']:.2f}, need ${margin:.2f}")
        return

    ledger["balance"] -= margin
    pos = {"side": side, "size": size, "entry_price": price,
           "timestamp": datetime.now().isoformat(),
           "tp_pct": 0.3, "sl_pct": 0.2, "scalp": True,
           "source": "high_conviction"}
    ledger["positions"].append(pos)
    ledger["cooldown"][side] = datetime.now().isoformat()
    total_val = ledger["balance"]
    ledger["total_pnl"] = round(total_val - ledger.get("initial_capital", 1000), 2)
    save_hc_ledger(ledger)

    print(f"\n[5] ✅ HC {side.upper()} {size} BTC @ ${price:,.2f}")
    print(f"    Margin: ${margin:.2f} | Balance: ${ledger['balance']:,.2f}")
    print(f"    TP: +0.30% | SL: -0.20%")

    elapsed = (datetime.now() - cycle_start).total_seconds()
    print(f"\n{'='*55}")
    print(f"  CYCLE COMPLETE ({elapsed:.0f}s)")
    print(f"{'='*55}")

if __name__ == "__main__":
    main()
