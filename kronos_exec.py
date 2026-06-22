"""
Kronos Trade Executor — leveraged paper trading with TP/SL.
Supports long (BUY) and short (SELL) positions with 1:200 leverage.

Two modes:
  SWING — default, wider TP/SL, larger positions (~31 trades/day)
  SCALP — HFT mode, tight TP/SL, small positions, time-stop, cooldown (~200+ trades/day)

Usage:
  python kronos_exec.py --status                          # Check positions & balance
  python kronos_exec.py --scalp --status                   # Check with scalp metrics
  python kronos_exec.py --paper buy  --size 0.01           # Swing long
  python kronos_exec.py --scalp --paper buy                # Scalp long (auto size)
  python kronos_exec.py --paper close                       # Close all positions
  python kronos_exec.py --check-tp-sl                       # Check TP/SL only
  python kronos_exec.py --scalp --check-tp-sl               # Check TP/SL + time-stop
"""

import ccxt
import json, os, sys
from datetime import datetime, timedelta
from pathlib import Path

# =========================================================================
# SWING CONFIG (default)
# =========================================================================
LEVERAGE = 200
INITIAL_CAPITAL = 10000.0
DEFAULT_LEDGER = "paper_trades.json"
LEDGER_FILE = Path(__file__).parent / DEFAULT_LEDGER

# INITIAL_CAPITAL is read from ledger's "initial_capital" field (set on first creation).
# Falls back to this default if not present.
DEFAULT_INITIAL_CAPITAL = 10000.0

# Default TP/SL (as fraction of price)
TP_PCT = 0.005   # 0.5% take profit
SL_PCT = 0.003   # 0.3% stop loss
POSITION_PCT = 0.10  # 10% of buying power per trade

# =========================================================================
# SCALP / HFT CONFIG — ultra-tight for high-frequency trading
# =========================================================================
SCALP_TP_PCT = 0.003        # 0.3% take profit (was 0.2)
SCALP_SL_PCT = 0.002        # 0.2% stop loss (was 0.12 — tighter stops were getting eaten by noise)
SCALP_POSITION_PCT = 0.24   # 24% of buying power (~0.75 BTC at $63k/$1k)
SCALP_TIMESTOP_MIN = 20     # Auto-close after 20 min (was 10 — gives TP room to hit)
SCALP_COOLDOWN_MIN = 3      # Skip if last same-direction trade was < N min ago (was 2)
SCALP_TRAIL_ACTIVATE = 0.10 # Activate trailing at +0.10% profit
SCALP_TRAIL_LOCK = 0.05     # Lock in 0.05% profit as new breakeven


def get_btc_price() -> float:
    ex = ccxt.binance({'options': {'defaultType': 'spot'}, 'timeout': 15000})
    return ex.fetch_ticker("BTC/USDT")["last"]

def load_ledger() -> dict:
    if LEDGER_FILE.exists():
        data = json.load(LEDGER_FILE.open())
        data.setdefault("balance", DEFAULT_INITIAL_CAPITAL)
        data.setdefault("positions", [])
        data.setdefault("trades", [])
        data.setdefault("total_pnl", 0.0)
        data.setdefault("cooldown", {})
        # Store initial capital on first load so PnL is always correct
        if "initial_capital" not in data:
            data["initial_capital"] = DEFAULT_INITIAL_CAPITAL
        data.setdefault("initial_capital", data.get("initial_capital", DEFAULT_INITIAL_CAPITAL))
        return data
    return {"balance": DEFAULT_INITIAL_CAPITAL, "positions": [], "trades": [],
            "total_pnl": 0.0, "cooldown": {}, "initial_capital": DEFAULT_INITIAL_CAPITAL}


def save_ledger(ledger: dict):
    LEDGER_FILE.write_text(json.dumps(ledger, indent=2))


def get_total_value(ledger: dict) -> float:
    """Total equity = USDT balance + unrealised PnL from open positions."""
    price = get_btc_price()
    total = ledger["balance"]
    for pos in ledger["positions"]:
        margin = pos["size"] * pos["entry_price"] / LEVERAGE
        if pos["side"] == "buy":
            pnl = (price - pos["entry_price"]) * pos["size"]
        else:
            pnl = (pos["entry_price"] - price) * pos["size"]
        total += margin + pnl
    return total


def check_time_stop(ledger: dict, max_minutes: int = SCALP_TIMESTOP_MIN):
    """Close positions that have been open longer than max_minutes (scalp safety)."""
    now = datetime.now()
    closed = []
    remaining = []
    price = get_btc_price()

    for pos in ledger["positions"]:
        entry_time = datetime.fromisoformat(pos["timestamp"])
        age_min = (now - entry_time).total_seconds() / 60.0
        if age_min < max_minutes:
            remaining.append(pos)
            continue

        # Close due to time-stop
        margin = pos["size"] * pos["entry_price"] / LEVERAGE
        if pos["side"] == "buy":
            pnl_usdt = (price - pos["entry_price"]) * pos["size"]
        else:
            pnl_usdt = (pos["entry_price"] - price) * pos["size"]
        ret = pnl_usdt / margin if margin > 0 else 0

        ledger["balance"] += margin + pnl_usdt
        trade_record = {
            "timestamp": datetime.now().isoformat(),
            "side": pos["side"],
            "close_reason": "TIMESTOP",
            "entry_price": round(pos["entry_price"], 2),
            "exit_price": round(price, 2),
            "size_btc": pos["size"],
            "leverage": LEVERAGE,
            "pnl_usdt": round(pnl_usdt, 2),
            "pnl_pct": round(ret * 100, 2),
            "balance_after": round(ledger["balance"], 2),
        }
        ledger["trades"].append(trade_record)
        closed.append(trade_record)
        print(f"  ⏰ TIMESTOP {pos['side'].upper()} (age={age_min:.0f}m) @ ${price:,.2f} — PnL: ${pnl_usdt:+.2f}")

    ledger["positions"] = remaining
    if closed:
        total_value = get_total_value(ledger)
        ledger["total_pnl"] = round(total_value - ledger.get("initial_capital", DEFAULT_INITIAL_CAPITAL), 2)
        save_ledger(ledger)
    return closed


def check_cooldown(ledger: dict, side: str, cooldown_min: int = SCALP_COOLDOWN_MIN) -> bool:
    """Return True if cooldown is active for this side (skip trading).
    Checks BOTH the ledger cooldown dict AND the last closed trade of the same side."""
    now = datetime.now()
    
    # Check ledger cooldown (set when previous trade was opened)
    if side in ledger.get("cooldown", {}):
        last_time = datetime.fromisoformat(ledger["cooldown"][side])
        elapsed_min = (now - last_time).total_seconds() / 60.0
        if elapsed_min < cooldown_min:
            remaining = cooldown_min - elapsed_min
            print(f"  ⏳ Cooldown active for {side.upper()} ({remaining:.0f}m remaining, ledger check)")
            return True
    
    # Check last closed trade of the same side
    for t in reversed(ledger.get("trades", [])):
        if t.get("side") == side:
            close_time = datetime.fromisoformat(t["timestamp"])
            elapsed_min = (now - close_time).total_seconds() / 60.0
            if elapsed_min < cooldown_min:
                remaining = cooldown_min - elapsed_min
                print(f"  ⏳ Cooldown active for {side.upper()} ({remaining:.0f}m remaining, last closed trade at {t['timestamp'][:16]})")
                return True
            break  # Only check the most recent trade of this side
    
    return False


def set_cooldown(ledger: dict, side: str):
    """Record cooldown timestamp for this side."""
    if "cooldown" not in ledger:
        ledger["cooldown"] = {}
    ledger["cooldown"][side] = datetime.now().isoformat()
    save_ledger(ledger)


def check_tp_sl(ledger: dict):
    """Check all open positions — close any that hit TP or SL."""
    price = get_btc_price()
    closed = []
    remaining = []
    
    for pos in ledger["positions"]:
        hit = False
        pnl_pct = 0.0
        if pos["side"] == "buy":
            ret = (price - pos["entry_price"]) / pos["entry_price"]
            pnl_pct = ret * 100
        else:
            ret = (pos["entry_price"] - price) / pos["entry_price"]
            pnl_pct = ret * 100

        # --- Trailing stop for scalp mode ---
        if pos.get("scalp") and pos.get("trailing_activated") is None and pnl_pct >= SCALP_TRAIL_ACTIVATE:
            pos["trailing_activated"] = True
            pos["trailing_peak"] = pnl_pct
            save_ledger(ledger)  # persist trailing state
            print(f"  📈 Trailing activated for {pos['side'].upper()} at +{pnl_pct:.2f}%")
        if pos.get("trailing_activated"):
            if pnl_pct > pos.get("trailing_peak", 0):
                pos["trailing_peak"] = pnl_pct
            # Lock in profit: if we drop SCALP_TRAIL_LOCK% from peak, close
            locked_profit = pos["trailing_peak"] - SCALP_TRAIL_LOCK
            if pnl_pct <= locked_profit:
                hit = True
                reason = "TRAIL"
                print(f"  🎯 Trailing stop hit: peak +{pos['trailing_peak']:.2f}%, locked at +{locked_profit:.2f}%")

        # Check TP / SL (only if trailing hasn't already triggered)
        margin = pos["size"] * pos["entry_price"] / LEVERAGE
        if not hit and ret * 100 >= pos["tp_pct"]:
            hit = True
            reason = "TP"
        elif not hit and ret * 100 <= -pos["sl_pct"]:
            hit = True
            reason = "SL"
        
        if hit:
            # Simulate real stop/limit fill at trigger price — not live market
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
            close_ret = (close_price - pos["entry_price"]) / pos["entry_price"] if pos["side"] == "buy" else (pos["entry_price"] - close_price) / pos["entry_price"]
            if pos["side"] == "buy":
                pnl_usdt = (close_price - pos["entry_price"]) * pos["size"]
            else:
                pnl_usdt = (pos["entry_price"] - close_price) * pos["size"]
            ledger["balance"] += margin + pnl_usdt
            trade_record = {
                "timestamp": datetime.now().isoformat(),
                "side": pos["side"],
                "close_reason": reason,
                "entry_price": round(pos["entry_price"], 2),
                "exit_price": round(close_price, 2),
                "size_btc": pos["size"],
                "leverage": LEVERAGE,
                "pnl_usdt": round(pnl_usdt, 2),
                "pnl_pct": round(close_ret * 100, 2),
                "balance_after": round(ledger["balance"], 2),
            }
            ledger["trades"].append(trade_record)
            closed.append(trade_record)
            print(f"  {'🟢' if reason=='TP' else '🔴'} CLOSED {pos['side'].upper()} @ ${price:,.2f} ({reason}) — PnL: ${pnl_usdt:+.2f}")
        else:
            remaining.append(pos)
    
    ledger["positions"] = remaining
    if closed:
        total_value = get_total_value(ledger)
        ledger["total_pnl"] = round(total_value - ledger.get("initial_capital", DEFAULT_INITIAL_CAPITAL), 2)
        save_ledger(ledger)
    
    return closed


def paper_trade(side: str, size_btc: float = None, tp_pct: float = TP_PCT,
                sl_pct: float = SL_PCT, scalp: bool = False):
    """
    Execute a leveraged paper trade.
    side: 'buy' (long) or 'sell' (short)
    size_btc: position size in BTC. If None, uses POSITION_PCT of buying power.
    scalp: if True, use scalp TP/SL, position %, and cooldown.
    """
    ledger = load_ledger()
    price = get_btc_price()

    # Use scalp params if requested
    use_tp = SCALP_TP_PCT if scalp else tp_pct
    use_sl = SCALP_SL_PCT if scalp else sl_pct
    use_pos_pct = SCALP_POSITION_PCT if scalp else POSITION_PCT

    # Check cooldown (scalp mode only — prevent oscillation)
    if scalp and check_cooldown(ledger, side):
        print(f"  → Trade skipped by cooldown.")
        return

    # Check TP/SL on existing positions first
    check_tp_sl(ledger)
    
    # Check time-stop (scalp only)
    if scalp:
        check_time_stop(ledger)

    # Calculate position size
    if size_btc is None:
        buying_power = ledger["balance"] * LEVERAGE
        position_usdt = buying_power * use_pos_pct
        size_btc = position_usdt / price
    
    margin_required = size_btc * price / LEVERAGE
    
    if margin_required > ledger["balance"]:
        print(f"❌ Insufficient margin: have ${ledger['balance']:.2f}, need ${margin_required:.2f}")
        return
    
    # Deduct margin
    ledger["balance"] -= margin_required
    
    pos = {
        "side": side,
        "size": size_btc,
        "entry_price": price,
        "timestamp": datetime.now().isoformat(),
        "tp_pct": use_tp * 100,  # store as percentage for display
        "sl_pct": use_sl * 100,
        "scalp": scalp,          # tag for status display
    }
    ledger["positions"].append(pos)
    
    total_value = get_total_value(ledger)
    ledger["total_pnl"] = round(total_value - ledger.get("initial_capital", DEFAULT_INITIAL_CAPITAL), 2)
    save_ledger(ledger)
    
    # Set cooldown (scalp only)
    if scalp:
        set_cooldown(ledger, side)
    
    mode_tag = " [SCALP]" if scalp else ""
    pos_size_usdt = size_btc * price
    print(f"✅ PAPER {side.upper()}{mode_tag} {size_btc:.6f} BTC @ ${price:,.2f}")
    print(f"   Position: ${pos_size_usdt:,.2f} (margin: ${margin_required:.2f})")
    print(f"   TP: +{use_tp*100:.2f}%, SL: -{use_sl*100:.2f}%")
    print(f"   Balance: ${ledger['balance']:,.2f}")
    print(f"   Open positions: {len(ledger['positions'])}")
    print(f"   Total PnL: ${ledger['total_pnl']:+.2f}")


def close_all():
    """Close all open positions at market price."""
    ledger = load_ledger()
    price = get_btc_price()
    
    if not ledger["positions"]:
        print("No open positions to close.")
        return
    
    for pos in ledger["positions"]:
        margin = pos["size"] * pos["entry_price"] / LEVERAGE
        if pos["side"] == "buy":
            pnl_usdt = (price - pos["entry_price"]) * pos["size"]
        else:
            pnl_usdt = (pos["entry_price"] - price) * pos["size"]
        
        # Record the closed trade
        ledger.setdefault("trades", []).append({
            "timestamp": datetime.now().isoformat(),
            "side": pos["side"],
            "close_reason": "MANUAL",
            "entry_price": pos["entry_price"],
            "exit_price": price,
            "size_btc": pos["size"],
            "leverage": LEVERAGE,
            "pnl_usdt": round(pnl_usdt, 2),
            "pnl_pct": round(pnl_usdt / max(margin, 0.01) * 100, 2),
            "balance_after": round(ledger["balance"] + margin + pnl_usdt, 2),
        })
        
        ledger["balance"] += margin + pnl_usdt
        print(f"  🔴 CLOSED {pos['side'].upper()} {pos['size']:.6f} BTC @ ${price:,.2f} — PnL: ${pnl_usdt:+.2f}")
    
    ledger["positions"] = []
    total_value = get_total_value(ledger)
    ledger["total_pnl"] = round(total_value - ledger.get("initial_capital", DEFAULT_INITIAL_CAPITAL), 2)
    save_ledger(ledger)
    print(f"  All positions closed. Balance: ${ledger['balance']:,.2f}, PnL: ${ledger['total_pnl']:+.2f}")


def show_status(scalp: bool = False):
    price = get_btc_price()
    ledger = load_ledger()
    total_value = get_total_value(ledger)
    real_pnl = total_value - ledger.get("initial_capital", DEFAULT_INITIAL_CAPITAL)
    # Update live PnL in ledger
    ledger["total_pnl"] = round(real_pnl, 2)
    save_ledger(ledger)
    
    mode_tag = " [SCALP MODE]" if scalp else ""
    print(f"\n{'='*55}")
    print(f"  KRONOS PAPER TRADER{mode_tag} — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*55}")
    print(f"  BTC/USDT:    ${price:,.2f}")
    print(f"  Balance:     ${ledger['balance']:,.2f}")
    print(f"  Total Value: ${total_value:,.2f}")
    print(f"  PnL:         ${real_pnl:+.2f}")
    
    # Show scalp config info
    if scalp:
        print(f"  ─────────────────────────────────────")
        print(f"  SCALP: TP={SCALP_TP_PCT*100:.1f}% SL={SCALP_SL_PCT*100:.1f}% "
              f"Pos={SCALP_POSITION_PCT:.0%} TimeStop={SCALP_TIMESTOP_MIN}m")
        lep = ledger["cooldown"].get("buy") or ledger["cooldown"].get("sell")
        if lep:
            print(f"  Cooldown expiry: {lep[:16]}")
    
    print(f"\n  📋 Open Positions ({len(ledger['positions'])}):")
    if ledger["positions"]:
        for i, pos in enumerate(ledger["positions"], 1):
            pos_value = pos["size"] * LEVERAGE * pos["entry_price"]
            margin = pos["size"] * pos["entry_price"] / LEVERAGE
            if pos["side"] == "buy":
                unrealized = (price - pos["entry_price"]) / pos["entry_price"] * 100
            else:
                unrealized = (pos["entry_price"] - price) / pos["entry_price"] * 100
            # Calculate age
            age_min = (datetime.now() - datetime.fromisoformat(pos["timestamp"])).total_seconds() / 60.0
            scalp_tag = " [SCALP]" if pos.get("scalp") else ""
            print(f"  {i}. {pos['side'].upper()}{scalp_tag} {pos['size']:.6f} BTC @ ${pos['entry_price']:,.2f}")
            print(f"     Exposure: ${pos_value:,.0f} | Margin: ${margin:.0f} | Age: {age_min:.0f}m")
            print(f"     Unrealized: {unrealized:+.2f}% | TP: +{pos['tp_pct']:.1f}% | SL: -{pos['sl_pct']:.1f}%")
    else:
        print(f"  (none)")
    
    total_trades = len(ledger["trades"])
    print(f"\n  📜 Recent Trades ({total_trades} total):")
    
    # Calculate trade rate if we have trades
    if total_trades > 0:
        first_trade = datetime.fromisoformat(ledger["trades"][0]["timestamp"])
        days_active = max((datetime.now() - first_trade).total_seconds() / 86400.0, 0.01)
        trades_per_day = total_trades / days_active
        print(f"  ⚡ Trade rate: {trades_per_day:.0f} trades/day")
    
    for t in ledger["trades"][-5:]:
        reason = t.get('close_reason', 'OPEN')
        print(f"  {t['timestamp'][:16]} | {t['side'].upper():4s} | {reason:8s} | PnL: ${t['pnl_usdt']:+.2f}")
    
    print(f"\n  💡 Commands:")
    print(f"  python kronos_exec.py --paper buy  --size 0.001")
    print(f"  python kronos_exec.py --scalp --paper buy")
    print(f"  python kronos_exec.py --paper close")
    print()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Kronos Paper Trade Executor")
    parser.add_argument("--paper", choices=["buy", "sell", "close"], help="Trade action")
    parser.add_argument("--size", type=float, default=None, help="Position size in BTC (default: auto)")
    parser.add_argument("--status", action="store_true", help="Show account status")
    parser.add_argument("--check-tp-sl", action="store_true", help="Check TP/SL on open positions")
    parser.add_argument("--scalp", action="store_true", help="Scalp/HFT mode: tight TP/SL, cooldown, time-stop")
    parser.add_argument("--close-trade", type=int, nargs="?", const=-1, default=None,
                        help="Close a specific position by index (1-based), or all if no index given")
    parser.add_argument("--ledger", type=str, default=None,
                        help="Ledger filename (default: paper_trades.json)")
    args = parser.parse_args()

    # Set custom ledger if specified
    if args.ledger:
        LEDGER_FILE = Path(__file__).parent / args.ledger

    if args.close_trade is not None:
        if args.close_trade == -1:
            # Close all positions
            close_all()
        else:
            # Close a specific position by index (1-based)
            idx = args.close_trade - 1  # convert to 0-based
            ledger = load_ledger()
            if idx < 0 or idx >= len(ledger["positions"]):
                print(f"Invalid position index {args.close_trade}. Open positions: {len(ledger['positions'])}")
                sys.exit(1)
            pos = ledger["positions"][idx]
            price = get_btc_price()
            margin = pos["size"] * pos["entry_price"] / LEVERAGE
            if pos["side"] == "buy":
                pnl_usdt = (price - pos["entry_price"]) * pos["size"]
            else:
                pnl_usdt = (pos["entry_price"] - price) * pos["size"]
            ledger["balance"] += margin + pnl_usdt
            ledger["positions"].pop(idx)
            trade_record = {
                "timestamp": datetime.now().isoformat(),
                "side": pos["side"],
                "close_reason": "MANUAL",
                "entry_price": round(pos["entry_price"], 2),
                "exit_price": round(price, 2),
                "size_btc": pos["size"],
                "leverage": LEVERAGE,
                "pnl_usdt": round(pnl_usdt, 2),
                "pnl_pct": round(pnl_usdt / margin * 100, 2) if margin > 0 else 0,
                "balance_after": round(ledger["balance"], 2),
            }
            ledger["trades"].append(trade_record)
            total_value_after = ledger["balance"]  # no open positions left
            ledger["total_pnl"] = round(total_value_after - ledger.get("initial_capital", DEFAULT_INITIAL_CAPITAL), 2)
            save_ledger(ledger)
            print(f"🔴 MANUAL CLOSE #{args.close_trade}: {pos['side'].upper()} {pos['size']:.6f} BTC @ ${price:,.2f}")
            print(f"   PnL: ${pnl_usdt:+.2f} | Balance: ${ledger['balance']:,.2f}")
    elif args.status:
        show_status(scalp=args.scalp)
    elif args.paper == "close":
        close_all()
    elif args.paper:
        paper_trade(args.paper, args.size, scalp=args.scalp)
    elif args.check_tp_sl:
        ledger = load_ledger()
        closed = check_tp_sl(ledger)
        if args.scalp:
            times_topped = check_time_stop(ledger)
            if not closed and not times_topped:
                print("No positions hit TP/SL or time-stop.")
        elif not closed:
            print("No positions hit TP/SL.")
    else:
        show_status(scalp=args.scalp)
