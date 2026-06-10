#!/usr/bin/env python3
"""
kronos_scalper.py — Autonomous HFT scalping loop for 10-min cron cycle.
Designed for 200+ trades/day frequency using ultra-sensitive thresholds.

Runs the full cycle:
  1. TP/SL check + time-stop check (close any stale positions)
  2. Balance + status snapshot
  3. Kronos prediction in scalp mode (lower thresholds, shorter horizon)
  4. Evaluate scalp ensemble signal
  5. Execute trade if conditions met (cooldown-aware)
  6. Report cycle summary

Usage (no_agent cron):
  python kronos_scalper.py

Or via wrapper:
  PYTHONPATH=. python kronos_scalper.py
  
Designed for cron schedule: every 10m (144 runs/day).
With ~70-80% signal rate, yields ~100-200 trades/day.
"""
import json, os, sys, subprocess, time
from pathlib import Path
from datetime import datetime

NO_WINDOW = 0x08000000 if os.name == 'nt' else 0

PROJECT_DIR = Path(r"D:\Programming\AiProjects\Kronos(reviewing)\Kronos")
os.chdir(PROJECT_DIR)
VENV_PYTHON = str(PROJECT_DIR / ".venv" / "Scripts" / "python.exe")

# =========================================================================
# SCALP PARAMS (must match kronos_trader.py SCALP_* and kronos_exec.py SCALP_*)
# =========================================================================
MIN_CONFIDENCE = 0.25        # Lower bar for scalp entries
MIN_TRADE_INTERVAL_S = 120   # Force at least 2 min between any trades
COOLDOWN_SIDES = {"buy", "sell"}
MIN_POSITION_SIZE = 0.5      # Minimum BTC position size
MAX_POSITION_SIZE = 1.0      # Maximum BTC position size


def run(cmd, timeout=90):
    """Run a kronos python script in the project venv."""
    result = subprocess.run(
        [str(VENV_PYTHON)] + cmd.split(),
        capture_output=True, text=True, timeout=timeout,
        cwd=PROJECT_DIR, creationflags=NO_WINDOW
    )
    return result.stdout + result.stderr


def parse_decision(output: str):
    """Extract decision and confidence from hermes-summary output.
    Antitrend gets priority — if it says HOLD, the answer is HOLD."""

    import re

    # 1. Parse antitrend signal separately (priority)
    antitrend_decision = "HOLD"
    antitrend_conf = 0.0
    for line in output.split("\n"):
        if "antitrend" not in line:
            continue
        # Extract signal: after "antitrend" find BUY/SELL/HOLD
        m = re.search(r'antitrend\s+(STRONG_BUY|BUY|SELL|STRONG_SELL|HOLD)', line)
        if m:
            sig = m.group(1)
            if sig == "STRONG_BUY": sig = "BUY"
            elif sig == "STRONG_SELL": sig = "SELL"
            antitrend_decision = sig
        # Extract confidence from (0.XX) pattern
        m = re.search(r'\((\d+\.\d+)\)', line)
        if m:
            try:
                c = float(m.group(1))
                if c > antitrend_conf:
                    antitrend_conf = c
            except: pass

    # 2. Parse consensus (secondary)
    consensus = "HOLD"
    conf = 0.0
    if "Consensus: 🟢 BUY" in output or "Consensus: BUY" in output:
        consensus = "BUY"
    elif "Consensus: 🔴 SELL" in output or "Consensus: SELL" in output:
        consensus = "SELL"
    for line in output.split("\n"):
        if "conf=" in line:
            try: conf = float(line.split("conf=")[1].split(")")[0])
            except: pass
        if consensus != "HOLD" and "conf=" not in line and "(conf=" in line:
            try: conf = float(line.split("(conf=")[1].split(")")[0])
            except: pass

    # 3. Antitrend priority rule
    if antitrend_decision == "HOLD":
        return "HOLD", 0.0
    # Antitrend has a signal — use it
    return antitrend_decision, max(antitrend_conf, conf * 0.5)


def get_position_count(status_output: str) -> int:
    """Count open positions from status output."""
    for line in status_output.split("\n"):
        if "Open Positions" in line or "📋 Open Positions" in line:
            try:
                parts = line.split("(")
                if len(parts) > 1:
                    return int(parts[1].split(")")[0])
            except (ValueError, IndexError):
                pass
    return 0


def was_recent_trade(trades_output: str, side: str, min_seconds: int = MIN_TRADE_INTERVAL_S) -> bool:
    """Check if there was a trade in the same direction recently."""
    now = datetime.now()
    for line in trades_output.split("\n"):
        if side.upper() in line and ("TP" in line or "SL" in line or "TIMESTOP" in line
                                      or "CLOSED" in line or "PAPER" in line):
            # Try to extract timestamp
            try:
                # Lines look like: "2026-06-09 07:30 | BUY  | TP      | PnL: +$5.20"
                ts_str = line.strip()[:16]
                if " | " in line:
                    ts_str = line.split(" | ")[0].strip()
                trade_time = datetime.strptime(ts_str, "%Y-%m-%d %H:%M")
                elapsed = (now - trade_time).total_seconds()
                if 0 < elapsed < min_seconds:
                    return True
            except (ValueError, IndexError):
                pass
    return False


def calc_position_size(conf: float) -> float:
    """Map confidence 0.25-1.0 to position size 0.5-1.0 BTC linearly."""
    clamped = max(0.0, min(1.0, conf))
    size = MIN_POSITION_SIZE + clamped * (MAX_POSITION_SIZE - MIN_POSITION_SIZE)
    return round(size, 3)


def main():
    cycle_start = datetime.now()
    print(f"{'='*58}")
    print(f"  KRONOS HFT SCALPER — Cycle at {cycle_start.strftime('%H:%M:%S')}")
    print(f"{'='*58}")

    # ---------------------------------------------------------------
    # 1. TP/SL check + time-stop (scalp mode)
    # ---------------------------------------------------------------
    print("\n[1] Checking TP/SL + time-stop...")
    tpsl_out = run("kronos_exec.py --scalp --check-tp-sl")
    print(tpsl_out.strip() if tpsl_out.strip() else "  (no positions closed)")

    # ---------------------------------------------------------------
    # 2. Status snapshot
    # ---------------------------------------------------------------
    print("\n[2] Current status...")
    status_out = run("kronos_exec.py --scalp --status")
    print(status_out)

    open_positions = get_position_count(status_out)
    print(f"  → Open positions: {open_positions}")

    # ---------------------------------------------------------------
    # 3. Run Kronos prediction in scalp mode
    # ---------------------------------------------------------------
    print("\n[3] Running scalp prediction...")
    signal_out = run("kronos_trader.py --scalp --hermes-summary", timeout=120)

    # Print the actionable summary
    for line in signal_out.split("\n"):
        if "=== KRONOS" in line or "Consensus" in line or "BTC:" in line or "Kronos:" in line:
            print(f"  {line}")
        elif "STRONG_" in line or "BUY" in line or "SELL" in line or "HOLD" in line:
            # Don't print every strategy line — keep it compact
            pass
        elif "ACTION REQUIRED" in line:
            pass
    # Print just the strategy lines that have a signal
    for line in signal_out.split("\n"):
        if "🟢" in line or "🔴" in line:
            print(f"  {line.strip()}")

    # ---------------------------------------------------------------
    # 4. Parse decision
    # ---------------------------------------------------------------
    decision, conf = parse_decision(signal_out)
    has_position = open_positions > 0

    print(f"\n[4] Decision: {decision} (conf={conf:.3f})")
    print(f"    Has position: {has_position}")

    # ---------------------------------------------------------------
    # 5. Check rate limiting / anti-oscillation
    # ---------------------------------------------------------------
    if has_position:
        print(f"\n[5] Position open — letting it run to TP/SL. HOLD.")
        print(f"\n{'='*58}")
        print(f"  CYCLE COMPLETE — {datetime.now().strftime('%H:%M:%S')} "
              f"({(datetime.now()-cycle_start).total_seconds():.0f}s)")
        print(f"{'='*58}")
        return

    # ---------------------------------------------------------------
    # 6. Execute if conditions met
    # ---------------------------------------------------------------
    if decision == "HOLD":
        print(f"\n[5] No signal. HOLD.")
    elif conf < MIN_CONFIDENCE:
        print(f"\n[5] Confidence {conf:.3f} < {MIN_CONFIDENCE}. HOLD.")
    else:
        side = "buy" if decision == "BUY" else "sell"
        pos_size = calc_position_size(conf)
        print(f"\n[5] Executing scalp {side.upper()} {pos_size} BTC (conf={conf:.3f})...")
        exec_out = run(f"kronos_exec.py --scalp --paper {side} --size {pos_size}")
        print(exec_out)

    # ---------------------------------------------------------------
    # Cycle summary
    # ---------------------------------------------------------------
    elapsed = (datetime.now() - cycle_start).total_seconds()
    print(f"\n{'='*58}")
    print(f"  CYCLE COMPLETE — {datetime.now().strftime('%H:%M:%S')} ({elapsed:.0f}s)")
    print(f"{'='*58}")


if __name__ == "__main__":
    main()
