#!/usr/bin/env python3
"""
kronos_watchdog.py — Live market TP/SL enforcer daemon.
Polling Binance price every 5s, closes positions at trigger price instantly.

Decoupled from signal generation (kronos_scalper.py cron) — this only enforces
TP/SL/TIMESTOP on open positions so slippage from cron-gaps is eliminated.

Usage:
  python kronos_watchdog.py                     # Default: both ledgers, 5s interval
  python kronos_watchdog.py --interval 3         # Custom poll interval (seconds)
  python kronos_watchdog.py --ledger paper_trades_agent.json  # Single ledger
"""
import sys, os, time, json, signal
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field

PROJECT_DIR = Path(r"D:\Programming\AiProjects\Kronos(reviewing)\Kronos")
os.chdir(PROJECT_DIR)

# Force import from the project's kronos_exec — it uses the venv's ccxt
sys.path.insert(0, str(PROJECT_DIR))
import importlib.util
exec_spec = importlib.util.spec_from_file_location("kronos_exec", PROJECT_DIR / "kronos_exec.py")
kexec = importlib.util.module_from_spec(exec_spec)
exec_spec.loader.exec_module(kexec)

# Re-export what we need
check_tp_sl = kexec.check_tp_sl
check_time_stop = kexec.check_time_stop
load_ledger = kexec.load_ledger
save_ledger = kexec.save_ledger

# =========================================================================
# CONFIG
# =========================================================================
DEFAULT_INTERVAL_S = 5
HEARTBEAT_INTERVAL_S = 300  # log a heartbeat every 5 min

LEDGERS = [
    {"file": "paper_trades.json",         "name": "BOT"},
    {"file": "paper_trades_agent.json",   "name": "AGENT"},
    {"file": "paper_trades_live.json",    "name": "LIVE"},
]

LOG_FILE = PROJECT_DIR / "kronos_watchdog.log"


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(f"[{datetime.now().isoformat()}] {msg}\n")


def check_one_ledger(ledger_file: str, name: str) -> bool:
    """Check TP/SL + time-stop for one ledger. Returns True if any position was closed."""
    original = kexec.LEDGER_FILE
    kexec.LEDGER_FILE = PROJECT_DIR / ledger_file
    try:
        ledger = load_ledger()
        if not ledger["positions"]:
            return False
        closed_tp_sl = check_tp_sl(ledger)
        closed_ts = check_time_stop(ledger)
        if closed_tp_sl or closed_ts:
            for t in closed_tp_sl + closed_ts:
                pnl = t["pnl_usdt"]
                log(f"[{name}] CLOSED {t['side'].upper()} {t['close_reason']} "
                     f"@ ${t['exit_price']:,.2f} — PnL: ${pnl:+.2f}")
            # Print balance state
            bal = ledger.get("balance", 0)
            log(f"[{name}] Balance: ${bal:,.2f} | Positions: {len(ledger['positions'])}")
            return True
        return False
    except Exception as e:
        log(f"[{name}] ERROR: {e}")
        return False
    finally:
        kexec.LEDGER_FILE = original


def daemon(interval_s: int = DEFAULT_INTERVAL_S):
    # Write PID file for health-check
    PID_FILE = PROJECT_DIR / "kronos_watchdog.pid"
    try:
        PID_FILE.write_text(str(os.getpid()))
    except Exception as e:
        print(f"[startup] Could not write PID file: {e}", file=sys.stderr)

    log(f"KRONOS WATCHDOG STARTED (poll every {interval_s}s, {len(LEDGERS)} ledgers, pid={os.getpid()})")
    log(f"  Ledgers: {', '.join(l['name'] + '=' + l['file'] for l in LEDGERS)}")
    log(f"  Log file: {LOG_FILE}")

    last_heartbeat = time.monotonic()
    running = True

    def shutdown(sig, frame):
        nonlocal running
        log(f"Shutdown signal received (signal={sig}) — exiting.")
        running = False

    signal.signal(signal.SIGINT, shutdown)
    # SIGTERM not available on Windows; skip gracefully
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, shutdown)

    cycle = 0
    while running:
        try:
            cycle += 1
            now = time.monotonic()

            # Heartbeat
            if now - last_heartbeat >= HEARTBEAT_INTERVAL_S:
                last_heartbeat = now
                bal_bot = kexec.LEDGER_FILE
                try:
                    kexec.LEDGER_FILE = PROJECT_DIR / "paper_trades.json"
                    b = load_ledger()
                    b_pos = len(b["positions"])
                    b_bal = b.get("balance", 0)
                except:
                    b_pos, b_bal = "?", "?"
                
                try:
                    kexec.LEDGER_FILE = PROJECT_DIR / "paper_trades_agent.json"
                    a = load_ledger()
                    a_pos = len(a["positions"])
                    a_bal = a.get("balance", 0)
                except:
                    a_pos, a_bal = "?", "?"
                
                kexec.LEDGER_FILE = bal_bot
                log(f"❤️  (cycle {cycle}) BOT: {b_pos} pos, ${b_bal} | AGENT: {a_pos} pos, ${a_bal}")

            # Check each ledger
            for l in LEDGERS:
                check_one_ledger(l["file"], l["name"])

            # Sleep with interruptible wait
            for _ in range(interval_s):
                if not running:
                    break
                time.sleep(1)
        except Exception as e:
            log(f"CRASH in main loop: {e}")
            import traceback
            log(traceback.format_exc())
            raise  # Let health-check cron restart us

    log("Watchdog exited cleanly.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Kronos TP/SL Watchdog Daemon")
    parser.add_argument("--interval", type=int, default=DEFAULT_INTERVAL_S,
                        help=f"Poll interval in seconds (default: {DEFAULT_INTERVAL_S})")
    args = parser.parse_args()
    daemon(interval_s=args.interval)
