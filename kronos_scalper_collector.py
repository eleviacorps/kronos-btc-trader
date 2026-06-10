#!/usr/bin/env python3
"""
kronos_scalper_collector.py — Data collector for Hermes-judged HFT scalping.

Collects TP/SL check + account status + prediction signal for Hermes to review.
Does NOT execute any trades — that's Hermes's job as the final decision-maker.

Designed as a cron data-collection script (agent mode).
"""
import os, subprocess, time, sys as _sys
from pathlib import Path
from datetime import datetime

NO_WINDOW = 0x08000000 if os.name == 'nt' else 0

PROJECT_DIR = Path(r"D:\Programming\AiProjects\Kronos(reviewing)\Kronos")
VENV_PYTHON = PROJECT_DIR / ".venv" / "Scripts" / "python.exe"
LEDGER_ARG = []


def run(cmd, timeout=90, passthrough=False):
    full_cmd = [str(VENV_PYTHON)] + cmd.split()
    if not passthrough:
        full_cmd += LEDGER_ARG
    result = subprocess.run(
        full_cmd,
        capture_output=True, text=True, timeout=timeout,
        cwd=PROJECT_DIR, creationflags=NO_WINDOW
    )
    return result.stdout + result.stderr


# Parse --ledger if specified (for agent-mode cron with separate account)
for _i, _a in enumerate(_sys.argv[1:], 1):
    if _a == "--ledger" and _i < len(_sys.argv):
        LEDGER_ARG = ["--ledger", _sys.argv[_i + 1]]

print(f"{'='*60}")
print(f"  KRONOS SCALP DATA COLLECTOR")
print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'='*60}")

# 1. TP/SL check — close any positions that hit target/stop
print(f"\n{'─'*40}")
print("  [1] TP/SL + TIME-STOP CHECK")
print(f"{'─'*40}")
tpsl_out = run("kronos_exec.py --scalp --check-tp-sl")
print(tpsl_out.strip() if tpsl_out.strip() else "  (no positions closed)")

# 2. Status — current balance, open positions, recent trades
print(f"\n{'─'*40}")
print("  [2] ACCOUNT STATUS")
print(f"{'─'*40}")
print(run("kronos_exec.py --scalp --status"))

# 3. Prediction — Kronos scalp-mode signal
print(f"{'─'*40}")
print("  [3] PREDICTION SIGNAL (KRONOS SCALP)")
print(f"{'─'*40}")
signal_out = run("kronos_trader.py --scalp --hermes-summary", timeout=120, passthrough=True)
# Print only the key signal lines, not the full verbose output
for line in signal_out.split("\n"):
    if any(kw in line for kw in [
        "=== KRONOS", "Consensus:", "BTC:", "Kronos:",
        "STRONG_", "🟢", "🔴", "HOLD", "antitrend",
        "net_change_pct", "range_pct", "ACTION REQUIRED"
    ]):
        print(f"  {line.strip()}")

print(f"\n{'='*60}")
print(f"  END DATA — Hermes, review the above and make the call.")
print(f"{'='*60}")
