#!/usr/bin/env python3
"""
kronos_autonomous.py — Runs the full Kronos trading loop without agent.
Prediction → TP/SL check → Status → Decision → Execution.
No LLM API calls needed. Designed for no_agent cron job.
"""
import json, os, sys, subprocess, time
from pathlib import Path

PROJECT_DIR = Path(r"D:\Programming\AiProjects\Kronos(reviewing)\Kronos")
os.chdir(PROJECT_DIR)
VENV_PYTHON = str(PROJECT_DIR / ".venv" / "Scripts" / "python.exe")


def run(cmd):
    result = subprocess.run(
        [str(VENV_PYTHON)] + cmd.split(),
        capture_output=True, text=True, timeout=60,
        cwd=PROJECT_DIR
    )
    return result.stdout + result.stderr


def main():
    print("=== KRONOS AUTONOMOUS TRADER ===")

    # 1. TP/SL check — close any positions that hit
    print("\n[1] Checking TP/SL...")
    print(run("kronos_exec.py --check-tp-sl"))

    # 2. Get status
    print("\n[2] Current status...")
    status_out = run("kronos_exec.py --status")
    print(status_out)

    # 3. Check if we have an open position
    has_position = "Open Positions (1)" in status_out or "📋 Open Positions (1)" in status_out

    # 4. Run prediction
    print("\n[3] Running prediction...")
    signal_out = run("kronos_trader.py --hermes-summary")
    print(signal_out)

    # 5. Parse decision from signal output
    if "Consensus: 🟢 BUY" in signal_out or "Consensus: BUY" in signal_out:
        decision = "BUY"
    elif "Consensus: 🔴 SELL" in signal_out or "Consensus: SELL" in signal_out:
        decision = "SELL"
    else:
        decision = "HOLD"

    conf = 0.0
    for line in signal_out.split("\n"):
        if "conf=" in line:
            try:
                conf = float(line.split("conf=")[1].split(")")[0])
            except:
                pass

    print(f"\n[4] Decision: {decision} (conf={conf:.2f})")
    print(f"    Has position: {has_position}")

    # 6. Execute if conditions met
    if decision == "HOLD":
        print("\n[5] No trade. HOLD.")
    elif has_position:
        print(f"\n[5] Already in a position. Skipping new {decision}.")
    elif conf < 0.3:
        print(f"\n[5] Confidence {conf:.2f} too low. HOLD.")
    else:
        size = "0.001"
        side = "buy" if decision == "BUY" else "sell"
        print(f"\n[5] Executing {side.upper()} {size} BTC...")
        print(run(f"kronos_exec.py --paper {side} --size {size}"))

    print("\n=== CYCLE COMPLETE ===")


if __name__ == "__main__":
    main()
