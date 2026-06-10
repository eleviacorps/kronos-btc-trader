#!/usr/bin/env python3
"""
kronos_watcher.py — Lightweight TP/SL + time-stop watcher.
Runs every 60 seconds, checks all open positions.
No prediction, no new trades — just closes what needs closing.
Meant to run alongside the dashboard.
"""
import json, os, sys, time, subprocess
from pathlib import Path

PROJECT_DIR = Path(r"D:\Programming\AiProjects\Kronos(reviewing)\Kronos")
VENV_PYTHON = PROJECT_DIR / ".venv" / "Scripts" / "python.exe"

def run(cmd, timeout=30):
    result = subprocess.run([str(VENV_PYTHON)] + cmd.split(), capture_output=True, text=True, timeout=timeout, cwd=PROJECT_DIR)
    return (result.stdout + result.stderr).strip()

print("KRONOS WATCHER — TP/SL check every 60s")
print("Press Ctrl+C to stop.")
print()

while True:
    out = run("kronos_exec.py --scalp --check-tp-sl")
    if out and "No positions" not in out:
        print(f"[{time.strftime('%H:%M:%S')}] {out}")
    time.sleep(60)
