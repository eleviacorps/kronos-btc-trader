#!/usr/bin/env python3
"""TP/SL watcher for fusion live account (paper_trades_live.json) — runs every 1m."""
import subprocess, sys, os
from pathlib import Path

NO_WINDOW = 0x08000000 if os.name == 'nt' else 0
PROJECT_DIR = Path(r"D:\Programming\AiProjects\Kronos(reviewing)\Kronos")
VENV_PYTHON = PROJECT_DIR / ".venv" / "Scripts" / "python.exe"

os.chdir(PROJECT_DIR)
result = subprocess.run(
    [str(VENV_PYTHON), str(PROJECT_DIR / "kronos_exec.py"), "--scalp", "--check-tp-sl", "--ledger", "paper_trades_live.json"],
    cwd=PROJECT_DIR,
    capture_output=True, text=True, timeout=30,
    creationflags=NO_WINDOW,
)
sys.stdout.write(result.stdout)
sys.stderr.write(result.stderr)
sys.exit(result.returncode)
