#!/usr/bin/env python3
"""Wrapper for kronos_ev_hermes_collector.py — no-agent cron."""
import subprocess, sys, os
from pathlib import Path
NO_WINDOW = 0x08000000 if os.name == 'nt' else 0
PROJECT_DIR = Path(r"D:\Programming\AiProjects\Kronos(reviewing)\Kronos")
VENV_PYTHON = PROJECT_DIR / ".venv" / "Scripts" / "python.exe"
SCRIPT = PROJECT_DIR / "kronos_ev_hermes_collector.py"
os.chdir(PROJECT_DIR)
result = subprocess.run([str(VENV_PYTHON), str(SCRIPT)], cwd=PROJECT_DIR,
    capture_output=True, text=True, timeout=180, creationflags=NO_WINDOW)
sys.stdout.write(result.stdout)
sys.stderr.write(result.stderr)
sys.exit(result.returncode)
