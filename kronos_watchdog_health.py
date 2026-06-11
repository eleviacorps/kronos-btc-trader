#!/usr/bin/env python3
"""Health-check cron for kronos_watchdog — restarts if dead. Runs every 5m."""
import subprocess, sys, os, time, json
from pathlib import Path
from datetime import datetime

NO_WINDOW = 0x08000000 if os.name == 'nt' else 0
PROJECT_DIR = Path(r"D:\Programming\AiProjects\Kronos(reviewing)\Kronos")
VENV_PYTHON = PROJECT_DIR / ".venv" / "Scripts" / "python.exe"
WATCHDOG = PROJECT_DIR / "kronos_watchdog.py"
PID_FILE = PROJECT_DIR / "kronos_watchdog.pid"
LOG_FILE = PROJECT_DIR / "kronos_watchdog.log"

os.chdir(PROJECT_DIR)

# Check if process is alive from PID file
if PID_FILE.exists():
    pid = int(PID_FILE.read_text().strip())
    try:
        os.kill(pid, 0)  # signal 0 = check existence only
        # Process exists — good
        sys.exit(0)
    except OSError:
        pass  # Dead, restart below

# Write a fresh log entry
with open(LOG_FILE, "a") as f:
    f.write(f"[{datetime.now().isoformat()}] HEALTH-CHECK: Watchdog dead, restarting...\n")

# Re-launch
proc = subprocess.Popen(
    [str(VENV_PYTHON), str(WATCHDOG)],
    cwd=PROJECT_DIR,
    creationflags=NO_WINDOW,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)
PID_FILE.write_text(str(proc.pid))

with open(LOG_FILE, "a") as f:
    f.write(f"[{datetime.now().isoformat()}] HEALTH-CHECK: Restarted watchdog (pid={proc.pid})\n")

print(f"Watchdog restarted (pid={proc.pid})")
