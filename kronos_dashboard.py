#!/usr/bin/env python3
"""
kronos_dashboard.py — Live CLI dashboard for the Kronos HFT scalper.
Auto-refreshes every 3 seconds. Shows account, positions, trade history.

Usage:
  python kronos_dashboard.py              # Read-only dashboard
  python kronos_dashboard.py --watch      # Also run TP/SL watcher every 60s
  python kronos_dashboard.py --run        # Run full scalper loop + dashboard
"""
import json, os, sys, time, subprocess, threading
from pathlib import Path
from datetime import datetime, timedelta
from collections import deque

from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.console import Console
from rich.text import Text
from rich import box

PROJECT_DIR = Path(r"D:\Programming\AiProjects\Kronos(reviewing)\Kronos")
LEDGER = PROJECT_DIR / "paper_trades.json"
VENV_PYTHON = PROJECT_DIR / ".venv" / "Scripts" / "python.exe"

console = Console()

# ── Helpers ──

def load():
    if LEDGER.exists():
        return json.loads(LEDGER.read_text())
    return {"balance": 1000.0, "positions": [], "trades": [], "total_pnl": 0.0, "cooldown": {}}

def run(cmd, timeout=120):
    try:
        result = subprocess.run([str(VENV_PYTHON)] + cmd.split(), capture_output=True, text=True, timeout=timeout, cwd=PROJECT_DIR)
        return (result.stdout + result.stderr).strip()
    except:
        return "[timeout/error]"

def get_btc_price():
    import ccxt
    try:
        ex = ccxt.binance({'timeout': 10000})
        return ex.fetch_ticker("BTC/USDT")["last"]
    except:
        return None

# ── Background watcher ──

class Watcher:
    def __init__(self):
        self.log = deque(maxlen=20)
        self.running = False
    
    def start(self, do_scalp=False):
        self.running = True
        self.do_scalp = do_scalp
        t = threading.Thread(target=self._loop, daemon=True)
        t.start()
    
    def _loop(self):
        while self.running:
            # TP/SL check every 60s
            out = run("kronos_exec.py --scalp --check-tp-sl", timeout=30)
            if out and "No positions" not in out:
                self.log.append(f"[{time.strftime('%H:%M:%S')}] {out}")
            
            # Full scalper cycle every 10min
            if self.do_scalp and int(time.time()) % 600 < 60:
                self.log.append(f"[{time.strftime('%H:%M:%S')}] 🔄 Running scalper cycle...")
                out = run("kronos_scalper.py", timeout=180)
                for line in out.split("\n")[:15]:
                    if line.strip():
                        self.log.append(f"  {line.strip()[:120]}")
                self.log.append(f"[{time.strftime('%H:%M:%S')}] ✅ Cycle done")
            
            time.sleep(60)

# ── Build dashboard ──

def build_dashboard(data, price, watcher_log):
    now = datetime.now()
    
    # ── Account Panel ──
    balance = data.get("balance", 0)
    total_pnl = data.get("total_pnl", 0)
    trades_list = data.get("trades", [])
    positions = data.get("positions", [])
    total_trades = len(trades_list)
    
    # Win rate
    if total_trades > 0:
        wins = sum(1 for t in trades_list if t.get("pnl_usdt", 0) > 0)
        win_rate = f"{wins/total_trades*100:.0f}%"
    else:
        win_rate = "N/A"
    
    # Trade rate
    if total_trades > 1:
        first = datetime.fromisoformat(trades_list[0]["timestamp"])
        days = max((now - first).total_seconds() / 86400, 0.01)
        trade_rate = f"{total_trades/days:.0f}/day"
    else:
        trade_rate = "N/A"
    
    # Total value (mark-to-market)
    total_value = balance
    price_val = price or 0
    for pos in positions:
        margin = pos["size"] * pos["entry_price"] / 200
        if pos["side"] == "buy":
            pnl = (price_val - pos["entry_price"]) * pos["size"]
        else:
            pnl = (pos["entry_price"] - price_val) * pos["size"]
        total_value += margin + pnl
    
    pnl_color = "green" if total_pnl >= 0 else "red"
    
    account_table = Table(show_header=False, box=box.SIMPLE, padding=(0,1))
    account_table.add_column("Field", style="cyan")
    account_table.add_column("Value")
    account_table.add_row("Balance", f"${balance:,.2f}")
    account_table.add_row("Total Value", f"${total_value:,.2f}")
    account_table.add_row("PnL", Text(f"${total_pnl:+,.2f}", style=pnl_color))
    account_table.add_row("Trades", str(total_trades))
    account_table.add_row("Win Rate", win_rate)
    account_table.add_row("Trade Rate", trade_rate)
    account_table.add_row("Price", f"${price_val:,.2f}" if price_val else "—")
    account_table.add_row("Cooldowns", str(dict(data.get("cooldown", {}))))
    
    # ── Positions Panel ──
    pos_table = Table(show_header=True, box=box.SIMPLE, padding=(0,1))
    pos_table.add_column("Side", style="yellow")
    pos_table.add_column("Size", justify="right")
    pos_table.add_column("Entry", justify="right")
    pos_table.add_column("Age", justify="right")
    pos_table.add_column("Unreal.", justify="right")
    
    if positions:
        for pos in positions:
            side = pos["side"].upper()
            size = f"{pos['size']:.4f}"
            entry = f"${pos['entry_price']:,.0f}"
            age_min = (now - datetime.fromisoformat(pos["timestamp"])).total_seconds() / 60
            age = f"{age_min:.0f}m"
            
            if price_val:
                if side == "BUY":
                    ur = (price_val - pos["entry_price"]) / pos["entry_price"] * 100
                else:
                    ur = (pos["entry_price"] - price_val) / pos["entry_price"] * 100
                ur_str = Text(f"{ur:+.2f}%", style="green" if ur >= 0 else "red")
            else:
                ur_str = "—"
            
            pos_table.add_row(side, size, entry, age, ur_str)
    else:
        pos_table.add_row("—", "—", "—", "—", "—")
    
    # ── Trade History Panel ──
    trade_table = Table(show_header=True, box=box.SIMPLE, padding=(0,1), header_style="bold")
    trade_table.add_column("Time", width=11)
    trade_table.add_column("Side", width=5)
    trade_table.add_column("Size", width=8, justify="right")
    trade_table.add_column("Exit", width=8)
    trade_table.add_column("Reason", width=9)
    trade_table.add_column("PnL", width=10, justify="right")
    
    if trades_list:
        for t in reversed(trades_list[-15:]):
            ts = t.get("timestamp", "")[5:16]
            side = t.get("side", "?").upper()
            size = f"{t.get('size_btc', 0):.4f}"
            exit_p = f"${t.get('exit_price', 0):,.0f}"
            reason = t.get("close_reason", "?")
            pnl = t.get("pnl_usdt", 0)
            pnl_str = Text(f"${pnl:+,.2f}", style="green" if pnl >= 0 else "red")
            trade_table.add_row(ts, side, size, exit_p, reason, pnl_str)
    else:
        trade_table.add_row("—", "—", "—", "—", "—", "—")
    
    # ── Watcher Log Panel ──
    log_text = "\n".join(watcher_log) if watcher_log else "Waiting for activity..."
    
    # ── Layout ──
    title = Text(" KRONOS HFT DASHBOARD ", style="bold white on blue")
    subtitle = Text(f" Live · {now.strftime('%H:%M:%S')} · Refresh every 3s ", style="dim")
    
    header = Table.grid(padding=0)
    header.add_column()
    header.add_row(title)
    header.add_row(subtitle)
    
    layout = Table.grid(padding=1)
    layout.add_column()
    layout.add_row(header)
    layout.add_row(Panel(account_table, title="📊 Account", border_style="cyan"))
    layout.add_row(Panel(pos_table, title="📈 Open Positions", border_style="yellow"))
    layout.add_row(Panel(trade_table, title="📜 Recent Trades", border_style="green"))
    layout.add_row(Panel(log_text, title="⚡ Activity Log", border_style="dim", height=6))
    
    return layout

# ── Main ──

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Kronos HFT Live Dashboard")
    parser.add_argument("--watch", action="store_true", help="Run TP/SL watcher in background")
    parser.add_argument("--run", action="store_true", help="Run full scalper + watcher")
    args = parser.parse_args()
    
    watcher = Watcher()
    if args.watch:
        watcher.start(do_scalp=False)
        watcher.log.append(f"[{time.strftime('%H:%M:%S')}] TP/SL watcher started (60s interval)")
    if args.run:
        watcher.start(do_scalp=True)
        watcher.log.append(f"[{time.strftime('%H:%M:%S')}] Full scalper + watcher started")
    
    console.clear()
    console.print("[bold green]Kronos HFT Dashboard[/bold green] — press Ctrl+C to stop")
    
    try:
        with Live(refresh_per_second=0.33, screen=False) as live:
            while True:
                data = load()
                price = get_btc_price()
                dashboard = build_dashboard(data, price, list(watcher.log))
                live.update(dashboard)
                time.sleep(3)
    except KeyboardInterrupt:
        watcher.running = False
        console.print("\n[yellow]Dashboard stopped.[/yellow]")

if __name__ == "__main__":
    main()
