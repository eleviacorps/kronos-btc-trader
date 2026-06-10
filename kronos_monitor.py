#!/usr/bin/env python3
"""kronos_monitor.py — Dual-account monitor, box-drawing layout, 2s refresh."""
import json, os, sys, time, subprocess
from pathlib import Path
from datetime import datetime

PROJECT_DIR = Path(r"D:\Programming\AiProjects\Kronos(reviewing)\Kronos")
VENV_PYTHON = PROJECT_DIR / ".venv" / "Scripts" / "python.exe"
EXEC_SCRIPT = PROJECT_DIR / "kronos_exec.py"
BOT_LEDGER = "paper_trades.json"
AGENT_LEDGER = "paper_trades_agent.json"
os.chdir(PROJECT_DIR)

# Inner width of each column (excluding borders/separator)
CW = 30

def run_exec(args, timeout=20):
    cmd = [str(VENV_PYTHON), str(EXEC_SCRIPT)] + args
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=PROJECT_DIR)
        return r.stdout + r.stderr
    except subprocess.TimeoutExpired:
        return "[TIMEOUT]"

def parse_status(output):
    info = {"balance":0,"total_value":0,"btc_price":0,"positions":0,"trades":0}
    for line in output.split("\n"):
        line = line.strip()
        if "BTC/USDT:" in line:
            try: info["btc_price"] = float(line.split("$")[1].replace(",",""))
            except: pass
        elif "Balance:" in line:
            try: info["balance"] = float(line.split("$")[1].replace(",",""))
            except: pass
        elif "Total Value:" in line:
            try: info["total_value"] = float(line.split("$")[1].replace(",",""))
            except: pass
        elif "Open Positions" in line:
            try: info["positions"] = int(line.split("(")[1].split(")")[0])
            except: pass
        elif "total):" in line:
            try: info["trades"] = int(line.split("(")[1].split(" ")[0])
            except: pass
    return info

def parse_ledger(name):
    p = PROJECT_DIR / name
    if p.exists():
        try: return json.load(open(p))
        except: pass
    return {"trades":[], "positions":[], "initial_capital":1000.0}

def fmt_pnl(v):
    if v > 0: return f"+${v:.2f}"
    if v < 0: return f"-${abs(v):.2f}"
    return "$0.00"

def cell(label, val, w=CW):
    """Format a cell: '  Label: VALUE' padded to width w."""
    content = f"  {label}: {val}"
    return content.ljust(w)

def monitor_loop(interval=2.0):
    sys.stdout.write("\033[?25l")
    first = True
    try:
        while True:
            t0 = time.time()
            b_out = run_exec(["--status","--ledger",BOT_LEDGER], timeout=15)
            a_out = run_exec(["--status","--ledger",AGENT_LEDGER], timeout=15)
            bi = parse_status(b_out); ai = parse_status(a_out)
            bd = parse_ledger(BOT_LEDGER); ad = parse_ledger(AGENT_LEDGER)
            b_init = bd.get("initial_capital",1000.0); a_init = ad.get("initial_capital",1000.0)
            bp = bi["total_value"] - b_init; ap = ai["total_value"] - a_init

            bt = bd.get("trades",[]); at = ad.get("trades",[])
            bw = sum(1 for t in bt if t.get("pnl_usdt",0)>0)
            bl = sum(1 for t in bt if t.get("pnl_usdt",0)<=0)
            aw = sum(1 for t in at if t.get("pnl_usdt",0)>0)
            al = sum(1 for t in at if t.get("pnl_usdt",0)<=0)
            btc = bw+bl; atc = aw+al

            now = datetime.now().strftime("%H:%M:%S")
            price = bi.get("btc_price",0)

            # Full width = left_border(1) + col(CW) + sep(3) + col(CW) + right_border(1)
            # = 1 + 30 + 3 + 30 + 1 = 65
            FW = 1 + CW + 3 + CW + 1
            S = " \u2551 "  # column separator
            L = "\u2551"    # left border
            R = "\u2551"    # right border

            def row(left_cell, right_cell):
                return L + left_cell.ljust(CW) + S + right_cell.ljust(CW) + R

            lines = []
            lines.append("\u2554" + "\u2550"*(FW-2) + "\u2557")
            lines.append(L + f"  KRONOS DUAL MONITOR  {now} ".ljust(FW-2) + R)
            lines.append(L + f"  BTC/USDT: \u0024{price:>,.2f}".ljust(FW-2) + R)
            lines.append("\u2560" + "\u2550"*CW + "\u256c" + "\u2550"*CW + "\u2563")
            lines.append(L + " BOT-ONLY".center(CW) + S + "AGENT".center(CW) + R)
            lines.append(L + "\u2500"*CW + S + "\u2500"*CW + R)

            lines.append(row(
                f"  Balance: \u0024{bi['balance']:>8,.2f}",
                f"  Balance: \u0024{ai['balance']:>8,.2f}"
            ))
            lines.append(row(
                f"  Value:   \u0024{bi['total_value']:>8,.2f}",
                f"  Value:   \u0024{ai['total_value']:>8,.2f}"
            ))

            bp_ind = "\U0001f7e2" if bp>=0 else "\U0001f534"
            ap_ind = "\U0001f7e2" if ap>=0 else "\U0001f534"
            lines.append(row(
                f"  PnL:     {bp_ind} {fmt_pnl(bp)}",
                f"  PnL:     {ap_ind} {fmt_pnl(ap)}"
            ))

            lines.append(row(
                f"  Trades:  {bi['trades']} closed",
                f"  Trades:  {ai['trades']} closed"
            ))

            bwr = f"{bw}/{btc} ({bw*100//max(btc,1)}%)" if btc else "--"
            awr = f"{aw}/{atc} ({aw*100//max(atc,1)}%)" if atc else "--"
            lines.append(row(f"  W/L:     {bwr}", f"  W/L:     {awr}"))

            # Position detail
            bpos_detail = f"{bi['positions']} pos"
            apos_detail = f"{ai['positions']} pos"
            if bi['positions'] > 0:
                sides = [p["side"].upper()[:1] for p in bd.get("positions",[])]
                bpos_detail += " " + ",".join(sides)
            if ai['positions'] > 0:
                sides = [p["side"].upper()[:1] for p in ad.get("positions",[])]
                apos_detail += " " + ",".join(sides)
            lines.append(row(f"  Open:    {bpos_detail}", f"  Open:    {apos_detail}"))

            lines.append("\u2560" + "\u2550"*CW + "\u2569" + "\u2550"*CW + "\u2563")
            ft = time.time()-t0
            foot = f" Fetch:{ft:.1f}s Refr:{interval}s | close bot/agent N | Ctrl+C "
            lines.append(L + foot.center(FW-2) + R)
            lines.append("\u255a" + "\u2550"*(FW-2) + "\u255d")

            out = "\n".join(lines)
            if first:
                print(out); first = False
            else:
                sys.stdout.write("\033[" + str(len(lines)) + "A")
                sys.stdout.write(out)
            sys.stdout.flush()
            time.sleep(interval)
    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write("\033[?25h")
        print("\nStopped.")

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--close", nargs="*", default=None)
    p.add_argument("--interval", type=float, default=2.0)
    args = p.parse_args()
    if args.close is not None:
        if not args.close: print("Use: --close bot|agent [index]"); return
        acct = args.close[0].lower()
        ledger = {"bot":BOT_LEDGER,"agent":AGENT_LEDGER}.get(acct)
        if not ledger: print(f"Unknown '{acct}'"); return
        if len(args.close)>=2 and args.close[1].isdigit():
            print(run_exec(["--close-trade",args.close[1],"--ledger",ledger]))
        else:
            print(run_exec(["--paper","close","--ledger",ledger]))
        return
    monitor_loop(args.interval)

if __name__ == "__main__":
    main()
