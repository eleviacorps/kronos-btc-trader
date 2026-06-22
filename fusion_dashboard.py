#!/usr/bin/env python3
"""
fusion_dashboard.py — Lightweight live dashboard for the Kronos Fusion system.
Serves fusion_analysis.json data as a simple auto-refreshing HTML page.
"""
import json, os, sys
from pathlib import Path
from flask import Flask, jsonify

PROJECT_DIR = Path(r"D:\Programming\AiProjects\Kronos(reviewing)\Kronos")
FUSION_FILE = PROJECT_DIR / "fusion_analysis.json"
LEDGER_FILE = PROJECT_DIR / "paper_trades_live.json"

app = Flask(__name__)

def load_json(path):
    try:
        return json.loads(path.read_text()) if path.exists() else {}
    except Exception:
        return {}

@app.route("/")
def index():
    fusion = load_json(FUSION_FILE)
    ledger = load_json(LEDGER_FILE)
    pos = ledger.get("positions", [])
    trades = ledger.get("trades", [])

    rows = ""
    for k in ["timestamp", "btc_price", "decision", "confidence", "source"]:
        v = fusion.get(k, "?")
        rows += f"<tr><td>{k}</td><td>{v}</td></tr>"
    
    regime = fusion.get("regime", {})
    for k in ["hmm_label", "hurst_H", "hurst_label", "vol_ratio"]:
        v = regime.get(k, "?")
        rows += f"<tr><td>regime.{k}</td><td>{v}</td></tr>"
    
    tpsl = fusion.get("tp_sl", {})
    for k in ["final_tp_pct", "final_sl_pct", "atr_pct", "r_r"]:
        v = tpsl.get(k, "?")
        rows += f"<tr><td>tp_sl.{k}</td><td>{v}</td></tr>"

    indicators = fusion.get("indicators", {})
    for k in ["rsi", "atr", "vol_ratio_to_avg"]:
        v = indicators.get(k, "?")
        rows += f"<tr><td>ind.{k}</td><td>{v}</td></tr>"

    pos_html = ""
    for p in pos:
        pos_html += f"<tr><td>{p['side']}</td><td>{p['size']}</td><td>{p['entry_price']}</td><td>{p.get('tp_pct','?')}%</td><td>{p.get('sl_pct','?')}%</td></tr>"
    
    trades_html = ""
    for t in trades[-5:]:
        trades_html += f"<tr><td>{t.get('side','?')}</td><td>{t.get('entry_price','?')}</td><td>{t.get('exit_price','?')}</td><td>{t.get('pnl_usdt',0):+.2f}</td><td>{t.get('close_reason','?')}</td></tr>"

    return f"""<!DOCTYPE html>
<html><head><title>Kronos Fusion Live</title>
<meta http-equiv="refresh" content="5">
<style>
body {{ font-family: monospace; background: #111; color: #0f0; padding: 20px; }}
table {{ border-collapse: collapse; margin: 10px 0; width: 100%; }}
td, th {{ border: 1px solid #333; padding: 6px 12px; text-align: left; }}
th {{ background: #222; }}
.buy {{ color: #0f0; }} .sell {{ color: #f44; }} .hold {{ color: #ff0; }}
</style></head><body>
<h1>Kronos Fusion v2 — Live Dashboard</h1>
<div style="display:flex;gap:20px;flex-wrap:wrap;">
<div><h3>Fusion Analysis</h3><table>{rows}</table></div>
<div><h3>Ledger</h3><p>Balance: <b>{ledger.get('balance','?')}</b> | PnL: <b>{ledger.get('total_pnl','?')}</b></p></div>
</div>
<h3>Open Positions</h3>
<table><tr><th>Side</th><th>Size</th><th>Entry</th><th>TP</th><th>SL</th></tr>{pos_html or '<tr><td colspan=5>None</td></tr>'}</table>
<h3>Recent Trades</h3>
<table><tr><th>Side</th><th>Entry</th><th>Exit</th><th>PnL</th><th>Reason</th></tr>{trades_html or '<tr><td colspan=5>None</td></tr>'}</table>
<p><i>Auto-refreshes every 5s &middot; Last: {fusion.get("timestamp","?")}</i></p>
</body></html>"""

@app.route("/api/fusion")
def api_fusion():
    return jsonify(load_json(FUSION_FILE))

if __name__ == "__main__":
    print(f"Fusion dashboard starting on http://localhost:8099", flush=True)
    app.run(host="0.0.0.0", port=8099, debug=False)
