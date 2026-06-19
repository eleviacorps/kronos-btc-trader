#!/usr/bin/env python3
"""
kronos_live_cron.py — Live trading cron wrapper for Kronos + Selector + Fusion.

Runs every 10 minutes via Hermes cron:
  1. Fetch latest BTC 5m candles from Binance
  2. Run Kronos inference (20 samples)
  3. Sample selector picks best sample
  4. Fusion engine makes final decision
  5. Execute trade via kronos_exec.py if BUY with confidence > 0.3
  6. Log everything to stdout for Hermes cron delivery

Usage (direct):
  python kronos_live_cron.py

Usage (cron):
  hermes cron create kronos-live --schedule "*/10 * * * *" \
    --script kronos_live_cron.py --profile tor-cron
"""

import sys, os, json, time, argparse
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

PROJECT_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_DIR))

# ── Imports ──
import ccxt
import torch
from model import Kronos, KronosTokenizer, KronosPredictor
from quant_models.fusion_layer import QuantFusionEngine

# =========================================================================
# CONFIG
# =========================================================================
LOOKBACK = 200
PRED_LEN = 4
SELECTOR_SAMPLES = 20
CAPITAL = 1000.0
LEVERAGE = 200
BASELINE_SIZE = 0.5
TP_PCT = 0.3
SL_PCT = 0.2

LEDGER = PROJECT_DIR / "paper_trades_live.json"

# =========================================================================
# MAIN
# =========================================================================
def main():
    print(f"═══ KRONOS LIVE CRON — {datetime.now().strftime('%Y-%m-%d %H:%M')} ═══")

    # ── 1. FETCH DATA ──
    print("\n[1] Fetching BTC 5m...")
    ex = ccxt.binance({'options': {'defaultType': 'spot'}, 'timeout': 30000})
    ex.load_markets()
    ohlcv = ex.fetch_ohlcv('BTC/USDT', '5m', limit=LOOKBACK + 50)
    df = pd.DataFrame(ohlcv, columns=['ts', 'o', 'h', 'l', 'c', 'v'])
    df['t'] = pd.to_datetime(df['ts'], unit='ms')
    df_eng = df.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'})
    current_price = float(df['c'].iloc[-1])
    print(f"  BTC: ${current_price:.2f}  |  {len(df)} candles")

    # ── 2. LOAD MODELS ──
    print("\n[2] Loading models...")
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # Kronos
    tok = KronosTokenizer.from_pretrained('NeoQuasar/Kronos-Tokenizer-base').to(device)
    model = Kronos.from_pretrained('NeoQuasar/Kronos-base').to(device)
    predictor = KronosPredictor(model, tok, max_context=512, device=device)
    print(f"  ✅ Kronos on {device}")

    # Fusion engine
    qf = QuantFusionEngine(capital=CAPITAL, leverage=LEVERAGE, base_tp_pct=TP_PCT, base_sl_pct=SL_PCT)
    qf.warmup(df_eng)
    qf.initialize(btc_price=current_price)

    # 1h EMA50 for HTF bias
    df_1h = df[['t', 'c']].copy().set_index('t').resample('1h').last().dropna().reset_index()
    df_1h['ema50'] = df_1h['c'].ewm(span=50, adjust=False).mean()
    qf.set_htf_bias(ema50=float(df_1h['ema50'].iloc[-1]), current_price=current_price)

    # Selector
    sel_path = PROJECT_DIR / 'quant_models' / 'models' / 'kronos_selector.xgb'
    if sel_path.exists():
        try:
            qf.load_selector(str(sel_path), predictor=predictor, samples=SELECTOR_SAMPLES)
            print(f"  ✅ Selector ({SELECTOR_SAMPLES} samples)")
        except Exception as e:
            print(f"  ⚠ Selector failed: {e}")
    else:
        print(f"  ⚠ No selector model at {sel_path}")
    print("  ✅ Fusion ready")

    # ── 3. KRONOS PREDICTION ──
    print(f"\n[3] Running Kronos...")
    ctx = df.iloc[-LOOKBACK:]
    ctx_eng = df_eng.iloc[-LOOKBACK:]

    with torch.no_grad():
        try:
            x = ctx[['o', 'h', 'l', 'c', 'v']].copy()
            x.columns = ['open', 'high', 'low', 'close', 'volume']
            x['amount'] = 0.0
            ts = ctx['t'].reset_index(drop=True)
            yt = pd.Series(pd.date_range(start=ts.iloc[-1] + pd.Timedelta(minutes=5), periods=PRED_LEN, freq='5min'))

            p = predictor.predict(df=x, x_timestamp=ts, y_timestamp=yt,
                                  pred_len=PRED_LEN, T=1.0, top_p=0.9, sample_count=5, verbose=False)
            pc = float(p['close'].iloc[-1])
            net = ((pc - current_price) / current_price) * 100
            rng = ((p['high'].max() - p['low'].min()) / current_price) * 100

            if net > 0.1:
                direction = "BULLISH"
            elif net < -0.1:
                direction = "BEARISH"
            else:
                direction = "NEUTRAL"

            kronos_conf = min(abs(net) / 0.3, 1.0) if abs(net) > 0.04 else 0.0
            print(f"  Net: {net:+.4f}%  |  Range: {rng:.4f}%  |  Dir: {direction}  |  Conf: {kronos_conf:.3f}")
        except Exception as e:
            print(f"  ❌ Kronos failed: {e}")
            return

    # ── 4. SELECTOR + FUSION ──
    print(f"\n[4] Making decision...")
    selector_result = qf.run_selector(ctx) if qf.selector and qf.selector.is_trained else {}
    # BUY-only from selector
    if selector_result and selector_result.get('decision') == 'SELL':
        selector_result['decision'] = 'HOLD'

    # Compute antitrend baseline for BMA
    if net is not None and rng >= 0.08:
        if net > 0.10:
            base_dir, base_conf = "SELL", min(abs(net) / 0.3, 1.0)
        elif net < -0.10:
            base_dir, base_conf = "BUY", min(abs(net) / 0.3, 1.0)
        else:
            base_dir, base_conf = "HOLD", 0.0
    else:
        base_dir, base_conf = "HOLD", 0.0

    fusion_result = qf.analyze(
        df=ctx_eng,
        kronos_direction=direction,
        kronos_net_pct=net,
        kronos_confidence=kronos_conf,
        current_price=current_price,
        strategy_signals={
            'antitrend': (1 if base_dir == 'BUY' else -1 if base_dir == 'SELL' else 0, base_conf),
            'kronos_raw': (1 if net > 0.04 else -1 if net < -0.04 else 0, kronos_conf),
        },
        selector_result=selector_result,
    )

    print(f"  Decision: {fusion_result['decision']}")
    print(f"  Confidence: {fusion_result['confidence']:.3f}")
    print(f"  Source: {fusion_result.get('optimizations', {}).get('decision_source', '?')}")

    # ── 5. EXECUTE ──
    if fusion_result['decision'] in ('BUY', 'SELL') and fusion_result['confidence'] > 0.3:
        side = fusion_result['decision'].lower()
        size_btc = fusion_result['size']['size_btc']
        tp_pct = fusion_result['tp_sl']['final_tp_pct']
        sl_pct = fusion_result['tp_sl']['final_sl_pct']

        print(f"\n[5] EXECUTING {side.upper()} {size_btc:.4f} BTC  |  TP: {tp_pct:.2f}%  SL: {sl_pct:.2f}%")

        # Execute via kronos_exec.py
        exec_script = PROJECT_DIR / "kronos_exec.py"
        venv_python = PROJECT_DIR / ".venv" / "Scripts" / "python.exe"
        cmd = [
            str(venv_python), str(exec_script),
            "--scalp", "--paper", side,
            "--size", str(size_btc),
            "--tp", str(tp_pct),
            "--sl", str(sl_pct),
            "--ledger", str(LEDGER),
        ]
        import subprocess
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30,
                                    creationflags=0x08000000 if os.name == 'nt' else 0)
            print(f"  Exit: {result.returncode}")
            if result.stdout:
                print(f"  Output: {result.stdout.strip()}")
            if result.stderr:
                print(f"  Stderr: {result.stderr.strip()}")
        except subprocess.TimeoutExpired:
            print("  ❌ Execution timed out")
        except Exception as e:
            print(f"  ❌ Execution failed: {e}")
    else:
        print(f"\n[5] No trade — HOLD")

    # ── 6. SUMMARY ──
    print(f"\n{qf.summary()}")
    print(f"\n═══ DONE ═══")

    # Save fusion analysis for dashboard
    try:
        analysis = {
            "timestamp": datetime.now().isoformat(),
            "btc_price": current_price,
            "decision": fusion_result.get("decision"),
            "confidence": fusion_result.get("confidence"),
            "source": fusion_result.get("optimizations", {}).get("decision_source", "?"),
            "regime": {
                "hmm_label": fusion_result.get("regime", {}).get("hmm_label", "?"),
                "hurst_H": fusion_result.get("regime", {}).get("hurst_H"),
                "hurst_label": fusion_result.get("regime", {}).get("hurst_label", "?"),
                "final_antitrend_mult": fusion_result.get("regime", {}).get("final_antitrend_mult"),
                "vol_ratio": fusion_result.get("regime", {}).get("vol_ratio"),
            },
            "tp_sl": {
                "final_tp_pct": fusion_result.get("tp_sl", {}).get("final_tp_pct"),
                "final_sl_pct": fusion_result.get("tp_sl", {}).get("final_sl_pct"),
                "atr_pct": fusion_result.get("tp_sl", {}).get("atr_pct"),
                "r_r": round(fusion_result.get("tp_sl", {}).get("final_tp_pct", 0) / max(fusion_result.get("tp_sl", {}).get("final_sl_pct", 0.01), 0.01), 2),
            },
            "size": {
                "size_btc": fusion_result.get("size", {}).get("size_btc"),
                "margin": fusion_result.get("size", {}).get("margin"),
                "details": fusion_result.get("size", {}).get("details", {}),
            },
            "indicators": {
                "rsi": fusion_result.get("quant_details", {}).get("rsi"),
                "atr": fusion_result.get("quant_details", {}).get("atr_pct"),
                "vol_ratio_to_avg": fusion_result.get("quant_details", {}).get("vol_ratio_to_avg"),
            },
            "selector": {
                "net_change": fusion_result.get("optimizations", {}).get("selector_net"),
                "best_prob": fusion_result.get("optimizations", {}).get("selector_best_prob"),
                "avg_prob": fusion_result.get("optimizations", {}).get("selector_avg_prob"),
            },
            "kronos": {
                "direction": direction,
                "net_pct": net,
                "range_pct": rng,
                "confidence": kronos_conf,
            },
        }
        with open(PROJECT_DIR / "fusion_analysis.json", "w") as f:
            class FloatEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, (np.floating, np.integer)):
                        return float(obj) if isinstance(obj, np.floating) else int(obj)
                    return super().default(obj)
            json.dump(analysis, f, indent=2, cls=FloatEncoder)
    except Exception as e:
        print(f"  ⚠ Failed to save analysis: {e}")


if __name__ == "__main__":
    main()
