#!/usr/bin/env python3
"""
kronos_live_cron_tuned.py — EV Fusion Tuned.
Asymmetric confidence thresholds: BUY 0.4 / SELL 0.2.
Scale-out exits, TimesFM ensemble. Separate ledger: paper_trades_ev_tuned.json
"""
import sys, os, json, time
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

PROJECT_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_DIR))
import ccxt, torch
from model import Kronos, KronosTokenizer, KronosPredictor
from quant_models.fusion_layer import QuantFusionEngine

# ── CONFIG (asymmetric — tuned for HFT) ──
LOOKBACK = 200
PRED_LEN = 4
SELECTOR_SAMPLES = 20
CAPITAL = 1000.0
LEVERAGE = 200
BASELINE_SIZE = 0.5
TP_PCT = 0.3
SL_PCT = 0.2
CONF_THRESHOLD_BUY = 0.4    # Higher bar for BUY (53% natural accuracy)
CONF_THRESHOLD_SELL = 0.2   # Lower bar for SELL (59% natural accuracy)

LEDGER = PROJECT_DIR / "paper_trades_ev_tuned.json"

# ── ProxyMan hermes3 ──
os.environ["HTTP_PROXY"] = "http://127.0.0.1:8091"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:8091"
os.environ["NO_PROXY"] = "localhost,127.0.0.1,huggingface.co,huggingface.com,.hf.co"

def main():
    print(f"═══ EV FUSION TUNED — {datetime.now().strftime('%Y-%m-%d %H:%M')} ═══")

    # ── 1. DATA ──
    print("\n[1] Fetching BTC 5m...")
    ex = ccxt.binance({'options': {'defaultType': 'spot'}, 'timeout': 30000})
    ex.load_markets()
    ohlcv = ex.fetch_ohlcv('BTC/USDT', '5m', limit=LOOKBACK + 50)
    df = pd.DataFrame(ohlcv, columns=['ts', 'o', 'h', 'l', 'c', 'v'])
    df['t'] = pd.to_datetime(df['ts'], unit='ms')
    df_eng = df.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'})
    current_price = float(df['c'].iloc[-1])
    print(f"  BTC: ${current_price:.2f}")

    # ── 2. MODELS ──
    print("\n[2] Loading models...")
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    tok = KronosTokenizer.from_pretrained('NeoQuasar/Kronos-Tokenizer-base').to(device)
    model = Kronos.from_pretrained('NeoQuasar/Kronos-base').to(device)
    predictor = KronosPredictor(model, tok, max_context=512, device=device)
    print(f"  ✅ Kronos on {device}")

    # TimesFM
    try:
        import timesfm as _tf
        torch.set_float32_matmul_precision('high')
        tf_model = _tf.TimesFM_2p5_200M_torch.from_pretrained('google/timesfm-2.5-200m-pytorch')
        tf_model.compile(_tf.ForecastConfig(max_context=512, max_horizon=64,
            normalize_inputs=True, use_continuous_quantile_head=True,
            force_flip_invariance=True, infer_is_positive=True, fix_quantile_crossing=True))
        HAS_TF = True
        print(f"  ✅ TimesFM 2.5 200M")
    except Exception as e:
        HAS_TF = False
        print(f"  ⚠ TimesFM: {e}")

    # Fusion
    qf = QuantFusionEngine(capital=CAPITAL, leverage=LEVERAGE, base_tp_pct=TP_PCT, base_sl_pct=SL_PCT)
    qf.warmup(df_eng)
    qf.initialize(btc_price=current_price)

    # Selector
    sel_path = PROJECT_DIR / 'quant_models' / 'models' / 'kronos_selector.xgb'
    if sel_path.exists():
        try:
            qf.load_selector(str(sel_path), predictor=predictor, samples=SELECTOR_SAMPLES)
            print(f"  ✅ Selector ({SELECTOR_SAMPLES} samples)")
        except Exception as e:
            print(f"  ⚠ Selector: {e}")
    print("  ✅ Fusion ready")

    # ── 3. KRONOS ──
    print(f"\n[3] Running Kronos...")
    ctx = df.iloc[-LOOKBACK:]
    ctx_eng = df_eng.iloc[-LOOKBACK:]
    net = 0; direction = "NEUTRAL"; kronos_conf = 0
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
            if net > 0.1: direction = "BULLISH"
            elif net < -0.1: direction = "BEARISH"
            kronos_conf = min(abs(net) / 0.3, 1.0) if abs(net) > 0.04 else 0.0
            print(f"  Net: {net:+.4f}% | Dir: {direction} | Conf: {kronos_conf:.3f}")
        except Exception as e:
            print(f"  ❌ Kronos failed: {e}")
            return

    # ── 3b. TIMESFM ──
    tf_dir, tf_conf = "NEUTRAL", 0.0
    if HAS_TF:
        try:
            _prices = np.array(ctx['c'].values, dtype=np.float64)
            _point, _q = tf_model.forecast(horizon=12, inputs=[_prices])
            _fe = float(_point[0, -1])
            _ch = (_fe - current_price) / current_price * 100
            _uc = (_q[0, -1, 9] - _q[0, -1, 1]) / _fe * 100
            if _uc <= 1.5:
                if _ch > 0.05: tf_dir, tf_conf = "BULLISH", min(abs(_ch)/0.3, 1.0)
                elif _ch < -0.05: tf_dir, tf_conf = "BEARISH", min(abs(_ch)/0.3, 1.0)
            print(f"  TimesFM: {tf_dir} ({_ch:+.3f}%, CI {_uc:.1f}%)")
        except Exception as e:
            print(f"  ⚠ TimesFM: {e}")

    # ── 4. SELECTOR + FUSION ──
    selector_result = {}
    if qf.selector and qf.selector.is_trained:
        try:
            selector_result = qf.run_selector(ctx)
            if selector_result and selector_result.get('decision'):
                print(f"  Selector: {selector_result['decision']} (conf={selector_result.get('confidence_adjusted',0):.3f})")
        except: pass

    if abs(net) >= 0.08:
        base_dir = "SELL" if net > 0.10 else "BUY" if net < -0.10 else "HOLD"
        base_conf = min(abs(net) / 0.3, 1.0) if base_dir != "HOLD" else 0.0
    else:
        base_dir, base_conf = "HOLD", 0.0

    fusion_result = qf.analyze(
        df=ctx_eng, kronos_direction=direction, kronos_net_pct=net,
        kronos_confidence=kronos_conf, current_price=current_price,
        strategy_signals={
            'antitrend': (1 if base_dir == 'BUY' else -1 if base_dir == 'SELL' else 0, base_conf),
            'kronos_raw': (1 if net > 0.04 else -1 if net < -0.04 else 0, kronos_conf),
            'timesfm': (1 if tf_dir == 'BULLISH' else -1 if tf_dir == 'BEARISH' else 0, tf_conf),
        },
        selector_result=selector_result,
    )

    print(f"  Decision: {fusion_result['decision']}")
    print(f"  Confidence: {fusion_result['confidence']:.3f}")

    # ── 5. EXECUTE (asymmetric thresholds) ──
    try:
        with open(LEDGER) as f:
            curr = json.load(f)
        open_pos = len(curr.get("positions", []))
    except:
        open_pos = 0

    if open_pos > 0:
        print(f"\n[5] SKIP — {open_pos} position(s) open")
    else:
        decision = fusion_result['decision']
        confidence = fusion_result['confidence']
        side = None
        if decision == 'BUY' and confidence > CONF_THRESHOLD_BUY:
            side = 'buy'
        elif decision == 'SELL' and confidence > CONF_THRESHOLD_SELL:
            side = 'sell'

        if side:
            size_btc = fusion_result['size']['size_btc']
            tp_pct = fusion_result['tp_sl']['final_tp_pct']
            sl_pct = fusion_result['tp_sl']['final_sl_pct']
            print(f"\n[5] EXECUTING {side.upper()} {size_btc:.4f} BTC | TP: {tp_pct:.2f}% SL: {sl_pct:.2f}%")
            import subprocess
            exec_script = PROJECT_DIR / "kronos_exec.py"
            venv_python = PROJECT_DIR / ".venv" / "Scripts" / "python.exe"
            cmd = [str(venv_python), str(exec_script), "--scalp", "--paper", side,
                   "--size", str(size_btc), "--ledger", str(LEDGER)]
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30,
                                        creationflags=0x08000000 if os.name == 'nt' else 0)
                print(f"  Output: {result.stdout.strip()}")
                if result.stderr: print(f"  Stderr: {result.stderr.strip()}")
            except Exception as e:
                print(f"  ❌ Execution: {e}")
        else:
            print(f"\n[5] No trade — HOLD (BUY needs >{CONF_THRESHOLD_BUY}, SELL needs >{CONF_THRESHOLD_SELL})")

    print(f"\n{qf.summary()}")
    print(f"\n═══ DONE ═══")

    # Save analysis
    try:
        analysis = {
            "timestamp": datetime.now().isoformat(),
            "btc_price": current_price, "decision": fusion_result.get("decision"),
            "confidence": fusion_result.get("confidence"),
            "source": fusion_result.get("optimizations", {}).get("decision_source", "?"),
            "conf_thresholds": {"buy": CONF_THRESHOLD_BUY, "sell": CONF_THRESHOLD_SELL},
            "regime": {"hmm_label": fusion_result.get("regime",{}).get("hmm_label","?"),
                       "hurst_H": fusion_result.get("regime",{}).get("hurst_H"),
                       "hurst_label": fusion_result.get("regime",{}).get("hurst_label","?")},
            "tp_sl": {"final_tp_pct": fusion_result.get("tp_sl",{}).get("final_tp_pct"),
                      "final_sl_pct": fusion_result.get("tp_sl",{}).get("final_sl_pct"),
                      "atr_pct": fusion_result.get("tp_sl",{}).get("atr_pct")},
            "indicators": {"rsi": fusion_result.get("quant_details",{}).get("rsi"),
                           "atr": fusion_result.get("quant_details",{}).get("atr_pct")},
            "timesfm": {"direction": tf_dir, "confidence": tf_conf},
        }
        with open(PROJECT_DIR / "ev_tuned_analysis.json", "w") as f:
            class Encoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, (np.floating, np.integer)): return float(obj) if isinstance(obj, np.floating) else int(obj)
                    return super().default(obj)
            json.dump(analysis, f, indent=2, cls=Encoder)
    except Exception as e:
        print(f"  ⚠ Save analysis: {e}")

if __name__ == "__main__":
    main()
