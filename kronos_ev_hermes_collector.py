#!/usr/bin/env python3
"""
kronos_ev_hermes_collector.py — Signal collector for EV Fusion + Hermes Agent.
Generates signal file for Hermes to review. No auto-execution.
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

SIGNAL_FILE = PROJECT_DIR / "ev_hermes_signal.json"
LEDGER = PROJECT_DIR / "paper_trades_ev_hermes.json"

os.environ["HTTP_PROXY"] = "http://127.0.0.1:8091"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:8091"
os.environ["NO_PROXY"] = "localhost,127.0.0.1,huggingface.co,huggingface.com,.hf.co"

def main():
    print(f"═══ EV HERMES COLLECTOR — {datetime.now().strftime('%Y-%m-%d %H:%M')} ═══")

    # ── 1. DATA ──
    ex = ccxt.binance({'options':{'defaultType':'spot'},'timeout':30000})
    ex.load_markets()
    ohlcv = ex.fetch_ohlcv('BTC/USDT','5m',limit=250)
    df = pd.DataFrame(ohlcv,columns=['ts','o','h','l','c','v'])
    df['t'] = pd.to_datetime(df['ts'],unit='ms')
    df_eng = df.rename(columns={'o':'open','h':'high','l':'low','c':'close','v':'volume'})
    current_price = float(df['c'].iloc[-1])
    print(f"  BTC: ${current_price:.2f}")

    # ── 2. MODELS ──
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    tok = KronosTokenizer.from_pretrained('NeoQuasar/Kronos-Tokenizer-base').to(device)
    model = Kronos.from_pretrained('NeoQuasar/Kronos-base').to(device)
    predictor = KronosPredictor(model,tok,max_context=512,device=device)
    print(f"  ✅ Kronos on {device}")

    try:
        import timesfm as _tf
        torch.set_float32_matmul_precision('high')
        tf_model = _tf.TimesFM_2p5_200M_torch.from_pretrained('google/timesfm-2.5-200m-pytorch')
        tf_model.compile(_tf.ForecastConfig(max_context=512,max_horizon=64,normalize_inputs=True,
            use_continuous_quantile_head=True,force_flip_invariance=True,infer_is_positive=True,fix_quantile_crossing=True))
        HAS_TF=True; print(f"  ✅ TimesFM")
    except: HAS_TF=False

    qf = QuantFusionEngine(capital=1000.0,leverage=200,base_tp_pct=0.3,base_sl_pct=0.2)
    qf.warmup(df_eng); qf.initialize(btc_price=current_price)
    sel_path = PROJECT_DIR/'quant_models'/'models'/'kronos_selector.xgb'
    if sel_path.exists():
        try: qf.load_selector(str(sel_path),predictor=predictor,samples=20)
        except: pass
    print("  ✅ Ready")

    # ── 3. SIGNALS ──
    ctx = df.iloc[-200:]; ctx_eng = df_eng.iloc[-200:]
    with torch.no_grad():
        try:
            x = ctx[['o','h','l','c','v']].copy()
            x.columns=['open','high','low','close','volume']; x['amount']=0.0
            ts = ctx['t'].reset_index(drop=True)
            yt = pd.Series(pd.date_range(start=ts.iloc[-1]+pd.Timedelta(minutes=5),periods=4,freq='5min'))
            p = predictor.predict(df=x,x_timestamp=ts,y_timestamp=yt,pred_len=4,T=1.0,top_p=0.9,sample_count=5,verbose=False)
            pc = float(p['close'].iloc[-1])
            net = ((pc-current_price)/current_price)*100
            direction = "BULLISH" if net>0.1 else "BEARISH" if net<-0.1 else "NEUTRAL"
            kronos_conf = min(abs(net)/0.3,1.0) if abs(net)>0.04 else 0.0
            print(f"  Kronos: {direction} ({net:+.4f}%, conf={kronos_conf:.3f})")
        except Exception as e:
            print(f"  ❌ Kronos: {e}"); return

    tf_dir,tf_conf="NEUTRAL",0.0; tf_uc=0
    if HAS_TF:
        try:
            _p=np.array(ctx['c'].values,dtype=np.float64)
            _pt,_q=tf_model.forecast(horizon=8,inputs=[_p])
            _fe=float(_pt[0,-1]); _ch=(_fe-current_price)/current_price*100
            tf_uc=(_q[0,-1,9]-_q[0,-1,1])/_fe*100
            if tf_uc<=1.5:
                if _ch>0.05: tf_dir,tf_conf='BULLISH',min(abs(_ch)/0.3,1.0)
                elif _ch<-0.05: tf_dir,tf_conf='BEARISH',min(abs(_ch)/0.3,1.0)
            print(f"  TimesFM: {tf_dir} ({_ch:+.3f}%, CI {tf_uc:.1f}%)")
        except: pass

    selector_result = {}
    if qf.selector and qf.selector.is_trained:
        try:
            selector_result = qf.run_selector(ctx)
            if selector_result and selector_result.get('decision'):
                print(f"  Selector: {selector_result['decision']}")
        except: pass

    if abs(net)>=0.08:
        base_dir = "SELL" if net>0.10 else "BUY" if net<-0.10 else "HOLD"
        base_conf = min(abs(net)/0.3,1.0) if base_dir!="HOLD" else 0.0
    else: base_dir,base_conf="HOLD",0.0

    fusion_result = qf.analyze(df=ctx_eng,kronos_direction=direction,kronos_net_pct=net,
        kronos_confidence=kronos_conf,current_price=current_price,
        strategy_signals={'antitrend':(1 if base_dir=='BUY' else -1 if base_dir=='SELL' else 0,base_conf),
                          'kronos_raw':(1 if net>0.04 else -1 if net<-0.04 else 0,kronos_conf),
                          'timesfm':(1 if tf_dir=='BULLISH' else -1 if tf_dir=='BEARISH' else 0,tf_conf)},
        selector_result=selector_result)

    # ── 4. SAVE SIGNAL ──
    signal = {
        "timestamp": datetime.now().isoformat(),
        "btc_price": current_price,
        "kronos": {"direction": direction, "confidence": round(kronos_conf,3), "net_pct": round(net,4)},
        "timesfm": {"direction": tf_dir, "confidence": round(tf_conf,3), "ci_width": round(tf_uc,2)},
        "selector": selector_result,
        "fusion": {
            "decision": fusion_result.get("decision"),
            "confidence": round(fusion_result.get("confidence",0),3),
            "source": fusion_result.get("optimizations",{}).get("decision_source","?"),
        },
        "regime": {
            "hmm": fusion_result.get("regime",{}).get("hmm_label","?"),
            "hurst": fusion_result.get("regime",{}).get("hurst_H"),
        },
        "indicators": {
            "rsi": fusion_result.get("quant_details",{}).get("rsi"),
            "atr": fusion_result.get("quant_details",{}).get("atr_pct"),
        },
        "tp_sl": {
            "tp_pct": fusion_result.get("tp_sl",{}).get("final_tp_pct"),
            "sl_pct": fusion_result.get("tp_sl",{}).get("final_sl_pct"),
        },
        "size_btc": fusion_result.get("size",{}).get("size_btc",0.5),
        "already_in_position": False,
    }

    try:
        with open(LEDGER) as f: led = json.load(f)
        if len(led.get("positions",[])) > 0: signal["already_in_position"] = True
    except: pass

    with open(SIGNAL_FILE,"w") as f:
        class Encoder(json.JSONEncoder):
            def default(self,obj):
                if isinstance(obj,(np.floating,np.integer)): return float(obj) if isinstance(obj,np.floating) else int(obj)
                return super().default(obj)
        json.dump(signal,f,indent=2,cls=Encoder)

    print(f"\n  Signal: {fusion_result.get('decision')} (conf={fusion_result.get('confidence',0):.3f})")
    print(f"  Needs Hermes review: {'YES' if fusion_result.get('confidence',0) > 0.15 else 'NO'}")
    print(f"  Saved to {SIGNAL_FILE.name}")
    print(f"  Position open: {signal['already_in_position']}")
    print(f"\n═══ DONE ═══")

if __name__ == "__main__":
    main()
