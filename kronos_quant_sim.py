"""
kronos_quant_sim.py — 10-day portfolio sim with Kronos + Quant Fusion (all 6 models).

Pipeline:
  1. Fetch 14 days BTC 5m data
  2. Load Kronos model on CUDA
  3. For each 10-min window:
       a) Kronos prediction → net_pct + range
       b) Quant fusion engine (HMM → Kalman → Hurst → GARCH → BMA → Kelly)
       c) Adaptive TP/SL from fusion
       d) Trade with Kelly-sized position
  4. Report results vs baseline antitrend

Usage:
  python kronos_quant_sim.py                          # 10 days (~1440 windows)
  python kronos_quant_sim.py --windows 200 --quick    # Quick test
  python kronos_quant_sim.py --capital 5000 --size 0.5
"""

import json, sys, os, time, argparse, warnings
from pathlib import Path
from datetime import datetime, timedelta
from collections import deque
import numpy as np
import pandas as pd
import torch

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Kronos ──
from model import Kronos, KronosTokenizer, KronosPredictor

# ── Quant Fusion ──
from quant_models.fusion_layer import QuantFusionEngine

# =========================================================================
# SETUP
# =========================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--windows', type=int, default=1440, help='Windows (~10 days at stride 2)')
parser.add_argument('--capital', type=float, default=5000)
parser.add_argument('--size', type=float, default=0.5, help='Baseline position size BTC')
parser.add_argument('--quick', action='store_true', help='Skip warmup progress bars')
parser.add_argument('--save', action='store_true', help='Save results')
args = parser.parse_args()

CAPITAL = args.capital
BASELINE_SIZE = args.size
LEVERAGE = 200
TP_PCT = 0.3   # base TP %
SL_PCT = 0.2   # base SL %
LOOKBACK = 200
STRIDE = 2     # every 2nd candle (10 min)

print(f"{'='*60}")
print(f"  KRONOS + QUANT FUSION — 10-DAY SIMULATION")
print(f"{'='*60}")
print(f"  Capital: ${CAPITAL:.0f} | Baseline size: {BASELINE_SIZE} BTC")
print(f"  Leverage: 1:{LEVERAGE} | Base TP/SL: {TP_PCT}%/{SL_PCT}%")
print(f"  Windows: {args.windows} (stride {STRIDE}, ~{args.windows * 5 * STRIDE // 60 // 24:.1f} days)")

# =========================================================================
# FETCH DATA
# =========================================================================
print("\n[1] Fetching 5m BTC data...")
import ccxt
ex = ccxt.binance({'options': {'defaultType': 'spot'}, 'timeout': 30000})
ex.load_markets()

all_c = []
since = ex.parse8601((datetime.utcnow() - timedelta(days=18)).isoformat())
while len(all_c) < 5000:
    o = ex.fetch_ohlcv('BTC/USDT', '5m', since=since, limit=1000)
    if not o:
        break
    all_c.extend(o)
    since = o[-1][0] + 1
    time.sleep(0.15)
    if not args.quick and len(all_c) % 1000 == 0:
        print(f"  Downloaded {len(all_c)} candles...")

df = pd.DataFrame(all_c, columns=['ts', 'o', 'h', 'l', 'c', 'v'])
df['t'] = pd.to_datetime(df['ts'], unit='ms')
df = df.drop_duplicates('ts').sort_values('ts').reset_index(drop=True)
print(f"  Loaded {len(df)} candles ({df['t'].iloc[0].strftime('%b %d')} → {df['t'].iloc[-1].strftime('%b %d')})")

# ── Fetch 1h data for HTF bias ──
print("\n  Fetching 1h BTC data for HTF bias...")
since_1h = ex.parse8601((datetime.utcnow() - timedelta(days=60)).isoformat())
all_1h = []
while len(all_1h) < 500:
    o = ex.fetch_ohlcv('BTC/USDT', '1h', since=since_1h, limit=500)
    if not o:
        break
    all_1h.extend(o)
    since_1h = o[-1][0] + 1
    time.sleep(0.1)
df_1h = pd.DataFrame(all_1h, columns=['ts', 'o', 'h', 'l', 'c', 'v'])
df_1h['t'] = pd.to_datetime(df_1h['ts'], unit='ms')
df_1h = df_1h.drop_duplicates('ts').sort_values('ts').reset_index(drop=True)
# Compute EMA50 on 1h closes
df_1h['ema50'] = df_1h['c'].ewm(span=50, adjust=False).mean()
print(f"  Loaded {len(df_1h)} 1h candles, latest EMA50=${df_1h['ema50'].iloc[-1]:.2f}")

# Rename for fusion engine
df_engine = df.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'})

# =========================================================================
# LOAD KRONOS
# =========================================================================
print("\n[2] Loading Kronos on CUDA...")
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f"  Device: {device}")
tok = KronosTokenizer.from_pretrained('NeoQuasar/Kronos-Tokenizer-base').to(device)
model = Kronos.from_pretrained('NeoQuasar/Kronos-base').to(device)
predictor = KronosPredictor(model, tok, max_context=512, device=device)
print("  ✅ Kronos loaded")

# =========================================================================
# KRONOS PREDICTION FUNCTION
# =========================================================================
def kronos_pred(df5, lb=LOOKBACK, pl=4):
    """Run Kronos inference on a window. Returns (net_pct, range_pct, direction)."""
    if len(df5) < lb:
        return None, 0, "NEUTRAL"
    use_lb = min(lb, len(df5))
    x = df5.iloc[-use_lb:][['o', 'h', 'l', 'c', 'v']].copy()
    x.columns = ['open', 'high', 'low', 'close', 'volume']
    x['amount'] = 0.0
    ts = df5['t'].iloc[-use_lb:].reset_index(drop=True)
    yt = pd.Series(pd.date_range(
        start=ts.iloc[-1] + pd.Timedelta(minutes=5),
        periods=pl, freq='5min'
    ))
    try:
        with torch.no_grad():
            p = predictor.predict(
                df=x, x_timestamp=ts, y_timestamp=yt,
                pred_len=pl, T=1.0, top_p=0.9,
                sample_count=5, verbose=False
            )
        pc = float(p['close'].iloc[-1])
        cp = float(df5['c'].iloc[-1])
        net = ((pc - cp) / cp) * 100
        rng = ((p['high'].max() - p['low'].min()) / cp) * 100

        if net > 0.1:
            direction = "BULLISH"
        elif net < -0.1:
            direction = "BEARISH"
        else:
            direction = "NEUTRAL"

        return net, rng, direction
    except Exception as e:
        return None, 0, "NEUTRAL"

# =========================================================================
# BASELINE STRATEGY
# =========================================================================
def antitrend_signal(net, rng):
    """Pure antitrend: trade AGAINST extreme Kronos predictions."""
    if net is None or rng < 0.08:
        return "HOLD", 0.0
    if net > 0.10:
        # Kronos says UP hard → SELL (mean reversion down)
        return "SELL", min(abs(net) / 0.3, 1.0)
    elif net < -0.10:
        # Kronos says DOWN hard → BUY (mean reversion up)
        return "BUY", min(abs(net) / 0.3, 1.0)
    return "HOLD", 0.0

# =========================================================================
# SIMULATOR
# =========================================================================
class Simulator:
    """Tracks positions, PnL, and trade history."""

    def __init__(self, capital, tp_pct, sl_pct, leverage=200):
        self.capital = capital
        self.initial_capital = capital
        self.tp_pct = tp_pct
        self.sl_pct = sl_pct
        self.leverage = leverage
        self.position = None  # {side, entry, size, tp, sl, bar}
        self.trades = []
        self.margin_locked = 0.0

    def open(self, side, price, size_btc, bar, tp_pct=None, sl_pct=None):
        if self.position is not None:
            return
        tp = tp_pct if tp_pct is not None else self.tp_pct
        sl = sl_pct if sl_pct is not None else self.sl_pct
        margin = size_btc * price / self.leverage
        if margin > self.capital:
            return
        self.capital -= margin
        self.margin_locked = margin
        self.position = {
            'side': side,
            'entry': price,
            'size': size_btc,
            'tp': price * (1 + tp/100) if side == 'BUY' else price * (1 - tp/100),
            'sl': price * (1 - sl/100) if side == 'BUY' else price * (1 + sl/100),
            'bar': bar,
        }

    def update(self, row_idx, df5):
        """Check candle for TP/SL/expiry."""
        if self.position is None:
            return
        pos = self.position
        hi = float(df5.iloc[row_idx]['h'])
        lo = float(df5.iloc[row_idx]['l'])
        cl = float(df5.iloc[row_idx]['c'])
        exit_price = None
        reason = None

        if pos['side'] == 'BUY':
            if hi >= pos['tp']:
                exit_price = pos['tp']; reason = 'TP'
            elif lo <= pos['sl']:
                exit_price = pos['sl']; reason = 'SL'
        else:
            if lo <= pos['tp']:
                exit_price = pos['tp']; reason = 'TP'
            elif hi >= pos['sl']:
                exit_price = pos['sl']; reason = 'SL'

        # Time-stop after 4 candles (20 min)
        if exit_price is None and (row_idx - pos['bar']) >= 4:
            exit_price = cl
            reason = 'EXPIRY'

        if exit_price is not None:
            self._close(exit_price, reason)

    def _close(self, exit_price, reason):
        pos = self.position
        if pos['side'] == 'BUY':
            pnl_pct = (exit_price - pos['entry']) / pos['entry']
        else:
            pnl_pct = (pos['entry'] - exit_price) / pos['entry']
        pnl_usd = pnl_pct * pos['size'] * pos['entry']
        self.capital += self.margin_locked + pnl_usd
        self.trades.append({
            'side': pos['side'],
            'entry': round(pos['entry'], 2),
            'exit': round(exit_price, 2),
            'size': pos['size'],
            'pnl': round(pnl_usd, 2),
            'pnl_pct': round(pnl_pct * 100, 2),
            'reason': reason,
        })
        self.position = None
        self.margin_locked = 0.0

    def stats(self):
        wins = [t for t in self.trades if t['pnl'] > 0]
        losses = [t for t in self.trades if t['pnl'] < 0]
        total_pnl = sum(t['pnl'] for t in self.trades)
        return {
            'total_trades': len(self.trades),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': round(len(wins) / max(len(self.trades), 1) * 100, 1),
            'total_pnl': round(total_pnl, 2),
            'final_capital': round(self.capital, 2),
            'return_pct': round((self.capital - self.initial_capital) / self.initial_capital * 100, 2),
            'avg_win': round(sum(t['pnl'] for t in wins) / max(len(wins), 1), 2),
            'avg_loss': round(abs(sum(t['pnl'] for t in losses)) / max(len(losses), 1), 2),
            'avg_win_pct': round(sum(t['pnl_pct'] for t in wins) / max(len(wins), 1), 2),
            'avg_loss_pct': round(abs(sum(t['pnl_pct'] for t in losses)) / max(len(losses), 1), 2),
            'tp_count': len([t for t in self.trades if t['reason'] == 'TP']),
            'sl_count': len([t for t in self.trades if t['reason'] == 'SL']),
            'expiry_count': len([t for t in self.trades if t['reason'] == 'EXPIRY']),
        }

# =========================================================================
# MAIN LOOP
# =========================================================================
print("\n[3] Initializing quant fusion engine...")
qf = QuantFusionEngine(
    capital=CAPITAL, leverage=LEVERAGE,
    base_tp_pct=TP_PCT, base_sl_pct=SL_PCT,
)
qf.warmup(df_engine)
qf.initialize(btc_price=float(df['c'].iloc[-1]))
print("  ✅ Fusion engine ready")

# Init simulators
sim_quant = Simulator(CAPITAL, TP_PCT, SL_PCT, LEVERAGE)
sim_baseline = Simulator(CAPITAL, TP_PCT, SL_PCT, LEVERAGE)
sim_kronos_raw = Simulator(CAPITAL, TP_PCT, SL_PCT, LEVERAGE)

# Build indices
indices = list(range(LOOKBACK + 5, min(len(df) - 5, args.windows + LOOKBACK + 5), STRIDE))
print(f"\n[4] Simulating {len(indices)} windows...")

start_time = time.time()
pred_times = []

for batch, idx in enumerate(indices):
    # Progress
    if batch % 50 == 0:
        elapsed = time.time() - start_time
        pct = batch / len(indices) * 100
        if batch > 0:
            eta = elapsed / (batch + 1) * (len(indices) - batch)
        else:
            eta = 0
        bar_len = 20
        filled = int(batch * bar_len / len(indices))
        prog = '█' * filled + '░' * (bar_len - filled)
        print(f"  [{prog}] {pct:.0f}% | {batch}/{len(indices)} | ETA: {eta:.0f}s", end='\r', flush=True)

    # Window
    ctx = df.iloc[idx - LOOKBACK:idx]
    ctx_eng = df_engine.iloc[idx - LOOKBACK:idx]
    price = float(ctx['c'].iloc[-1])
    high = float(ctx['h'].iloc[-1])
    low = float(ctx['l'].iloc[-1])

    # ── KRONOS PREDICTION ──
    t0 = time.time()
    net, rng, direction = kronos_pred(ctx)
    pred_times.append(time.time() - t0)

    # ── Update HTF bias every 12 batches (~1h at stride 2) ──
    if batch % 12 == 0:
        # Find the nearest 1h candle to current time
        curr_ts = df['t'].iloc[idx]
        idx_1h = (df_1h['t'] - curr_ts).abs().idxmin()
        if 0 <= idx_1h < len(df_1h):
            qf.set_htf_bias(
                ema50=float(df_1h['ema50'].iloc[idx_1h]),
                current_price=price,
            )

    if net is None:
        # Update positions and continue
        sim_quant.update(idx, df)
        sim_baseline.update(idx, df)
        sim_kronos_raw.update(idx, df)
        continue

    # Determine Kronos confidence from prediction magnitude
    kronos_conf = min(abs(net) / 0.3, 1.0) if abs(net) > 0.04 else 0.0

    # ── BASELINE: Pure antitrend ──
    base_dir, base_conf = antitrend_signal(net, rng)
    if base_dir != 'HOLD' and base_conf > 0.3:
        sim_baseline.open(base_dir, price, BASELINE_SIZE, idx)

    # ── BASELINE 2: Kronos raw (follow direction directly) ──
    if abs(net) > 0.04:
        kronos_dir = 'BUY' if net > 0.04 else 'SELL'
        sim_kronos_raw.open(kronos_dir, price, BASELINE_SIZE, idx)

    # ── QUANT FUSION ──
    fusion_result = qf.analyze(
        df=ctx_eng,
        kronos_direction=direction,
        kronos_net_pct=net,
        kronos_confidence=kronos_conf,
        current_price=price,
        strategy_signals={
            'antitrend': (1 if base_dir == 'BUY' else -1 if base_dir == 'SELL' else 0, base_conf),
            'kronos_raw': (1 if net > 0.04 else -1 if net < -0.04 else 0, kronos_conf),
        },
    )

    # Execute quant fusion trade
    if fusion_result['decision'] in ('BUY', 'SELL') and fusion_result['confidence'] > 0.3:
        sim_quant.open(
            fusion_result['decision'],
            price,
            fusion_result['size']['size_btc'],
            idx,
            tp_pct=fusion_result['tp_sl']['final_tp_pct'],
            sl_pct=fusion_result['tp_sl']['final_sl_pct'],
        )

    # ── Update positions ──
    sim_quant.update(idx, df)
    sim_baseline.update(idx, df)
    sim_kronos_raw.update(idx, df)

print(f"  [{'█' * 20}] 100% ✅")

# =========================================================================
# RESULTS
# =========================================================================
avg_pred_time = np.mean(pred_times) if pred_times else 0
print(f"\n{'='*60}")
print(f"  RESULTS — {len(indices)} windows ({avg_pred_time:.2f}s avg/pred)")
print(f"{'='*60}")

quant_stats = sim_quant.stats()
base_stats = sim_baseline.stats()
kronos_raw_stats = sim_kronos_raw.stats()

print(f"\n  {'Metric':22s} {'Quant Fusion':>16s} {'Antitrend':>12s} {'Kronos Raw':>12s}")
print(f"  {'─'*22} {'─'*16} {'─'*12} {'─'*12}")
print(f"  {'Final capital':22s} ${quant_stats['final_capital']:>10.2f}  ${base_stats['final_capital']:>8.2f}  ${kronos_raw_stats['final_capital']:>8.2f}")
print(f"  {'Return':22s} {quant_stats['return_pct']:>10.2f}%  {base_stats['return_pct']:>8.2f}%  {kronos_raw_stats['return_pct']:>8.2f}%")
print(f"  {'Total trades':22s} {quant_stats['total_trades']:>10d}  {base_stats['total_trades']:>8d}  {kronos_raw_stats['total_trades']:>8d}")
print(f"  {'Win rate':22s} {quant_stats['win_rate']:>10.1f}%  {base_stats['win_rate']:>8.1f}%  {kronos_raw_stats['win_rate']:>8.1f}%")
print(f"  {'Avg win':22s} ${quant_stats['avg_win']:>10.2f}  ${base_stats['avg_win']:>8.2f}  ${kronos_raw_stats['avg_win']:>8.2f}")
print(f"  {'Avg loss':22s} ${quant_stats['avg_loss']:>10.2f}  ${base_stats['avg_loss']:>8.2f}  ${kronos_raw_stats['avg_loss']:>8.2f}")
print(f"  {'Avg win %':22s} {quant_stats['avg_win_pct']:>10.2f}%  {base_stats['avg_win_pct']:>8.2f}%  {kronos_raw_stats['avg_win_pct']:>8.2f}%")
print(f"  {'Avg loss %':22s} {quant_stats['avg_loss_pct']:>10.2f}%  {base_stats['avg_loss_pct']:>8.2f}%  {kronos_raw_stats['avg_loss_pct']:>8.2f}%")
print(f"  {'TP/SL/Expiry':22s} {quant_stats['tp_count']:>4d}/{quant_stats['sl_count']}/{quant_stats['expiry_count']:>4d}  "
      f"{base_stats['tp_count']:>4d}/{base_stats['sl_count']}/{base_stats['expiry_count']:>4d}  "
      f"{kronos_raw_stats['tp_count']:>4d}/{kronos_raw_stats['sl_count']}/{kronos_raw_stats['expiry_count']:>4d}")

# Quant vs antitrend improvement
if base_stats['final_capital'] != 0:
    diff = quant_stats['final_capital'] - base_stats['final_capital']
    pct_imp = (quant_stats['final_capital'] - base_stats['final_capital']) / base_stats['final_capital'] * 100
    print(f"\n  Quant vs Antitrend: ${diff:+.2f} ({pct_imp:+.1f}%)")

# Quant vs kronos raw
if kronos_raw_stats['final_capital'] != 0:
    diff2 = quant_stats['final_capital'] - kronos_raw_stats['final_capital']
    pct2 = (quant_stats['final_capital'] - kronos_raw_stats['final_capital']) / kronos_raw_stats['final_capital'] * 100
    print(f"  Quant vs Kronos Raw: ${diff2:+.2f} ({pct2:+.1f}%)")

print(f"\n  Avg prediction time: {avg_pred_time:.2f}s")
print(f"  Total simulation time: {time.time() - start_time:.0f}s")

# ── Print fusion summary example ──
print(f"\n  ── Last Fusion Decision ──")
print(qf.summary())

# ── Save ──
if args.save:
    results = {
        'config': {
            'windows': len(indices),
            'capital': CAPITAL,
            'baseline_size': BASELINE_SIZE,
            'leverage': LEVERAGE,
            'tp_pct': TP_PCT,
            'sl_pct': SL_PCT,
        },
        'quant_fusion': quant_stats,
        'antitrend_baseline': base_stats,
        'kronos_raw': kronos_raw_stats,
    }
    path = f'quant_sim_results_{datetime.now().strftime("%Y%m%d_%H%M")}.json'
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), path), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to {path}")
