#!/usr/bin/env python3
"""
kronos_selector_train.py — Train XGBoost sample selector for Kronos.

Pipeline:
  1. Fetch 15000+ BTC 5m candles from Binance (~52 days)
  2. For each window, generate 50 Kronos samples
  3. Extract features and label each sample (PROFIT-BASED: would TP have hit before SL?)
  4. Train XGBoost classifier on ~150k labeled samples
  5. Save trained model + feature importance report

Labels use TP=0.3%, SL=0.2% to simulate actual trade profitability,
not just directional accuracy. This trains the selector to pick samples
that make money, not just samples that guess direction correctly.

Usage:
  python kronos_selector_train.py --windows 500 --samples 20 --quick
  python kronos_selector_train.py --windows 3000 --samples 50 --save
"""

import sys, os, time, argparse, json, warnings
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import Kronos, KronosTokenizer, KronosPredictor
from quant_models.sample_selector import (
    predict_samples, extract_feature_vector, FEATURE_NAMES,
    label_sample, SampleSelector,
)

# =========================================================================
# CONFIG
# =========================================================================
LOOKBACK = 200
PRED_LEN = 4
STRIDE = 5  # every 5th window to reduce overlap
SAMPLE_COUNT = 50  # samples per window (higher = training data quality)

parser = argparse.ArgumentParser()
parser.add_argument('--windows', type=int, default=3000, help='Total windows to process')
parser.add_argument('--samples', type=int, default=SAMPLE_COUNT, help='Samples per Kronos prediction')
parser.add_argument('--quick', action='store_true', help='Run fewer windows')
parser.add_argument('--save', action='store_true', help='Save model')
args = parser.parse_args()

if args.quick:
    args.windows = min(args.windows, 200)

print(f"{'='*60}")
print(f"  KRONOS SAMPLE SELECTOR TRAINING")
print(f"{'='*60}")
print(f"  Windows: {args.windows} | Samples/window: {args.samples}")
print(f"  Lookback: {LOOKBACK} | Pred len: {PRED_LEN} | Stride: {STRIDE}")

# =========================================================================
# FETCH DATA
# =========================================================================
print("\n[1] Fetching BTC 5m data...")
import ccxt
ex = ccxt.binance({'options': {'defaultType': 'spot'}, 'timeout': 30000})
ex.load_markets()

all_c = []
since = ex.parse8601((datetime.utcnow() - timedelta(days=60)).isoformat())
while len(all_c) < 15000:
    o = ex.fetch_ohlcv('BTC/USDT', '5m', since=since, limit=1000)
    if not o: break
    all_c.extend(o)
    since = o[-1][0] + 1
    time.sleep(0.12)

df = pd.DataFrame(all_c, columns=['ts', 'o', 'h', 'l', 'c', 'v'])
df['t'] = pd.to_datetime(df['ts'], unit='ms')
df = df.drop_duplicates('ts').sort_values('ts').reset_index(drop=True)
print(f"  {len(df)} candles ({df['t'].iloc[0].strftime('%b %d')} → {df['t'].iloc[-1].strftime('%b %d')})")

# =========================================================================
# LOAD KRONOS
# =========================================================================
print("\n[2] Loading Kronos on CUDA...")
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
tok = KronosTokenizer.from_pretrained('NeoQuasar/Kronos-Tokenizer-base').to(device)
model = Kronos.from_pretrained('NeoQuasar/Kronos-base').to(device)
predictor = KronosPredictor(model, tok, max_context=512, device=device)
print(f"  ✅ Kronos loaded on {device}")

# =========================================================================
# GENERATE SAMPLES + LABELS
# =========================================================================
print(f"\n[3] Generating {args.samples} samples per window...")
print(f"  Total predictions: {args.windows} windows × {args.samples} samples = {args.windows * args.samples}")

indices = list(range(LOOKBACK + PRED_LEN + 1, len(df) - PRED_LEN, STRIDE))
indices = indices[:args.windows]
print(f"  Actual windows: {len(indices)}")

all_features = []
all_labels = []
window_metrics = []
pred_times = []
total_samples = 0
correct_by_direction = defaultdict(lambda: {'correct': 0, 'total': 0})

start_time = time.time()

for batch, idx in enumerate(indices):
    if batch % 50 == 0:
        elapsed = time.time() - start_time
        pct = batch / len(indices) * 100
        rate = (batch + 1) / max(elapsed, 0.1)
        eta = (len(indices) - batch) / max(rate, 0.1) if rate > 0 else 0
        bar_len = 20
        filled = int(batch * bar_len / len(indices))
        prog = '█' * filled + '░' * (bar_len - filled)
        print(f"  [{prog}] {pct:.0f}% | {batch}/{len(indices)} | {rate:.1f}w/s | ETA: {eta:.0f}s", end='\r', flush=True)

    ctx = df.iloc[idx - LOOKBACK:idx]

    # Kronos multiple samples
    t0 = time.time()
    try:
        samples, current_price, avg_info = predict_samples(
            predictor, ctx,
            lookback=LOOKBACK, pred_len=PRED_LEN,
            sample_count=args.samples,
        )
    except Exception as e:
        continue
    pred_times.append(time.time() - t0)

    if not samples:
        continue

    # Look ahead 4 candles for labeling
    future = df.iloc[idx:idx + PRED_LEN]
    if len(future) < PRED_LEN:
        continue

    # Label each sample and extract features
    for s in samples:
        features = extract_feature_vector(s)
        label = label_sample(s, future, current_price, tp_pct=0.3, sl_pct=0.2)
        all_features.append(features)
        all_labels.append(label)
        total_samples += 1

        if s['direction'] in ('BULLISH', 'BEARISH'):
            correct_by_direction[s['direction']]['total'] += 1
            if label == 1:
                correct_by_direction[s['direction']]['correct'] += 1

    # Track overall accuracy per window (using avg prediction as baseline)
    avg_dir = avg_info['direction']
    f_high = float(future['h'].max()) if 'h' in future.columns else float(future['high'].max())
    f_low = float(future['l'].min()) if 'l' in future.columns else float(future['low'].min())
    f_close = float(future['c'].iloc[-1]) if 'c' in future.columns else float(future['close'].iloc[-1])
    if avg_dir == 'BULLISH':
        avg_correct = 1 if f_close > f_high * 0.99 else 0
    elif avg_dir == 'BEARISH':
        avg_correct = 1 if f_close < f_low * 1.01 else 0
    else:
        avg_correct = 1
    window_metrics.append({
        'window': batch,
        'price': current_price,
        'avg_direction': avg_dir,
        'avg_correct': avg_correct,
        'avg_net': avg_info['net_change_pct'],
        'sample_count': len(samples),
    })

print(f"\n  [{'█' * 20}] ✅ Done")

# =========================================================================
# TRAIN XGBOOST
# =========================================================================
print(f"\n[4] Training XGBoost on {len(all_features)} samples...")
X = np.array(all_features, dtype=np.float32)
y = np.array(all_labels, dtype=np.int32)

# Stats
n_correct = int(y.sum())
n_total = len(y)
print(f"  Dataset: {n_total} samples ({n_correct} correct, {n_total - n_correct} wrong)")
print(f"  Baseline accuracy: {n_correct / max(n_total, 1) * 100:.1f}% (direction, 0.1% threshold)")

for d, counts in correct_by_direction.items():
    wr = counts['correct'] / max(counts['total'], 1) * 100
    print(f"    {d}: {counts['correct']}/{counts['total']} ({wr:.1f}%)")

# Train/test split
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

selector = SampleSelector()
print(f"  Training on {len(X_train)} samples, testing on {len(X_test)}...")
importance = selector.train(X_train, y_train)

# Evaluate
from xgboost import DMatrix
dtest = DMatrix(X_test, feature_names=FEATURE_NAMES)
y_pred_prob = selector.model.predict(dtest)
y_pred = (y_pred_prob > 0.5).astype(int)

test_acc = np.mean(y_pred == y_test)
print(f"\n  Test accuracy: {test_acc * 100:.1f}%")

# Confidence bin analysis
bins = [(0.0, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.0)]
print(f"\n  Confidence Calibration:")
print(f"  {'Bin':>12s} | {'Count':>6s} | {'Correct':>6s} | {'Acc':>6s}")
print(f"  {'─'*12}─┼─{'─'*6}─┼─{'─'*6}─┼─{'─'*6}")
for lo, hi in bins:
    mask = (y_pred_prob >= lo) & (y_pred_prob < hi)
    n_bin = int(mask.sum())
    if n_bin > 0:
        acc_bin = np.mean(y_test[mask] == y_pred[mask])
        print(f"  [{lo:.1f}-{hi:.1f})  | {n_bin:>5d}  | {int(y_test[mask].sum()):>5d}  | {acc_bin*100:>5.1f}%")

# Feature importance
print(f"\n  Feature Importance (gain):")
for feat, gain in sorted(importance.items(), key=lambda x: -x[1])[:10]:
    print(f"    {feat:20s}: {gain:.3f}")

# ── How much better than average? ──
# For each test window, simulate picking best sample
print(f"\n  Picking Best Sample Simulation (test set):")
from collections import defaultdict
test_windows = defaultdict(list)
for i in range(split, len(X)):
    test_windows[i // args.samples].append((y_test[i - split], y_pred_prob[i - split]))

selector_wins = 0
avg_wins = 0
total_win = 0
for wid, samples in test_windows.items():
    if len(samples) < 5:
        continue
    # Best sample (highest prob)
    best_label = max(samples, key=lambda x: x[1])[0]
    selector_wins += best_label
    # Average sample
    avg_label = np.mean([s[0] for s in samples])
    avg_wins += 1 if avg_label > 0.5 else 0
    total_win += 1

if total_win > 0:
    print(f"    Selector picks best sample: {selector_wins}/{total_win} ({selector_wins/total_win*100:.1f}%)")
    print(f"    Average of all samples:     {avg_wins}/{total_win} ({avg_wins/total_win*100:.1f}%)")
    print(f"    Improvement: +{(selector_wins - avg_wins) / max(total_win, 1) * 100:.1f}pp")

# ═══════════════════════════════════════════════
# SAVE
# ═══════════════════════════════════════════════
if args.save:
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'quant_models', 'models')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'kronos_selector.xgb')
    selector.save(model_path)
    print(f"\n  ✅ Model saved to {model_path}")

    # Save training stats
    report = {
        'windows': len(indices),
        'samples_per_window': args.samples,
        'total_samples': len(all_features),
        'baseline_accuracy': round(n_correct / max(n_total, 1), 3),
        'test_accuracy': round(float(test_acc), 3),
        'feature_importance': {k: round(v, 3) for k, v in sorted(importance.items(), key=lambda x: -x[1])},
        'avg_pred_time': round(float(np.mean(pred_times)), 3),
        'total_time': round(time.time() - start_time, 1),
    }
    report_path = os.path.join(model_dir, 'training_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"  Report saved to {report_path}")

print(f"\n  Avg prediction time: {np.mean(pred_times):.3f}s")
print(f"  Total time: {time.time() - start_time:.0f}s")
print(f"{'='*60}")
