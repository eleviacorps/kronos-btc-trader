"""
Sample Selector — Learns to pick the best Kronos sample from multiple predictions.

Instead of averaging all Kronos samples (which dilutes accuracy to ~40%),
a selector model identifies which individual samples are likely correct.
If the selector achieves even 55-60% accuracy in picking the winning sample,
overall WR jumps significantly.

Architecture:
  1. Kronos generates N samples per window (50+)
  2. Feature extraction: per-sample characteristics (net, range, vol, candle shape, etc.)
  3. XGBoost classifier: predicts P(sample is directionally correct)
  4. Pick sample with highest P(correct) → direction + confidence

Training:
  - Run Kronos over 2000+ historical BTC windows
  - For each window, label each sample: "was the predicted direction correct?"
  - Train XGBoost on (sample_features → label)
  - Feature importance reveals what makes a Kronos sample trustworthy
"""

import numpy as np
import pandas as pd
import pickle
import os
from pathlib import Path
from collections import defaultdict

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    xgb = None
    HAS_XGB = False


def predict_samples(predictor, df_input, lookback=200, pred_len=4, sample_count=50, T=1.0, top_p=0.9):
    """
    Run Kronos prediction and return ALL individual sample predictions.

    This bypasses the internal averaging in auto_regressive_inference
    by calling the model's generator directly.

    Args:
        predictor: KronosPredictor instance (loaded on CUDA)
        df_input: DataFrame with OHLCV columns (o/h/l/c/v)
        lookback: candles of history
        pred_len: candles to predict
        sample_count: how many samples to generate (50+)
        T: sampling temperature
        top_p: nucleus sampling threshold

    Returns:
        samples: list of dicts, one per sample, each with:
            - 'net_change_pct': predicted % change
            - 'range_pct': predicted range
            - 'direction': 'BULLISH'/'BEARISH'/'NEUTRAL'
            - 'candle_0_change': first candle % change
            - 'last_close': predicted last close
            - 'high': max high
            - 'low': min low
            - 'candle_changes': list of per-candle % changes
            - 'volatility': std of predicted closes / mean
            - 'max_drawdown': max peak-to-trough in prediction
            - 'linearity': R² of linear fit to predicted closes
            - 'acceleration': second derivative of predicted closes
            - 'consensus_divergence': how much this sample differs from avg
        current_price: latest BTC price
        avg_prediction: averaged prediction dict
    """
    import torch

    # Prepare data same as KronosEngine
    x = df_input.iloc[-lookback:][['o', 'h', 'l', 'c', 'v']].copy()
    x.columns = ['open', 'high', 'low', 'close', 'volume']
    x['amount'] = 0.0

    ts = df_input['t'].iloc[-lookback:].reset_index(drop=True) if 't' in df_input.columns else pd.Series(pd.date_range(end=pd.Timestamp.now(), periods=lookback, freq='5min'))
    yt = pd.Series(pd.date_range(start=ts.iloc[-1] + pd.Timedelta(minutes=5), periods=pred_len, freq='5min'))

    current_price = float(x['close'].iloc[-1])

    # Import Kronos helper functions
    from model.kronos import calc_time_stamps, sample_from_logits

    x_time_df = calc_time_stamps(ts)
    y_time_df = calc_time_stamps(yt)

    x_vals = x[predictor.price_cols + [predictor.vol_col, predictor.amt_vol]].values.astype(np.float32)
    x_stamp = x_time_df.values.astype(np.float32)
    y_stamp = y_time_df.values.astype(np.float32)

    x_mean, x_std = np.mean(x_vals, axis=0), np.std(x_vals, axis=0)
    x_norm = ((x_vals - x_mean) / (x_std + 1e-5)).clip(-predictor.clip, predictor.clip)

    x_norm = x_norm[np.newaxis, :]  # [1, lookback, 6]
    x_stamp = x_stamp[np.newaxis, :]
    y_stamp = y_stamp[np.newaxis, :]

    # Generate with multiple samples
    with torch.no_grad():
        x_t = torch.from_numpy(x_norm).to(predictor.device)
        xs_t = torch.from_numpy(x_stamp).to(predictor.device)
        ys_t = torch.from_numpy(y_stamp).to(predictor.device)

        # Duplicate samples
        x_t = x_t.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(-1, x_t.size(1), x_t.size(2))
        xs_t = xs_t.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(-1, xs_t.size(1), xs_t.size(2))
        ys_t = ys_t.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(-1, ys_t.size(1), ys_t.size(2))

        x_token = predictor.tokenizer.encode(x_t, half=True)

        initial_seq_len = x_t.size(1)
        batch_size = x_token[0].size(0)
        max_context = predictor.max_context

        generated_pre = x_token[0].new_empty(batch_size, pred_len)
        generated_post = x_token[1].new_empty(batch_size, pred_len)

        # Buffers
        pre_buffer = x_token[0].new_zeros(batch_size, max_context)
        post_buffer = x_token[1].new_zeros(batch_size, max_context)
        buffer_len = min(initial_seq_len, max_context)
        if buffer_len > 0:
            start_idx = max(0, initial_seq_len - max_context)
            pre_buffer[:, :buffer_len] = x_token[0][:, start_idx:start_idx + buffer_len]
            post_buffer[:, :buffer_len] = x_token[1][:, start_idx:start_idx + buffer_len]

        full_stamp = torch.cat([xs_t, ys_t], dim=1)

        # Autoregressive generation
        for i in range(pred_len):
            current_seq_len = initial_seq_len + i
            window_len = min(current_seq_len, max_context)

            if current_seq_len <= max_context:
                input_tokens = [pre_buffer[:, :window_len], post_buffer[:, :window_len]]
            else:
                input_tokens = [pre_buffer, post_buffer]

            context_end = current_seq_len
            context_start = max(0, context_end - max_context)
            current_stamp = full_stamp[:, context_start:context_end, :].contiguous()

            s1_logits, context = predictor.model.decode_s1(input_tokens[0], input_tokens[1], current_stamp)
            s1_logits = s1_logits[:, -1, :]
            sample_pre = sample_from_logits(s1_logits, temperature=T, top_k=0, top_p=top_p, sample_logits=True)

            s2_logits = predictor.model.decode_s2(context, sample_pre)
            s2_logits = s2_logits[:, -1, :]
            sample_post = sample_from_logits(s2_logits, temperature=T, top_k=0, top_p=top_p, sample_logits=True)

            generated_pre[:, i] = sample_pre.squeeze(-1)
            generated_post[:, i] = sample_post.squeeze(-1)

            if current_seq_len < max_context:
                pre_buffer[:, current_seq_len] = sample_pre.squeeze(-1)
                post_buffer[:, current_seq_len] = sample_post.squeeze(-1)
            else:
                pre_buffer.copy_(torch.roll(pre_buffer, shifts=-1, dims=1))
                post_buffer.copy_(torch.roll(post_buffer, shifts=-1, dims=1))
                pre_buffer[:, -1] = sample_pre.squeeze(-1)
                post_buffer[:, -1] = sample_post.squeeze(-1)

        # Decode all samples
        full_pre = torch.cat([x_token[0], generated_pre], dim=1)
        full_post = torch.cat([x_token[1], generated_post], dim=1)

        context_start = max(0, initial_seq_len + pred_len - max_context)
        input_tokens = [
            full_pre[:, context_start:initial_seq_len + pred_len].contiguous(),
            full_post[:, context_start:initial_seq_len + pred_len].contiguous()
        ]
        z = predictor.tokenizer.decode(input_tokens, half=True)
        # z shape: [batch * sample_count, pred_len, 6]
        z = z.reshape(-1, sample_count, z.size(1), z.size(2))
        all_samples = z.cpu().numpy()  # [1, sample_count, pred_len, 6]

    # Denormalize and extract features per sample
    samples = []
    for s in range(sample_count):
        raw = all_samples[0, s] * (x_std + 1e-5) + x_mean  # [pred_len, 6]
        pred_close = raw[:, 3]  # close col
        pred_high = raw[:, 1]   # high col
        pred_low = raw[:, 2]    # low col

        net_change = ((pred_close[-1] - current_price) / current_price) * 100
        range_pct = ((pred_high.max() - pred_low.min()) / current_price) * 100
        candle_changes = [float(pred_close[i+1] - pred_close[i]) / pred_close[i] * 100 for i in range(len(pred_close) - 1)]

        if net_change > 0.1:
            direction = "BULLISH"
        elif net_change < -0.1:
            direction = "BEARISH"
        else:
            direction = "NEUTRAL"

        # Volatility of predicted closes
        vol = float(np.std(pred_close) / np.mean(pred_close) * 100) if np.mean(pred_close) > 0 else 0

        # Linearity (R² of linear fit) — straight line = less informative
        x_lin = np.arange(len(pred_close))
        if np.std(pred_close) > 0:
            corr = np.corrcoef(x_lin, pred_close)[0, 1]
            linearity = float(corr ** 2)
        else:
            linearity = 1.0

        # Acceleration (second derivative approximation)
        if len(candle_changes) >= 2:
            accel = float(np.mean(np.diff(candle_changes)))
        else:
            accel = 0.0

        # Max drawdown within prediction
        peak = pred_close[0]
        mdd = 0.0
        for p in pred_close:
            if p > peak:
                peak = p
            dd = (peak - p) / peak * 100
            if dd > mdd:
                mdd = float(dd)

        # First candle move
        first_candle = float(candle_changes[0]) if candle_changes else 0.0

        samples.append({
            'sample_id': s,
            'net_change_pct': round(net_change, 4),
            'range_pct': round(range_pct, 4),
            'direction': direction,
            'last_close': round(float(pred_close[-1]), 2),
            'high': round(float(pred_high.max()), 2),
            'low': round(float(pred_low.min()), 2),
            'candle_changes': [round(c, 4) for c in candle_changes],
            'volatility': round(vol, 4),
            'linearity': round(linearity, 4),
            'acceleration': round(accel, 4),
            'max_drawdown': round(mdd, 4),
            'first_candle_pct': round(first_candle, 4),
        })

    # Avg prediction (what the current pipeline uses)
    avg_raw = np.mean(all_samples[0], axis=0) * (x_std + 1e-5) + x_mean
    avg_close = avg_raw[:, 3]
    avg_net = ((avg_close[-1] - current_price) / current_price) * 100

    avg_info = {
        'net_change_pct': round(float(avg_net), 4),
        'range_pct': round(float(((avg_raw[:, 1].max() - avg_raw[:, 2].min()) / current_price) * 100), 4),
        'direction': 'BULLISH' if avg_net > 0.1 else 'BEARISH' if avg_net < -0.1 else 'NEUTRAL',
    }

    return samples, current_price, avg_info


def extract_feature_vector(sample: dict) -> np.ndarray:
    """Convert a sample dict to a flat feature vector for XGBoost."""
    features = [
        sample['net_change_pct'],
        sample['range_pct'],
        sample['volatility'],
        sample['linearity'],
        sample['acceleration'],
        sample['max_drawdown'],
        sample['first_candle_pct'],
        # Candle-by-candle momentum
        sample['candle_changes'][0] if len(sample['candle_changes']) > 0 else 0,
        sample['candle_changes'][1] if len(sample['candle_changes']) > 1 else 0,
        sample['candle_changes'][2] if len(sample['candle_changes']) > 2 else 0,
        # Direction dummies
        1.0 if sample['direction'] == 'BULLISH' else 0.0,
        1.0 if sample['direction'] == 'BEARISH' else 0.0,
    ]
    return np.array(features, dtype=np.float32)


FEATURE_NAMES = [
    'net_change_pct', 'range_pct', 'volatility', 'linearity',
    'acceleration', 'max_drawdown', 'first_candle_pct',
    'candle_0', 'candle_1', 'candle_2',
    'is_bullish', 'is_bearish',
]


def label_sample(sample: dict, df_future: pd.DataFrame, entry_price: float,
                 tp_pct: float = 0, sl_pct: float = 0) -> int:
    direction = sample['direction']
    if direction == 'BULLISH':
        hi = float(df_future['h'].max()) if 'h' in df_future.columns else float(df_future['high'].max())
        return 1 if hi >= entry_price * 1.001 else 0
    elif direction == 'BEARISH':
        lo = float(df_future['l'].min()) if 'l' in df_future.columns else float(df_future['low'].min())
        return 1 if lo <= entry_price * 0.999 else 0
    return 0


def label_sample_strict(sample: dict, future_high: float, future_low: float,
                        entry_price: float, tp_pct: float, sl_pct: float) -> int:
    """
    Label using TP/SL logic — was this sample's direction profitable?

    1 = sample's direction would have hit TP before SL in next 4 candles
    0 = sample's direction would have hit SL or expired flat
    """
    direction = sample['direction']
    if direction == 'BULLISH':
        tp = entry_price * (1 + tp_pct / 100)
        sl = entry_price * (1 - sl_pct / 100)
        return 1 if future_high >= tp else 0
    elif direction == 'SELL':
        tp = entry_price * (1 - tp_pct / 100)
        sl = entry_price * (1 + sl_pct / 100)
        return 1 if future_low <= tp else 0
    else:
        return 0


class SampleSelector:
    """XGBoost-based selector that picks the best Kronos sample."""

    def __init__(self, model_path: str = None):
        self.model = None
        self.feature_names = FEATURE_NAMES
        self._is_trained = False

        if model_path and os.path.exists(model_path):
            self.load(model_path)

    def train(self, X: np.ndarray, y: np.ndarray, params: dict = None) -> dict:
        """Train XGBoost classifier on sample features vs correctness labels."""
        if not HAS_XGB:
            raise ImportError("xgboost not installed. Run: pip install xgboost")

        if params is None:
            params = {
                'n_estimators': 300,
                'max_depth': 5,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 3,
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'early_stopping_rounds': 30,
                'random_state': 42,
            }

        dtrain = xgb.DMatrix(X, label=y, feature_names=self.feature_names)

        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=params.pop('n_estimators', 300),
            evals=[(dtrain, 'train')],
            early_stopping_rounds=params.pop('early_stopping_rounds', 30),
            verbose_eval=False,
        )
        self._is_trained = True

        # Feature importance
        importance = self.model.get_score(importance_type='gain')
        return importance

    def predict_proba(self, features: np.ndarray) -> float:
        """Predict probability that a sample is directionally correct."""
        if not self._is_trained:
            return 0.5
        d = xgb.DMatrix(features.reshape(1, -1), feature_names=self.feature_names)
        return float(self.model.predict(d)[0])

    def select_best(self, samples: list) -> dict:
        """
        From a list of Kronos samples, pick the best one.

        Returns:
            dict with:
                - best_sample: the selected sample dict
                - confidence: P(correct) of the selected sample
                - all_probs: list of (sample_id, prob) for all samples
                - avg_prob: average P(correct) across all samples
                - decision: BUY/SELL/HOLD based on best sample
                - confidence_adjusted: confidence × (avg_prob / 0.5) — penalized if most samples disagree
        """
        if not self._is_trained or not samples:
            # Fallback: pick sample with strongest signal (largest |net_change_pct|)
            best = max(samples, key=lambda s: abs(s['net_change_pct']))
            return {
                'best_sample': best,
                'confidence': min(abs(best['net_change_pct']) / 0.3, 1.0),
                'all_probs': [(s['sample_id'], 0.5) for s in samples],
                'avg_prob': 0.5,
                'decision': best['direction'],
                'confidence_adjusted': 0.5,
            }

        probs = []
        for s in samples:
            fv = extract_feature_vector(s)
            prob = self.predict_proba(fv)
            probs.append((s['sample_id'], prob, s))

        # Pick sample with highest P(correct)
        best_id, best_prob, best_sample = max(probs, key=lambda x: x[1])

        # Average probability across all samples (consensus signal)
        avg_prob = float(np.mean([p[1] for p in probs]))

        # Weight confidence: high if best sample is confident AND avg confirms
        conf = best_prob * (avg_prob / 0.5)  # boost if consensus > 0.5
        conf = min(1.0, max(0.0, conf))

        # Direction from best sample
        direction = best_sample['direction']

        return {
            'best_sample': best_sample,
            'confidence': round(conf, 3),
            'all_probs': [(pid, round(p, 3)) for pid, p, _ in probs],
            'avg_prob': round(avg_prob, 3),
            'best_prob': round(best_prob, 3),
            'decision': 'BUY' if direction == 'BULLISH' else 'SELL' if direction == 'BEARISH' else 'HOLD',
            'confidence_adjusted': round(conf, 3),
            'net_change': best_sample['net_change_pct'],
        }

    def save(self, path: str):
        """Save trained model."""
        if self.model is None:
            raise ValueError("No model to save")
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        self.model.save_model(path)

    def load(self, path: str):
        """Load trained model."""
        if not HAS_XGB:
            raise ImportError("xgboost not installed")
        self.model = xgb.Booster()
        self.model.load_model(path)
        self._is_trained = True

    @property
    def is_trained(self) -> bool:
        return self._is_trained
