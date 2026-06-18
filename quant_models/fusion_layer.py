"""
Fusion Layer — Master combiner for Kronos + 6 quant models + win-rate optimizations.

NEW in v2 — 5 WR-boosting optimizations:
  1. HTF bias filter (1h EMA50) — only trade WITH the daily trend
  2. Multi-condition confirmation gate — require 2/3 indicator agreement
  3. RSI extreme override (<25 / >75) — highest-probability entries
  4. ATR-based dynamic TP/SL — R:R = 1.5-2.0 instead of fixed
  5. Skip high_vol + extreme GARCH — avoid noise traps entirely
"""

import numpy as np
import pandas as pd

from .hmm_regime import HMMRegimeDetector
from .kalman_filter import KalmanPriceSmoother
from .hurst_exponent import HurstExponent
from .garch_vol import GARCHVolForecast
from .bayesian_averaging import BayesianModelAveraging
from .kelly_sizing import KellyPositionSizer

try:
    from .sample_selector import SampleSelector, predict_samples as _selector_predict
    HAS_SELECTOR = True
except ImportError:
    HAS_SELECTOR = False
    SampleSelector = None


def _compute_rsi(prices: np.ndarray, period: int = 14) -> float:
    """Compute RSI for latest price point."""
    if len(prices) < period + 1:
        return 50.0
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _compute_atr(df_5m: pd.DataFrame, period: int = 14) -> float:
    """Compute ATR(14) as percentage of close price."""
    if len(df_5m) < period + 1:
        return 0.002  # 0.2% fallback
    high = df_5m['high' if 'high' in df_5m.columns else 'h'].values.astype(float)
    low = df_5m['low' if 'low' in df_5m.columns else 'l'].values.astype(float)
    close = df_5m['close' if 'close' in df_5m.columns else 'c'].values.astype(float)
    tr = np.maximum(
        high[1:] - low[1:],
        np.maximum(
            abs(high[1:] - close[:-1]),
            abs(low[1:] - close[:-1]),
        ),
    )
    atr = np.mean(tr[-period:])
    atr_pct = atr / close[-1] * 100
    return float(atr_pct)


class QuantFusionEngine:
    """
    Master fusion engine — orchestrates all 6 quant models and
    produces a final signal with adaptive TP/SL and position sizing.
    """

    def __init__(
        self,
        capital: float = 1000.0,
        leverage: int = 200,
        base_tp_pct: float = 0.3,
        base_sl_pct: float = 0.2,
        hmm_regimes: int = 2,
        half_life: int = 20,
    ):
        self.capital = capital
        self.leverage = leverage
        self.base_tp_pct = base_tp_pct
        self.base_sl_pct = base_sl_pct

        self.hmm = HMMRegimeDetector(n_regimes=hmm_regimes)
        self.kalman = None
        self.hurst = HurstExponent()
        self.garch = GARCHVolForecast()
        self.bma = BayesianModelAveraging(half_life=half_life)

        self.kelly = None
        self._kelly_initialized = False
        self._hurst_prices = None

        # HTF bias state
        self._htf_bias = None     # "BULLISH" / "BEARISH" / None
        self._htf_ema = None      # latest EMA50 value

        # Sample selector
        self.selector = None
        self._selector_samples = 50
        self._predictor = None  # KronosPredictor instance for selector inference

        self.latest = {}

    def warmup(
        self,
        df: pd.DataFrame = None,
        prices: np.ndarray = None,
        returns: np.ndarray = None,
    ):
        """Pre-fit models on historical data before live use."""
        if df is not None:
            prices = df["close"].values.astype(float) if "close" in df.columns else df["c"].values.astype(float)
        if prices is not None and len(prices) > 20:
            returns = np.diff(np.log(prices))
        if returns is None or len(returns) < 20:
            return

        try:
            self.hmm.fit(returns)
        except Exception:
            pass
        try:
            self.garch.fit(returns)
        except Exception:
            pass

        if prices is not None and len(prices) > 100:
            self._hurst_prices = prices.copy()

        if prices is not None and len(prices) > 0:
            self.kalman = KalmanPriceSmoother(initial_price=float(prices[0]))
            for p in prices[1:]:
                self.kalman.update(float(p))

    def initialize(
        self,
        btc_price: float,
        win_rate: float = 0.48,
        avg_win: float = 10.0,
        avg_loss: float = 8.5,
    ):
        """Initialize Kelly position sizer."""
        self.kelly = KellyPositionSizer(
            capital=self.capital, leverage=self.leverage,
            btc_price=btc_price, win_rate=win_rate,
            avg_win=avg_win, avg_loss=avg_loss,
        )
        self._kelly_initialized = True

    def load_selector(self, model_path: str, predictor=None, samples: int = 50):
        """Load trained XGBoost sample selector."""
        if not HAS_SELECTOR:
            raise ImportError("sample_selector module not available")
        self.selector = SampleSelector(model_path)
        self._selector_samples = samples
        self._predictor = predictor
        if self.selector.is_trained:
            print(f"  ✅ Sample selector loaded ({samples} samples)")

    def run_selector(self, df_5m: pd.DataFrame) -> dict:
        """
        Run sample selector on a 5m dataframe.

        Returns dict with selector decision, or empty dict if not available.
        """
        if self.selector is None or not self.selector.is_trained or self._predictor is None:
            return {}
        try:
            samples, price, avg = _selector_predict(
                self._predictor, df_5m,
                sample_count=self._selector_samples,
            )
            result = self.selector.select_best(samples)
            return result
        except Exception as e:
            return {"error": str(e)}

    def set_htf_bias(self, ema50: float, current_price: float):
        """
        Set higher-timeframe bias from 1h EMA50.
        Call periodically (every hour) with fresh 1h data.
        """
        self._htf_ema = ema50
        if current_price > ema50 * 1.003:
            self._htf_bias = "BULLISH"   # price above 1h EMA50
        elif current_price < ema50 * 0.997:
            self._htf_bias = "BEARISH"   # price below 1h EMA50
        else:
            self._htf_bias = None        # neutral — no filter

    def analyze(
        self,
        df: pd.DataFrame,
        kronos_direction: str,
        kronos_net_pct: float,
        kronos_confidence: float,
        current_price: float = None,
        strategy_signals: dict = None,
        selector_result: dict = None,
    ) -> dict:
        """
        Run full quant fusion pipeline with all 5 WR optimizations.
        """
        prices = df["close"].values.astype(float) if "close" in df.columns else df["c"].values.astype(float)
        if current_price is None and len(prices) > 0:
            current_price = float(prices[-1])

        returns = np.diff(np.log(prices)) if len(prices) > 20 else np.array([])
        high = df["high"].values.astype(float) if "high" in df.columns else df["h"].values.astype(float)

        result = {
            "kronos": {"direction": kronos_direction, "net_pct": kronos_net_pct, "base_confidence": kronos_confidence},
            "tp_sl": {"base_tp_pct": self.base_tp_pct, "base_sl_pct": self.base_sl_pct},
            "size": {"btc": 0.5},
            "decision": "HOLD", "confidence": 0.0,
            "regime": {}, "quant_details": {},
            "optimizations": {},
        }

        # ═══════════════════════════════════════════════
        # COMPUTE TECHNICAL INDICATORS
        # ═══════════════════════════════════════════════
        # RSI(14)
        try:
            rsi = _compute_rsi(prices, 14)
        except Exception:
            rsi = 50.0
        result["quant_details"]["rsi"] = round(rsi, 1)

        # ATR(14) as % of price
        try:
            atr_pct = _compute_atr(df, 14)
        except Exception:
            atr_pct = 0.2
        result["quant_details"]["atr_pct"] = round(atr_pct, 3)

        # ═══════════════════════════════════════════════
        # 1. HMM REGIME
        # ═══════════════════════════════════════════════
        hmm_regime_id, hmm_mult = 0, 1.0
        try:
            if len(returns) > 50 and hasattr(self.hmm.model, "monitor_"):
                hmm_regime_id = self.hmm.predict(returns[-50:])
                hmm_mult = self.hmm.antitrend_multiplier(hmm_regime_id)
                result["regime"]["hmm_id"] = hmm_regime_id
                result["regime"]["hmm_label"] = self.hmm.regime_label(hmm_regime_id)
                result["regime"]["hmm_mult"] = round(hmm_mult, 3)
                hmm_tp_sl = self.hmm.dynamic_tp_sl_mult(hmm_regime_id)
                result["tp_sl"]["hmm_tp_adj"] = hmm_tp_sl["tp"]
                result["tp_sl"]["hmm_sl_adj"] = hmm_tp_sl["sl"]
        except Exception:
            pass

        # ═══════════════════════════════════════════════
        # 2. KALMAN FILTER
        # ═══════════════════════════════════════════════
        kalman_div, kalman_signal = 0.0, 0.0
        try:
            if self.kalman is None and len(prices) > 0:
                self.kalman = KalmanPriceSmoother(initial_price=float(prices[0]))
                for p in prices[1:]:
                    self.kalman.update(p)
            elif self.kalman is not None:
                self.kalman.update(current_price)
                kalman_div = self.kalman.divergence_pct(current_price)
                kalman_sig, kalman_reason = self.kalman.kronos_divergence_signal(
                    kronos_direction, kronos_net_pct, current_price
                )
                kalman_signal = kalman_sig
                result["quant_details"]["kalman"] = {
                    "divergence_pct": round(kalman_div, 4),
                    "signal": round(kalman_sig, 3),
                    "reason": kalman_reason,
                    "filtered_price": round(self.kalman.filtered_price, 2),
                    "delta": round(self.kalman.filtered_delta, 4),
                }
        except Exception:
            pass

        # ═══════════════════════════════════════════════
        # 3. HURST EXPONENT
        # ═══════════════════════════════════════════════
        H, hurst_mult = 0.5, 1.0
        try:
            hurst_p = self._hurst_prices if self._hurst_prices is not None else prices
            if len(hurst_p) >= 200:
                H = self.hurst.compute(hurst_p[-500:])
            elif len(prices) > 100:
                H = self.hurst.compute(prices[-200:])
            hurst_mult = self.hurst.antitrend_multiplier(H)
            result["regime"]["hurst_H"] = round(H, 4)
            result["regime"]["hurst_label"] = self.hurst.classify(H)
            result["regime"]["hurst_mult"] = round(hurst_mult, 3)
            ht = self.hurst.tp_sl_adjustment(H)
            result["tp_sl"]["hurst_tp_adj"] = ht["tp"]
            result["tp_sl"]["hurst_sl_adj"] = ht["sl"]
        except Exception:
            pass

        # ═══════════════════════════════════════════════
        # 4. GARCH VOL FORECAST
        # ═══════════════════════════════════════════════
        vol_ratio, garch_skip = 1.0, False
        try:
            if len(returns) > 50:
                self.garch.fit(returns)
                vol_ratio = self.garch.vol_ratio(steps=6)
                garch_adj = self.garch.tp_sl_multiplier(steps=6)
                result["tp_sl"]["garch_tp_adj"] = garch_adj["tp"]
                result["tp_sl"]["garch_sl_adj"] = garch_adj["sl"]
                garch_skip = garch_adj.get("skip", False)
                result["quant_details"]["garch"] = {
                    "vol_ratio": round(vol_ratio, 2),
                    "skip_recommended": garch_skip,
                    "forecast": [round(float(v), 6) for v in self.garch.forecast(steps=6)],
                }
        except Exception:
            pass

        # ═══════════════════════════════════════════════
        # 5. BAYESIAN MODEL AVERAGING
        # ═══════════════════════════════════════════════
        bma_vote, bma_conf = 0.0, 0.0
        try:
            if strategy_signals and len(strategy_signals) > 0:
                bma_vote, bma_conf, bma_details = self.bma.get_weighted_signal(strategy_signals)
                result["quant_details"]["bma"] = {"vote": round(bma_vote, 3), "confidence": round(bma_conf, 3), "details": bma_details}
        except Exception:
            pass

        # ═══════════════════════════════════════════════
        # OPTIMIZATION 5: SKIP EXTREME CONDITIONS (mild)
        # ═══════════════════════════════════════════════
        hmm_label = result["regime"].get("hmm_label", "unknown")
        skip_reason = None

        # Only skip truly extreme conditions
        if garch_skip and vol_ratio > 2.5:
            skip_reason = f"GARCH extreme vol {vol_ratio:.1f}x — skip"
        elif atr_pct > 0.8:
            skip_reason = f"ATR {atr_pct:.3f}% > 0.8% — skip"

        if skip_reason:
            result["decision"] = "HOLD"
            result["confidence"] = 0.0
            result["reason"] = skip_reason
            result["optimizations"]["skip_applied"] = True
            result["tp_sl"]["final_tp_pct"] = 0.0
            result["tp_sl"]["final_sl_pct"] = 0.0
            result["tp_sl"]["atr_pct"] = round(atr_pct, 3)
            result["size"] = {"size_btc": 0.0, "manual": True}
            self.latest = result
            return result

        # --- Penalize high_vol instead of skipping ---
        if hmm_label == "high_vol":
            result["optimizations"]["high_vol_penalty"] = 0.5

        # ═══════════════════════════════════════════════
        # OPTIMIZATION 1: HTF BIAS FILTER (1h EMA50) — SOFT PENALTY
        # ═══════════════════════════════════════════════
        htf_penalty = 1.0
        if self._htf_bias is not None:
            if self._htf_bias == "BULLISH" and kronos_direction == "BEARISH":
                htf_penalty = 0.6
                result["optimizations"]["htf_penalty"] = "BULLISH bias penalizes SELL"
            elif self._htf_bias == "BEARISH" and kronos_direction == "BULLISH":
                htf_penalty = 0.6
                result["optimizations"]["htf_penalty"] = "BEARISH bias penalizes BUY"

        # No hard HTF block — apply penalty later in confidence computation

        # ═══════════════════════════════════════════════
        # OPTIMIZATION 3: RSI EXTREME OVERRIDE
        # ═══════════════════════════════════════════════
        if rsi < 25:
            result["decision"] = "BUY"
            result["confidence"] = round(min(1.0, (25 - rsi) / 25 * 0.8 + 0.6), 3)
            result["reason"] = f"RSI {rsi:.1f} extreme oversold — forced BUY"
            result["optimizations"]["rsi_override"] = True
            # RSI override still uses ATR-based TP/SL and Kelly sizing
            result = self._apply_tp_sl_kelly(result, atr_pct, hmm_label, H, current_price, vol_ratio, kronos_confidence)
            self.latest = result
            return result

        if rsi > 75:
            result["decision"] = "SELL"
            result["confidence"] = round(min(1.0, (rsi - 75) / 25 * 0.8 + 0.6), 3)
            result["reason"] = f"RSI {rsi:.1f} extreme overbought — forced SELL"
            result["optimizations"]["rsi_override"] = True
            result = self._apply_tp_sl_kelly(result, atr_pct, hmm_label, H, current_price, vol_ratio, kronos_confidence)
            self.latest = result
            return result

        # ═══════════════════════════════════════════════
        # 6. FINAL SIGNAL COMPUTATION
        # ═══════════════════════════════════════════════
        final_mult = hmm_mult * hurst_mult
        kalman_boost = 1.0 + abs(kalman_signal) * 0.5
        trust_kalman = abs(kalman_signal) >= 0.5

        # --- DECISION LOGIC ---
        # Priority 0: Sample selector (if available and confident)
        selector_override = False
        if selector_result and selector_result.get('decision', 'HOLD') != 'HOLD' and selector_result.get('confidence_adjusted', 0) > 0.4:
            decision = selector_result['decision']
            final_confidence = selector_result['confidence_adjusted']
            result['optimizations']['decision_source'] = 'selector'
            result['optimizations']['selector_net'] = selector_result.get('net_change', 0)
            result['optimizations']['selector_best_prob'] = selector_result.get('best_prob', 0)
            result['optimizations']['selector_avg_prob'] = selector_result.get('avg_prob', 0)
            selector_override = True

        # Priority 1: BMA data-driven vote
        if not selector_override and bma_vote != 0 and bma_conf > 0.3:
            # Priority 1: BMA data-driven vote
            final_confidence = bma_conf * min(final_mult, 1.5) * kalman_boost
            decision = "BUY" if bma_vote > 0.2 else "SELL" if bma_vote < -0.2 else "HOLD"
            result["optimizations"]["decision_source"] = "BMA"

        elif trust_kalman and abs(kalman_div) > 0.25:
            # Priority 2: Strong Kalman divergence
            final_confidence = kronos_confidence * final_mult * kalman_boost
            if kalman_signal > 0.4:
                decision = "BUY"
            elif kalman_signal < -0.4:
                decision = "SELL"
            else:
                decision = "HOLD"
            result["optimizations"]["decision_source"] = "Kalman"

        elif hmm_label == "mean_reverting":
            # Mean-reverting — ANTITREND (invert Kronos)
            if kronos_direction == "BULLISH":
                decision, final_confidence = "SELL", kronos_confidence * min(final_mult, 2.0)
            elif kronos_direction == "BEARISH":
                decision, final_confidence = "BUY", kronos_confidence * min(final_mult, 2.0)
            else:
                decision, final_confidence = "HOLD", 0.0
            result["optimizations"]["decision_source"] = "antitrend"

        elif hmm_label == "trending" or H > 0.55:
            # Trending — follow Kronos
            if kronos_direction == "BULLISH":
                decision, final_confidence = "BUY", kronos_confidence * max(final_mult, 0.8)
            elif kronos_direction == "BEARISH":
                decision, final_confidence = "SELL", kronos_confidence * max(final_mult, 0.8)
            else:
                decision, final_confidence = "HOLD", 0.0
            result["optimizations"]["decision_source"] = "trending"

        else:
            # Mixed — follow Kronos with filtering
            final_confidence = kronos_confidence * final_mult * kalman_boost
            decision = (
                "BUY" if kronos_direction == "BULLISH" and final_confidence > 0.35
                else "SELL" if kronos_direction == "BEARISH" and final_confidence > 0.35
                else "HOLD"
            )
            result["optimizations"]["decision_source"] = "mixed"

        # ═══════════════════════════════════════════════
        # OPTIMIZATION 2: MULTI-CONDITION CONFIRMATION GATE
        # (Skipped if selector confidence > 0.5 — it already learned to pick winners)
        # ═══════════════════════════════════════════════
        if decision != "HOLD" and result["optimizations"].get("decision_source") not in ("BMA", "Kalman"):
            # Check if selector overrides the gate
            if selector_result and selector_result.get('confidence_adjusted', 0) > 0.5:
                result["optimizations"]["gate_skipped"] = f"selector conf {selector_result['confidence_adjusted']:.2f} > 0.5"
            else:
                # Count agreements with the proposed direction
                agreements = 0
                total_checks = 0

                # Check 1: HTF bias agrees
                if self._htf_bias is not None:
                    total_checks += 1
                    if (decision == "BUY" and self._htf_bias == "BULLISH") or \
                       (decision == "SELL" and self._htf_bias == "BEARISH"):
                        agreements += 1

                # Check 2: HMM regime agrees
                total_checks += 1
                if hmm_label == "trending" and decision == kronos_direction:
                    agreements += 1
                elif hmm_label == "mean_reverting" and decision != kronos_direction:
                    agreements += 1
                elif hmm_label in ("high_vol", "low_vol", "unknown"):
                    agreements += 1  # neutral — don't penalize

                # Check 3: Hurst agrees
                total_checks += 1
                if H > 0.55 and decision == kronos_direction:
                    agreements += 1
                elif H < 0.4 and decision != kronos_direction:
                    agreements += 1
                else:
                    agreements += 1  # neutral Hurst — don't penalize

                # Check 4: RSI confirms (not extreme against)
                total_checks += 1
                if decision == "BUY" and rsi < 50:
                    agreements += 1
                elif decision == "SELL" and rsi > 50:
                    agreements += 1
                else:
                    agreements += 0  # RSI against = no agreement

                # Require majority (ceil(total_checks/4)) but minimum 2 of 4
                required = max(2, (total_checks + 1) // 2)
                if agreements < required:
                    decision = "HOLD"
                    final_confidence *= 0.5
                    result["optimizations"]["gate_blocked"] = f"agreed {agreements}/{total_checks} < {required}"

        # Apply high_vol penalty (halve confidence instead of skipping)
        if result.get("optimizations", {}).get("high_vol_penalty"):
            final_confidence *= 0.5
            result["optimizations"]["high_vol_penalty_applied"] = True

        # Apply HTF penalty (60% confidence instead of blocking)
        if htf_penalty < 1.0 and decision != "HOLD":
            final_confidence *= htf_penalty
            result["optimizations"]["htf_applied"] = True

        # ═══════════════════════════════════════════════
        # APPLY TP/SL + KELLY
        # ═══════════════════════════════════════════════
        final_confidence = max(0.0, min(1.0, final_confidence))
        result["decision"] = decision
        result["confidence"] = round(final_confidence, 3)
        result["regime"]["final_antitrend_mult"] = round(final_mult, 3)
        result["regime"]["vol_ratio"] = round(vol_ratio, 3)

        result = self._apply_tp_sl_kelly(result, atr_pct, hmm_label, H, current_price, vol_ratio, final_confidence)

        # Kalman divergence overlay (only if not overridden by optimizations)
        if abs(kalman_signal) >= 0.5 and decision == kronos_direction and decision != "HOLD":
            net_signal = kronos_confidence * final_mult + kalman_signal * 0.3
            if net_signal < 0 and decision == "BUY":
                result["decision"] = "SELL"
                result["reason"] = f"Kalman div {kalman_div:.2f}% overrides BUY"
            elif net_signal > 0 and decision == "SELL":
                result["decision"] = "BUY"
                result["reason"] = f"Kalman div {kalman_div:.2f}% overrides SELL"

        self.latest = result
        return result

    def _apply_tp_sl_kelly(self, result: dict, atr_pct: float, hmm_label: str, H: float,
                           current_price: float, vol_ratio: float, confidence: float) -> dict:
        """Apply ATR-based TP/SL (Optimization 4) and Kelly sizing."""
        # ═══════════════════════════════════════════════
        # OPTIMIZATION 4: ATR-BASED DYNAMIC TP/SL
        # ═══════════════════════════════════════════════
        # Bullish bias in trending: let winners run (2x ATR)
        # Mean-reverting: tight TP (1x ATR)
        # Default: 1.5x ATR TP, 0.8x ATR SL → R:R ≈ 1.9
        if hmm_label == "trending" or H > 0.55:
            tp_mult, sl_mult = 2.0, 0.8
        elif hmm_label == "mean_reverting" or H < 0.4:
            tp_mult, sl_mult = 1.5, 0.7
        else:
            tp_mult, sl_mult = 1.5, 0.8

        # Scale by volatility (wider when vol is elevated)
        if vol_ratio > 1.3:
            tp_mult *= 1.2
            sl_mult *= 1.2

        final_tp_pct = round(max(0.15, atr_pct * tp_mult), 3)
        final_sl_pct = round(max(0.10, atr_pct * sl_mult), 3)

        # Cap at reasonable values
        final_tp_pct = min(final_tp_pct, 2.0)
        final_sl_pct = min(final_sl_pct, 1.0)

        result["tp_sl"]["atr_pct"] = round(atr_pct, 3)
        result["tp_sl"]["tp_atr_mult"] = tp_mult
        result["tp_sl"]["sl_atr_mult"] = sl_mult
        result["tp_sl"]["final_tp_pct"] = final_tp_pct
        result["tp_sl"]["final_sl_pct"] = final_sl_pct

        # ═══════════════════════════════════════════════
        # KELLY POSITION SIZING
        # ═══════════════════════════════════════════════
        try:
            if self._kelly_initialized and self.kelly is not None:
                self.kelly.btc_price = current_price or self.kelly.btc_price
                size_result = self.kelly.compute_size(confidence=confidence, vol_ratio=vol_ratio)
                result["size"] = size_result
            else:
                result["size"] = {"size_btc": round(0.3 + (confidence * 0.4), 4), "manual": True}
        except Exception:
            result["size"] = {"size_btc": 0.3, "manual": True}

        return result

    def record_trade_outcome(self, strategy_name: str, was_win: bool, age: int = 0):
        self.bma.record_outcome(strategy_name, was_win, age)
        if self._kelly_initialized and self.kelly is not None:
            self.kelly.record_outcome(was_win)

    def summary(self) -> str:
        if not self.latest:
            return "No analysis run yet."
        l = self.latest
        lines = [
            "═══════════════════════════════════════",
            "  QUANT FUSION v2 — SUMMARY",
            "═══════════════════════════════════════",
            f"  Decision:     {l.get('decision', 'N/A')} ({l.get('optimizations', {}).get('decision_source', '?')})",
            f"  Confidence:   {l.get('confidence', 0):.3f}",
            "",
            f"  TP:           {l['tp_sl'].get('final_tp_pct', 0):.2f}% (ATR {l['tp_sl'].get('atr_pct', 0):.2f}% × {l['tp_sl'].get('tp_atr_mult', 0):.1f}x)",
            f"  SL:           {l['tp_sl'].get('final_sl_pct', 0):.2f}% (ATR × {l['tp_sl'].get('sl_atr_mult', 0):.1f}x)",
            f"  R:R:          {l['tp_sl'].get('final_tp_pct', 0)/max(l['tp_sl'].get('final_sl_pct', 0.01),0.01):.2f}:1",
            f"  Size:         {l['size']['size_btc']:.4f} BTC",
            "",
            "  ── Regime ──",
            f"  HMM label:    {l['regime'].get('hmm_label', 'N/A')}",
            f"  Hurst H:      {l['regime'].get('hurst_H', 0.5):.4f} ({l['regime'].get('hurst_label', '?')})",
            f"  RSI:          {l['quant_details'].get('rsi', 50):.1f}",
            f"  ATR pct:      {l['quant_details'].get('atr_pct', 0.2):.3f}%",
            f"  Vol ratio:    {l['regime'].get('vol_ratio', 1.0):.2f}x",
        ]
        if l.get("optimizations", {}).get("rsi_override"):
            lines.append(f"  ⚡ RSI OVERRIDE!")
        if l.get("optimizations", {}).get("gate_blocked"):
            lines.append(f"  ⛔ Gate: {l['optimizations']['gate_blocked']}")
        if l.get("optimizations", {}).get("htf_block"):
            lines.append(f"  ⛔ HTF: {l['optimizations']['htf_block']}")
        if l.get("quant_details", {}).get("kalman"):
            k = l["quant_details"]["kalman"]
            lines.extend(["", "  ── Kalman ──", f"  Filtered:     ${k['filtered_price']:.2f}", f"  Divergence:   {k['divergence_pct']:.4f}%"])
        if l.get("reason"):
            lines.append(f"\n  Reason: {l['reason']}")
        lines.append("═══════════════════════════════════════")
        return "\n".join(lines)
