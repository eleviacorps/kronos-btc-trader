"""
Fusion Layer — Master combiner for Kronos + 6 quant models.

Pipeline:
  1. Data preprocessing (returns, prices, recent windows)
  2. HMM regime detection → regime_id + antitrend_mult
  3. Kalman smoothing → filtered price + divergence signal
  4. Hurst exponent → trending/mean-reverting classification + mult
  5. GARCH vol forecast → vol_ratio + TP/SL adjustment
  6. Bayesian model averaging → strategy weights (fed from trade results)
  7. Kelly position sizing → optimal size_btc
  8. Final signal assembly

     Final_confidence = base_confidence × HMM_mult × Hurst_mult × Kalman_mult
     Final_TP/SL = base_TP/SL × HMM_adj × GARCH_adj

Integration with kronos_trader.py:
  - Replace hardcoded SCALP_TP_PCT / SCALP_SL_PCT with dynamic values
  - Replace fixed position sizing with Kelly output
  - Add regime context to Hermes summary output
"""

import numpy as np
import pandas as pd

from .hmm_regime import HMMRegimeDetector
from .kalman_filter import KalmanPriceSmoother
from .hurst_exponent import HurstExponent
from .garch_vol import GARCHVolForecast
from .bayesian_averaging import BayesianModelAveraging
from .kelly_sizing import KellyPositionSizer


class QuantFusionEngine:
    """
    Master fusion engine — orchestrates all 6 quant models and
    produces a final signal with adaptive TP/SL and position sizing.
    """

    def __init__(
        self,
        capital: float = 1000.0,
        leverage: int = 200,
        # Default TP/SL (scalp mode base)
        base_tp_pct: float = 0.3,
        base_sl_pct: float = 0.2,
        # HMM config
        hmm_regimes: int = 4,
        half_life: int = 20,  # BMA half-life
    ):
        self.capital = capital
        self.leverage = leverage
        self.base_tp_pct = base_tp_pct
        self.base_sl_pct = base_sl_pct

        # Init sub-models
        self.hmm = HMMRegimeDetector(n_regimes=hmm_regimes)
        self.kalman = None  # initialized with first price
        self.hurst = HurstExponent()
        self.garch = GARCHVolForecast()
        self.bma = BayesianModelAveraging(half_life=half_life)

        # Kelly initialized lazily when we have stats
        self.kelly = None
        self._kelly_initialized = False

        # Store latest state
        self.latest = {}

    def warmup(
        self,
        df: pd.DataFrame = None,
        prices: np.ndarray = None,
        returns: np.ndarray = None,
    ):
        """Pre-fit models on historical data before live use."""
        if df is not None:
            prices = df["close"].values.astype(float)
        if prices is not None and len(prices) > 20:
            returns = np.diff(np.log(prices))
        if returns is None or len(returns) < 20:
            return  # not enough data — use defaults

        # Fit HMM
        try:
            self.hmm.fit(returns)
        except Exception:
            pass

        # Fit GARCH
        try:
            self.garch.fit(returns)
        except Exception:
            pass

        # Initialize Kalman with first price
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
            capital=self.capital,
            leverage=self.leverage,
            btc_price=btc_price,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
        )
        self._kelly_initialized = True

    def analyze(
        self,
        df: pd.DataFrame,
        kronos_direction: str,
        kronos_net_pct: float,
        kronos_confidence: float,
        current_price: float = None,
        strategy_signals: dict = None,
    ) -> dict:
        """
        Run full quant fusion pipeline on a single signal tick.

        Args:
            df: recent OHLCV DataFrame (at least ~100 rows)
            kronos_direction: "BULLISH", "BEARISH", "NEUTRAL"
            kronos_net_pct: Kronos predicted net change %
            kronos_confidence: base confidence from ensemble (0-1)
            current_price: latest BTC price (default: df['close'].iloc[-1])
            strategy_signals: {name: (signal_int, confidence)} for BMA

        Returns:
            dict with: decision, confidence, tp_sl, size, regime, quant_details
        """
        prices = df["close"].values.astype(float)
        if current_price is None and len(prices) > 0:
            current_price = float(prices[-1])

        returns = np.diff(np.log(prices)) if len(prices) > 20 else np.array([])

        result = {
            "kronos": {
                "direction": kronos_direction,
                "net_pct": kronos_net_pct,
                "base_confidence": kronos_confidence,
            },
            "tp_sl": {
                "base_tp_pct": self.base_tp_pct,
                "base_sl_pct": self.base_sl_pct,
            },
            "size": {"btc": 0.5},  # default fallback
            "decision": "HOLD",
            "confidence": 0.0,
            "regime": {},
            "quant_details": {},
        }

        # ── 1. HMM Regime ──
        hmm_regime_id = 0
        hmm_mult = 1.0
        try:
            if len(returns) > 50 and hasattr(self.hmm.model, "monitor_"):
                hmm_regime_id = self.hmm.predict(returns[-50:])
                hmm_mult = self.hmm.antitrend_multiplier(hmm_regime_id)
                result["regime"]["hmm_id"] = hmm_regime_id
                result["regime"]["hmm_label"] = self.hmm.regime_label(hmm_regime_id)
                result["regime"]["hmm_mult"] = round(hmm_mult, 3)
                # TP/SL adjustment from HMM
                hmm_tp_sl = self.hmm.dynamic_tp_sl_mult(hmm_regime_id)
                result["tp_sl"]["hmm_tp_adj"] = hmm_tp_sl["tp"]
                result["tp_sl"]["hmm_sl_adj"] = hmm_tp_sl["sl"]
        except Exception as e:
            result["quant_details"]["hmm_error"] = str(e)

        # ── 2. Kalman Filter ──
        kalman_div = 0.0
        kalman_signal = 0.0
        try:
            if self.kalman is None and len(prices) > 0:
                self.kalman = KalmanPriceSmoother(initial_price=float(prices[0]))
                for p in prices[1:]:
                    self.kalman.update(p)
            elif self.kalman is not None and current_price is not None:
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
        except Exception as e:
            result["quant_details"]["kalman_error"] = str(e)

        # ── 3. Hurst Exponent ──
        H = 0.5
        hurst_mult = 1.0
        try:
            if len(prices) > 100:
                H = self.hurst.compute(prices[-200:])
                hurst_mult = self.hurst.antitrend_multiplier(H)
                result["regime"]["hurst_H"] = round(H, 4)
                result["regime"]["hurst_label"] = self.hurst.classify(H)
                result["regime"]["hurst_mult"] = round(hurst_mult, 3)
                # TP/SL from Hurst
                hurst_tp_sl = self.hurst.tp_sl_adjustment(H)
                result["tp_sl"]["hurst_tp_adj"] = hurst_tp_sl["tp"]
                result["tp_sl"]["hurst_sl_adj"] = hurst_tp_sl["sl"]
        except Exception as e:
            result["quant_details"]["hurst_error"] = str(e)

        # ── 4. GARCH Vol Forecast ──
        vol_ratio = 1.0
        garch_skip = False
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
                    "forecast": [
                        round(float(v), 6) for v in self.garch.forecast(steps=6)
                    ],
                }
        except Exception as e:
            result["quant_details"]["garch_error"] = str(e)

        # ── 5. Bayesian Model Averaging ──
        bma_vote = 0.0
        bma_conf = 0.0
        try:
            if strategy_signals and len(strategy_signals) > 0:
                bma_vote, bma_conf, bma_details = self.bma.get_weighted_signal(
                    strategy_signals
                )
                result["quant_details"]["bma"] = {
                    "vote": round(bma_vote, 3),
                    "confidence": round(bma_conf, 3),
                    "details": bma_details,
                }
        except Exception as e:
            result["quant_details"]["bma_error"] = str(e)

        # ── 6. Compute Final Signal ──
        # Combine antitrend multipliers from HMM + Hurst
        final_mult = hmm_mult * hurst_mult

        # Apply Kalman divergence signal as adjustment to confidence
        kalman_boost = 1.0 + abs(kalman_signal) * 0.5
        if kalman_signal < 0:
            kalman_boost = 1.0  # don't boost from negative kalman

        # Final confidence
        if bma_vote != 0 and bma_conf > 0:
            # BMA gives us a direction and confidence
            final_confidence = bma_conf * final_mult * kalman_boost
            if bma_vote > 0.1:
                decision = "BUY"
            elif bma_vote < -0.1:
                decision = "SELL"
            else:
                decision = "HOLD"
        else:
            # Fall back to Kronos antitrend with multipliers
            final_confidence = kronos_confidence * final_mult * kalman_boost
            decision = (
                "BUY" if kronos_direction == "BULLISH" and final_mult >= 1.0
                else "SELL" if kronos_direction == "BEARISH" and final_mult >= 1.0
                else "HOLD"
            )
            # Override with antitrend signal if HMM+Hurst strongly suggests mean reversion
            if hmm_mult >= 2.0 or H < 0.4:
                # Mean-reverting regime — invert Kronos
                if kronos_direction == "BULLISH":
                    decision = "SELL"
                elif kronos_direction == "BEARISH":
                    decision = "BUY"
                final_confidence = kronos_confidence * final_mult

        # GARCH vol override: skip if extreme vol
        if garch_skip and vol_ratio > 2.0:
            decision = "HOLD"
            result["decision"] = "HOLD"
            result["confidence"] = 0.0
            result["reason"] = f"GARCH extreme vol {vol_ratio:.1f}x — skip"

        # Clamp confidence
        final_confidence = max(0.0, min(1.0, final_confidence))

        result["decision"] = decision
        result["confidence"] = round(final_confidence, 3)
        result["regime"]["final_antitrend_mult"] = round(final_mult, 3)
        result["regime"]["vol_ratio"] = round(vol_ratio, 3)

        # ── 7. Dynamic TP/SL ──
        # Combine HMM + Hurst + GARCH adjustments
        tp_adj = (
            result["tp_sl"].get("hmm_tp_adj", 1.0)
            * result["tp_sl"].get("hurst_tp_adj", 1.0)
            * result["tp_sl"].get("garch_tp_adj", 1.0)
        )
        sl_adj = (
            result["tp_sl"].get("hmm_sl_adj", 1.0)
            * result["tp_sl"].get("hurst_sl_adj", 1.0)
            * result["tp_sl"].get("garch_sl_adj", 1.0)
        )
        result["tp_sl"]["final_tp_pct"] = round(self.base_tp_pct * tp_adj, 3)
        result["tp_sl"]["final_sl_pct"] = round(self.base_sl_pct * sl_adj, 3)

        # ── 8. Kelly Position Sizing ──
        try:
            if self._kelly_initialized and self.kelly is not None:
                self.kelly.btc_price = current_price or self.kelly.btc_price
                size_result = self.kelly.compute_size(
                    confidence=final_confidence, vol_ratio=vol_ratio
                )
                result["size"] = size_result
            else:
                # Default sizing (confidence-scaled)
                result["size"] = {
                    "size_btc": round(0.5 + (final_confidence * 0.5), 4),
                    "manual": True,
                }
        except Exception as e:
            result["quant_details"]["kelly_error"] = str(e)

        # ── 9. Kalman divergence overlay ──
        # If Kalman strongly diverges from Kronos, it can override decision
        if abs(kalman_signal) >= 0.5 and decision == kronos_direction:
            # Kalman says one thing, Kronos says another, and we're following Kronos
            # Check if we should flip
            net_signal = kronos_confidence * final_mult + kalman_signal * 0.3
            if net_signal < 0 and decision == "BUY":
                decision = "SELL"
                result["decision"] = "SELL"
                result["reason"] = (
                    f"Kalman divergence {kalman_div:.2f}% overrides BUY"
                )
            elif net_signal > 0 and decision == "SELL":
                decision = "BUY"
                result["decision"] = "BUY"
                result["reason"] = (
                    f"Kalman divergence {kalman_div:.2f}% overrides SELL"
                )

        result["decision"] = decision
        self.latest = result
        return result

    def record_trade_outcome(self, strategy_name: str, was_win: bool, age: int = 0):
        """Feed trade results back to BMA and Kelly."""
        self.bma.record_outcome(strategy_name, was_win, age)
        if self._kelly_initialized and self.kelly is not None:
            self.kelly.record_outcome(was_win)

    def summary(self) -> str:
        """Printable summary of latest analysis."""
        if not self.latest:
            return "No analysis run yet."
        l = self.latest
        lines = [
            "═══════════════════════════════════════",
            "  QUANT FUSION ENGINE — SUMMARY",
            "═══════════════════════════════════════",
            f"  Decision:     {l.get('decision', 'N/A')}",
            f"  Confidence:   {l.get('confidence', 0):.3f}",
            "",
            f"  TP:           {l['tp_sl']['final_tp_pct']:.2f}%",
            f"  SL:           {l['tp_sl']['final_sl_pct']:.2f}%",
            f"  Size:         {l['size']['size_btc']:.4f} BTC",
            "",
            "  ── Regime ──",
            f"  HMM label:    {l['regime'].get('hmm_label', 'N/A')}",
            f"  Hurst H:      {l['regime'].get('hurst_H', 0.5):.4f}",
            f"  Hurst label:  {l['regime'].get('hurst_label', 'N/A')}",
            f"  Vol ratio:    {l['regime'].get('vol_ratio', 1.0):.2f}x",
            f"  Antitrend mult: {l['regime'].get('final_antitrend_mult', 1.0):.3f}x",
        ]
        if l.get("quant_details", {}).get("kalman"):
            k = l["quant_details"]["kalman"]
            lines.extend([
                "",
                "  ── Kalman ──",
                f"  Filtered:     ${k['filtered_price']:.2f}",
                f"  Divergence:   {k['divergence_pct']:.4f}%",
                f"  Signal:       {k['signal']:.3f}",
            ])
        if l.get("reason"):
            lines.append(f"\n  Reason: {l['reason']}")
        lines.append("═══════════════════════════════════════")
        return "\n".join(lines)
