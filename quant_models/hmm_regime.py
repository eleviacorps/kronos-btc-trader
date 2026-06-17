"""
HMM Regime Detector — Hidden Markov Model for market regime classification.

4 regimes (learned from data):
  0 → Trending (bull/bear persistent)
  1 → Mean-Reverting (oscillating)
  2 → High Volatility (wide range)
  3 → Low Volatility (tight range)

How it hybridizes with Kronos:
  - If HMM says mean-reverting → BOOST antitrend confidence (2x)
  - If HMM says trending → SUPPRESS antitrend, use Kronos direction directly
  - If HMM says high vol → widen TP/SL dynamically
  - If HMM says low vol → tighten TP/SL, reduce position size

Usage:
  detector = HMMRegimeDetector(n_regimes=4)
  detector.fit(price_returns)  # historical fit
  regime = detector.predict(window_of_returns)  # latest regime
  multiplier = detector.antitrend_multiplier(regime)
"""

import numpy as np
from hmmlearn import hmm


class HMMRegimeDetector:
    """HMM-based regime classifier for price returns."""

    # Regime labels — assigned by fitting; order may vary
    REGIME_NAMES = {
        0: "trending",
        1: "mean_reverting",
        2: "high_vol",
        3: "low_vol",
    }

    def __init__(self, n_regimes: int = 4, n_iter: int = 200, random_state: int = 42):
        self.n_regimes = n_regimes
        self.model = hmm.GaussianHMM(
            n_components=n_regimes,
            covariance_type="full",
            n_iter=n_iter,
            random_state=random_state,
            init_params="stmc",
            params="stmc",
        )
        self._fitted = False
        self._regime_profiles = {}  # regime_id -> {mean, std, label}

    def fit(self, returns: np.ndarray) -> "HMMRegimeDetector":
        """Fit HMM on historical returns (1D array, log or simple)."""
        X = returns.reshape(-1, 1)
        self.model.fit(X)
        self._fitted = True
        self._label_regimes(X)
        return self

    def _label_regimes(self, X: np.ndarray):
        """Label each regime by its mean return and volatility profile."""
        states = self.model.predict(X)
        for s in range(self.n_regimes):
            mask = states == s
            if mask.sum() == 0:
                self._regime_profiles[s] = {"mean": 0.0, "std": 0.0, "label": "unknown"}
                continue
            state_returns = X[mask]
            mean_r = float(np.mean(state_returns))
            std_r = float(np.std(state_returns))

            # Heuristic labeling — calibrated for 5m BTC (std ~0.0012)
            if abs(mean_r) > 0.0003 and std_r > 0.0010:
                label = "trending"
            elif abs(mean_r) < 0.0002 and std_r > 0.0020:
                label = "high_vol"
            elif abs(mean_r) < 0.0002 and std_r < 0.0006:
                label = "low_vol"
            else:
                label = "mean_reverting"

            self._regime_profiles[s] = {"mean": mean_r, "std": std_r, "label": label}

    def predict(self, returns: np.ndarray) -> int:
        """Predict regime for latest window of returns."""
        if not self._fitted:
            raise RuntimeError("HMM not fitted — call .fit() first")
        X = returns.reshape(-1, 1)
        return int(self.model.predict(X)[-1])

    def predict_proba(self, returns: np.ndarray) -> np.ndarray:
        """Get probability distribution over regimes for latest point."""
        if not self._fitted:
            raise RuntimeError("HMM not fitted — call .fit() first")
        X = returns.reshape(-1, 1)
        return self.model.predict_proba(X)[-1]

    def regime_label(self, regime_id: int) -> str:
        """Get human-readable label for a regime."""
        return self.REGIME_NAMES.get(regime_id, "unknown")

    def profile(self, regime_id: int) -> dict:
        """Get statistical profile of a regime."""
        return self._regime_profiles.get(regime_id, {})

    def antitrend_multiplier(self, regime_id: int) -> float:
        """
        How much to boost (or suppress) antitrend signal based on regime.

        Returns:
          > 1.0 → boost antitrend (mean-reverting regime)
          < 1.0 → suppress antitrend (trending regime)
          1.0   → neutral
        """
        label = self.regime_label(regime_id)
        multipliers = {
            "mean_reverting": 2.0,   # antitrend thrives here
            "trending":       0.6,   # mild suppression — follow Kronos direction
            "high_vol":       0.9,   # mild suppression — noisy but still tradeable
            "low_vol":        0.8,   # low edge — reduce conviction
            "unknown":        1.0,
        }
        return multipliers.get(label, 1.0)

    def dynamic_tp_sl_mult(self, regime_id: int) -> dict:
        """
        TP/SL adjustment factor based on regime.
        Returns {'tp': factor, 'sl': factor} to multiply base TP/SL by.
        """
        label = self.regime_label(regime_id)
        factors = {
            "mean_reverting": {"tp": 1.0,  "sl": 1.0},   # standard
            "trending":       {"tp": 1.3,  "sl": 1.2},   # wider — trends run further
            "high_vol":       {"tp": 1.5,  "sl": 1.5},   # much wider — avoid stop-hunting
            "low_vol":        {"tp": 0.7,  "sl": 0.7},   # tighter — small moves
            "unknown":        {"tp": 1.0,  "sl": 1.0},
        }
        return factors.get(label, {"tp": 1.0, "sl": 1.0})
