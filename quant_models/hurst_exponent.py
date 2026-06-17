"""
Hurst Exponent — Classification of financial time series as trending
or mean-reverting.

H < 0.5 → mean-reverting (price pulls back to mean)
H = 0.5 → random walk (no edge)
H > 0.5 → trending (momentum persists)

How it hybridizes with Kronos:
  - Dynamic strategy selector based on Hurst value:
    H < 0.4 → mean-reverting regime → BOOST antitrend
    H > 0.6 → trending regime → SUPPRESS antitrend, use Kronos direction or trend-following
    0.4 <= H <= 0.6 → mixed → use ensemble consensus as-is

Usage:
  hurst = HurstExponent()
  h = hurst.compute(prices)
  regime = hurst.classify(h)
  mult = hurst.antitrend_multiplier(h)
"""

import numpy as np


class HurstExponent:
    """Compute Hurst exponent via R/S analysis."""

    def compute(self, prices: np.ndarray, max_lag: int = None) -> float:
        """
        Estimate Hurst exponent using R/S (rescaled range) analysis.

        Args:
            prices: 1D price array (log prices preferred for stability)
            max_lag: maximum lag to consider (default: len(prices) // 2)

        Returns:
            H value between 0 and 1
        """
        prices = np.asarray(prices, dtype=float)
        if len(prices) < 100:
            return 0.5  # insufficient data — default to random walk

        if max_lag is None:
            max_lag = len(prices) // 2

        # Log returns
        log_returns = np.diff(np.log(prices))

        # Mean-adjusted returns
        mean = np.mean(log_returns)
        deviations = log_returns - mean

        # Compute R/S for each lag
        lags = range(10, min(max_lag, len(log_returns) // 2), 5)
        rs_values = []

        for lag in lags:
            n_windows = len(deviations) // lag
            if n_windows < 1:
                continue

            rs = 0.0
            for i in range(n_windows):
                window = deviations[i * lag : (i + 1) * lag]
                # Cumulative deviate
                cum = np.cumsum(window)
                # Range
                r = np.max(cum) - np.min(cum)
                # Standard deviation
                s = np.std(window, ddof=1)
                if s > 0:
                    rs += r / s

            rs_mean = rs / n_windows
            rs_values.append(rs_mean)

        if len(rs_values) < 3:
            return 0.5

        # Log-log regression: log(R/S) = H * log(lag) + c
        log_lags = np.log(np.array(lags[:len(rs_values)]))
        log_rs = np.log(np.array(rs_values))

        A = np.vstack([log_lags, np.ones(len(log_lags))]).T
        H, _ = np.linalg.lstsq(A, log_rs, rcond=None)[0]

        return max(0.0, min(1.0, float(H)))

    def classify(self, H: float) -> str:
        """Classify H value into regime."""
        if H < 0.4:
            return "mean_reverting"
        elif H > 0.6:
            return "trending"
        else:
            return "random_walk"

    def antitrend_multiplier(self, H: float) -> float:
        """
        How much to boost/suppress antitrend based on Hurst.

        H < 0.4 (mean-reverting) → boost antitrend (2.0x at extreme)
        H > 0.6 (trending)       → suppress antitrend (0.3x)
        In between → linear interpolation
        """
        if H < 0.4:
            # Mean-reverting: strong antitrend boost
            # Scale from 1.0 at H=0.5 to 2.0 at H=0.0
            return 1.0 + (0.5 - H) * 5.0  # 2.0 at H=0.3, 3.0 at H=0.2
        elif H > 0.6:
            # Trending: suppress antitrend
            # Scale from 1.0 at H=0.5 to 0.3 at H=0.8
            return max(0.2, 1.0 - (H - 0.5) * 2.33)  # ~0.77 at 0.6, 0.3 at 0.8
        else:
            # Random walk / mixed: neutral to mild boost
            return 1.0

    def tp_sl_adjustment(self, H: float) -> dict:
        """
        TP/SL adjustment based on Hurst.

        Trending (H > 0.6): wider TP (trends run), normal SL
        Mean-reverting (H < 0.4): tighter TP (reversion is fast)
        """
        return {"tp": 1.2 if H > 0.6 else (0.8 if H < 0.4 else 1.0),
                "sl": 1.0 if H > 0.6 else (0.8 if H < 0.4 else 1.0)}
