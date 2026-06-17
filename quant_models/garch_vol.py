"""
GARCH Volatility Forecast — Predicts future volatility using GARCH(1,1).

GARCH(1,1) model:
  σ²_t = ω + α * ε²_{t-1} + β * σ²_{t-1}

How it hybridizes with Kronos:
  - Forecast next N-periods volatility → dynamic TP/SL
  - If GARCH predicts HIGH vol → widen stops to avoid stop-hunting
  - If GARCH predicts LOW vol → tighten stops, reduce position size
  - Can also be used to SKIP trades when vol is extreme (>3x normal)

Usage:
  garch = GARCHVolForecast()
  garch.fit(returns)
  forecast = garch.forecast(steps=6)  # next 6 candles
  mean_vol = garch.forecast_mean(steps=6)
  mult = garch.tp_sl_multiplier(forecast)
"""

import numpy as np
import pandas as pd


class GARCHVolForecast:
    """GARCH(1,1) volatility forecaster with simplified estimation."""

    def __init__(self, p: int = 1, q: int = 1):
        self.p = p  # GARCH lag
        self.q = q  # ARCH lag
        self.params = None  # (omega, alpha, beta)
        self._residuals = None
        self._conditional_vol = None
        self._baseline_vol = None

    def fit(self, returns: np.ndarray) -> "GARCHVolForecast":
        """
        Fit GARCH(1,1) using MLE via scipy.
        Falls back to moment estimation if MLE fails.
        """
        from scipy.optimize import minimize

        returns = np.asarray(returns, dtype=float)
        n = len(returns)

        # Compute residuals (demeaned returns)
        mu = np.mean(returns)
        residuals = returns - mu
        self._residuals = residuals

        # Initialize with moment estimates
        var = np.var(returns, ddof=1)
        # Starting params: omega, alpha, beta
        # alpha + beta < 1 for stationarity
        init_params = np.array([var * 0.05, 0.1, 0.85])

        # Bounds: omega > 0, alpha >= 0, beta >= 0, alpha+beta < 1
        bounds = [(1e-12, None), (0.0, 1.0), (0.0, 1.0)]

        def garch_likelihood(params):
            omega, alpha, beta = params
            if alpha + beta >= 1.0:
                return 1e12  # penalty

            T = len(residuals)
            sigma2 = np.full(T, var)
            # Recursive variance
            for t in range(1, T):
                sigma2[t] = omega + alpha * residuals[t-1]**2 + beta * sigma2[t-1]

            # Negative log-likelihood (Gaussian)
            likelihood = 0.5 * np.sum(
                np.log(sigma2[1:]) + residuals[1:]**2 / sigma2[1:]
            )
            return likelihood + 0.5 * (T - 1) * np.log(2 * np.pi)

        try:
            result = minimize(
                garch_likelihood,
                init_params,
                bounds=bounds,
                method="L-BFGS-B",
                options={"maxiter": 500, "ftol": 1e-8},
            )
            self.params = result.x
        except Exception:
            # Fallback to moment estimates
            self.params = np.array([var * 0.05, 0.1, 0.8])

        omega, alpha, beta = self.params

        # Compute conditional volatility series
        sigma2 = np.full(n, var)
        for t in range(1, n):
            sigma2[t] = omega + alpha * residuals[t-1]**2 + beta * sigma2[t-1]
        self._conditional_vol = np.sqrt(sigma2)
        self._baseline_vol = float(np.sqrt(np.mean(sigma2)))

        return self

    def forecast(self, steps: int = 6) -> np.ndarray:
        """
        Forecast volatility for next `steps` periods.

        Returns:
            array of forecasted volatilities (standard deviation per period)
        """
        if self.params is None:
            raise RuntimeError("GARCH not fitted — call .fit() first")

        omega, alpha, beta = self.params
        last_sigma2 = self._conditional_vol[-1]**2 if self._conditional_vol is not None else 0.0
        last_residual = self._residuals[-1]**2 if self._residuals is not None else 0.0

        forecasts = []
        sigma2 = last_sigma2
        residual2 = last_residual

        for _ in range(steps):
            sigma2 = omega + alpha * residual2 + beta * sigma2
            forecasts.append(float(np.sqrt(sigma2)))
            # Multi-step: use expected value of epsilon² = σ²
            residual2 = sigma2

        return np.array(forecasts)

    def forecast_mean(self, steps: int = 6) -> float:
        """Mean forecasted volatility over next N steps."""
        return float(np.mean(self.forecast(steps)))

    def vol_ratio(self, steps: int = 6) -> float:
        """Forecasted vol relative to historical baseline. >1 = elevated."""
        if self._baseline_vol is None or self._baseline_vol == 0:
            return 1.0
        return self.forecast_mean(steps) / self._baseline_vol

    def tp_sl_multiplier(self, steps: int = 6) -> dict:
        """
        Dynamic TP/SL multiplier based on volatility forecast.

        vol_ratio < 0.7   → low vol, tighten (0.8x)
        0.7-1.3           → normal vol (1.0x)
        1.3-2.0           → high vol, widen (1.3x)
        > 2.0             → extreme vol, widen (1.5x) AND consider skipping

        Returns {'tp': float, 'sl': float}
        """
        vr = self.vol_ratio(steps)

        if vr > 2.0:
            tp_sl = {"tp": 1.5, "sl": 1.5, "skip": True}
        elif vr > 1.3:
            tp_sl = {"tp": 1.3, "sl": 1.3, "skip": False}
        elif vr < 0.7:
            tp_sl = {"tp": 0.8, "sl": 0.8, "skip": False}
        else:
            tp_sl = {"tp": 1.0, "sl": 1.0, "skip": False}

        return tp_sl
