"""
Kalman Price Smoother — Adaptive 1D Kalman Filter for noisy price data.

Filters out 5m noise to reveal the "true" price trajectory.
Divergence between Kalman estimate and current/latest Kronos prediction
creates an edge signal.

How it hybridizes with Kronos:
  - Kalman filtered price ≈ fair value estimate
  - If Kronos says DOWN but price is ALREADY below Kalman estimate
    → mean reversion likely → STRONGER BUY signal
  - If Kronos says UP but price is ALREADY above Kalman estimate
    → mean reversion likely → STRONGER SELL signal
  - If Kronos aligns with Kalman divergence → confirms direction

Usage:
  smoother = KalmanPriceSmoother(initial_price=50000, process_noise=1e-5)
  for price in prices:
      filtered = smoother.update(price)
  divergence = smoother.divergence_pct(current_price)
"""

import numpy as np


class KalmanPriceSmoother:
    """
    1D Kalman Filter for price smoothing.

    State: [price, delta]  (price + first derivative / velocity)
    Measurement: raw price
    """

    def __init__(
        self,
        initial_price: float = 50000.0,
        initial_delta: float = 0.0,
        process_noise: float = 1e-5,
        measurement_noise: float = 1e-3,
        delta_decay: float = 0.9,
    ):
        # State vector [price, delta]
        self.x = np.array([[initial_price], [initial_delta]], dtype=float)

        # State covariance
        self.P = np.eye(2, dtype=float) * 10.0

        # State transition matrix (constant velocity model)
        # price = price + delta, delta = delta * decay
        self.F = np.array([[1.0, 1.0], [0.0, delta_decay]], dtype=float)

        # Control matrix (unused)
        self.B = np.zeros((2, 1), dtype=float)

        # Observation matrix (we observe price only)
        self.H = np.array([[1.0, 0.0]], dtype=float)

        # Process noise covariance
        self.Q = np.eye(2, dtype=float) * process_noise
        self.Q[1, 1] = process_noise * 0.1  # delta noise smaller

        # Measurement noise covariance
        self.R = np.eye(1, dtype=float) * measurement_noise

        # Track history for divergence calc
        self._history = [float(initial_price)]
        self._filtered_history = [float(initial_price)]

    def update(self, measurement: float) -> float:
        """Ingest a new price measurement, return filtered estimate."""
        z = np.array([[measurement]], dtype=float)

        # Predict
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q

        # Update
        y = z - self.H @ x_pred  # innovation
        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ np.linalg.inv(S)  # Kalman gain

        self.x = x_pred + K @ y
        self.P = (np.eye(2) - K @ self.H) @ P_pred

        filtered = float(self.x[0, 0])
        self._history.append(measurement)
        self._filtered_history.append(filtered)
        return filtered

    @property
    def filtered_price(self) -> float:
        """Current Kalman-filtered price estimate."""
        return float(self.x[0, 0])

    @property
    def filtered_delta(self) -> float:
        """Current price velocity (rate of change per step)."""
        return float(self.x[1, 0])

    def divergence_pct(self, current_price: float) -> float:
        """
        How far current price diverges from Kalman estimate, in percent.

        Positive: price ABOVE filtered estimate (overextended up)
        Negative: price BELOW filtered estimate (overextended down)
        """
        f = self.filtered_price
        if f == 0:
            return 0.0
        return ((current_price - f) / f) * 100

    def kronos_divergence_signal(
        self, kronos_direction: str, kronos_net_pct: float, current_price: float
    ) -> tuple:
        """
        Combine Kalman divergence with Kronos prediction.

        Returns:
          (signal_strength, reason)
          signal_strength: -1..1 (negative = support SELL, positive = support BUY)
          0 = neutral
        """
        div = self.divergence_pct(current_price)

        # Kronos says UP
        if kronos_direction == "BULLISH":
            if div > 0.15:
                # Price already above fair value AND Kronos says up → overextended
                # → Supports SELL (mean reversion down)
                return (-0.5, f"Kalman div +{div:.3f}% overextended UP, Kronos bullish → bearish divergence")
            elif div < -0.15:
                # Price below fair value AND Kronos says up → alignment
                # → Supports BUY
                return (0.6, f"Kalman div {div:.3f}% undervalued, Kronos bullish → bullish alignment")
            else:
                return (0.2, f"Kalman div {div:.3f}% neutral, Kronos bullish → mild bullish")

        # Kronos says DOWN
        elif kronos_direction == "BEARISH":
            if div < -0.15:
                # Price already below fair value AND Kronos says down → overextended
                # → Supports BUY (mean reversion up)
                return (0.5, f"Kalman div {div:.3f}% overextended DOWN, Kronos bearish → bullish divergence")
            elif div > 0.15:
                # Price above fair value AND Kronos says down → alignment
                # → Supports SELL
                return (-0.6, f"Kalman div +{div:.3f}% overvalued, Kronos bearish → bearish alignment")
            else:
                return (-0.2, f"Kalman div {div:.3f}% neutral, Kronos bearish → mild bearish")

        # Kronos says NEUTRAL
        else:
            if div > 0.3:
                return (-0.3, f"Kalman div +{div:.3f}% overextended, Kronos neutral → lean SELL")
            elif div < -0.3:
                return (0.3, f"Kalman div {div:.3f}% undervalued, Kronos neutral → lean BUY")
            return (0.0, f"Kalman div {div:.3f}% neutral, Kronos neutral → HOLD")

    def reset(self, initial_price: float):
        """Reset filter state."""
        self.x = np.array([[initial_price], [0.0]], dtype=float)
        self.P = np.eye(2, dtype=float) * 10.0
        self._history = [initial_price]
        self._filtered_history = [initial_price]
