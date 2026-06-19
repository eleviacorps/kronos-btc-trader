"""
Kelly Position Sizer — Optimal position sizing using the Kelly Criterion.

Kelly % = W - (1-W) / (R)
  Where:
    W = win rate (probability of winning)
    R = win/loss ratio (avg win / avg loss)

Then fraction_of_capital = Kelly % × confidence × safety_reduction

How it hybridizes with Kronos:
  - Replaces the fixed 0.5/0.75/1.0 BTC size buckets
  - Dynamically sizes based on actual strategy performance
  - More wins + larger avg win → bigger position
  - Loss streak → automatically reduces size (conservative Kelly)
  - GARCH vol ratio further scales down in high vol

Usage:
  kelly = KellyPositionSizer(
    capital=1000.0,
    leverage=200,
    btc_price=62000.0,
    win_rate=0.48,
    avg_win=10.0,
    avg_loss=8.5,
  )
  size = kelly.compute_size(confidence=0.6, vol_ratio=1.2)
"""

import numpy as np


class KellyPositionSizer:
    """
    Kelly Criterion position sizing for Kronos scalper.

    Supports:
      - Standard Kelly:  fraction = W - (1-W)/R
      - Half-Kelly (default, conservative):  size = Kelly / 2
      - Confidence scaling:  size *= confidence
      - Volatility scaling:  size *= (1.0 / max(vol_ratio, 0.5))
      - Max position cap:  configurable absolute or % of capital
    """

    def __init__(
        self,
        capital: float = 1000.0,
        leverage: int = 200,
        btc_price: float = 60000.0,
        win_rate: float = 0.48,
        avg_win: float = 10.0,
        avg_loss: float = 8.5,
        kelly_fraction: float = 0.5,
        max_position_pct: float = 0.50,
        min_size_btc: float = 0.5,
        max_size_btc: float = 1.5,
    ):
        """
        Args:
            capital: account balance in USDT
            leverage: margin leverage (200x)
            btc_price: current BTC/USDT price
            win_rate: historical win rate (0.0-1.0)
            avg_win: average winning trade profit in USDT
            avg_loss: average losing trade loss in USDT (positive)
            kelly_fraction: safety factor (0.5 = half-Kelly)
            max_position_pct: max % of capital to risk per trade
            min_size_btc: minimum position size in BTC (default 0.5)
            max_size_btc: maximum position size in BTC (default 1.5)
        """
        self.capital = capital
        self.leverage = leverage
        self.btc_price = btc_price
        self.win_rate = win_rate
        self.avg_win = avg_win
        self.avg_loss = avg_loss
        self.kelly_fraction = kelly_fraction
        self.max_position_pct = max_position_pct
        self.min_size_btc = min_size_btc
        self.max_size_btc = max_size_btc

        # Risk of ruin tracker
        self._consecutive_losses = 0
        self._loss_streak_penalty = 1.0
        self._initial_capital = capital
        self._peak_capital = capital

    def compute_kelly_pct(self) -> float:
        """
        Compute optimal Kelly fraction.

        Kelly = W - (1-W)/R
        where R = avg_win / avg_loss

        Returns fraction of capital to risk (0.0-1.0), or negative if no edge.
        """
        if self.avg_loss <= 0:
            return 0.0

        R = self.avg_win / self.avg_loss

        if R <= 0:
            return 0.0

        kelly = self.win_rate - (1.0 - self.win_rate) / R

        # If negative, no edge — don't trade
        return max(0.0, kelly)

    def compute_size(
        self,
        confidence: float = 1.0,
        vol_ratio: float = 1.0,
        risk_factor: float = 1.0,
        use_kelly: bool = True,
    ) -> dict:
        """
        Compute position size in BTC. Dynamic range 0.5-1.5 BTC.

        Sizing factors:
          - Kelly % (based on win rate and avg win/loss)
          - confidence (signal quality, 0-1)
          - vol_ratio (>1 = reduce, <1 = increase)
          - equity growth (account grew → size up, account shrank → size down)
          - loss streak penalty (consecutive losses reduce size)
          - risk_factor (1.0 = normal, 0.5 = conservative, 1.5 = aggressive)

        Args:
            confidence: signal confidence (0.0-1.0)
            vol_ratio: GARCH vol ratio (>1 = elevated vol)
            risk_factor: user-specified risk appetite (0.5-1.5)
            use_kelly: if False, uses fixed % of capital instead

        Returns:
            dict with keys: size_btc, size_usd, margin, kelly_pct, details
        """
        buying_power = self.capital * self.leverage

        # Equity growth multiplier: scale with account performance
        if self._initial_capital > 0:
            equity_growth = self.capital / self._initial_capital
        else:
            equity_growth = 1.0
        # Track peak for drawdown protection
        self._peak_capital = max(self._peak_capital, self.capital)
        if self._peak_capital > 0:
            drawdown = max(0.0, 1.0 - self.capital / self._peak_capital)
        else:
            drawdown = 0.0
        # Drawdown penalty: reduce size by 2x the drawdown amount
        dd_penalty = max(0.3, 1.0 - drawdown * 2.0)

        if use_kelly:
            kelly_pct = self.compute_kelly_pct()
            if kelly_pct <= 0:
                kelly_pct = 0.01
            # Apply all scaling factors
            risk_pct = (
                kelly_pct
                * self.kelly_fraction
                * min(confidence * 1.2, 1.0)  # scale confidence up to 1.0
                * (1.0 / max(vol_ratio, 0.5))  # vol scaling
                * self._loss_streak_penalty
                * min(equity_growth, 1.5)  # cap equity growth at 1.5x
                * dd_penalty
                * risk_factor
            )
        else:
            risk_pct = self.max_position_pct * confidence * risk_factor

        # Cap at max allowed risk
        risk_pct = min(risk_pct, self.max_position_pct)

        # Convert to notional value
        notional = buying_power * risk_pct
        margin = self.capital * risk_pct

        # Convert to BTC, clamped to [0.5, 1.5]
        size_btc = notional / max(self.btc_price, 1)
        size_btc = max(self.min_size_btc, min(self.max_size_btc, size_btc))
        size_usd = size_btc * self.btc_price

        return {
            "size_btc": round(size_btc, 4),
            "size_usd": round(size_usd, 2),
            "margin": round(margin, 2),
            "risk_pct": round(risk_pct * 100, 2),
            "kelly_pct": round(self.compute_kelly_pct() * 100, 1) if use_kelly else None,
            "details": {
                "consecutive_losses": self._consecutive_losses,
                "streak_penalty": self._loss_streak_penalty,
                "confidence": confidence,
                "vol_ratio": round(vol_ratio, 2),
                "equity_growth": round(equity_growth, 3),
                "drawdown": round(drawdown * 100, 1),
                "dd_penalty": round(dd_penalty, 2),
                "risk_factor": risk_factor,
            },
        }

    def record_outcome(self, was_win: bool):
        """Update streak tracker after a trade closes."""
        if was_win:
            self._consecutive_losses = 0
            self._loss_streak_penalty = 1.0
        else:
            self._consecutive_losses += 1
            # Penalty increases with streak: 1.0 -> 0.8 -> 0.6 -> 0.4 -> 0.2
            self._loss_streak_penalty = max(0.2, 1.0 - self._consecutive_losses * 0.2)

    def update_stats(
        self,
        win_rate: float = None,
        avg_win: float = None,
        avg_loss: float = None,
        btc_price: float = None,
        capital: float = None,
    ):
        """Update statistics without creating a new instance."""
        if win_rate is not None:
            self.win_rate = win_rate
        if avg_win is not None:
            self.avg_win = avg_win
        if avg_loss is not None:
            self.avg_loss = avg_loss
        if btc_price is not None:
            self.btc_price = btc_price
        if capital is not None:
            self.capital = capital

    def set_capital(self, capital: float):
        self.capital = capital
