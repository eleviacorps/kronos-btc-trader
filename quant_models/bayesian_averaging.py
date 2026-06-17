"""
Bayesian Model Averaging — Adaptive weighting of strategy predictions
based on recent performance.

Each strategy in the Kronos ensemble gets a weight proportional to its
recent Sharpe / win rate / PnL. Models that have been wrong lately
get downweighted automatically. This replaces the fixed consensus weights.

How it hybridizes with Kronos:
  - Each strategy tracks its last N outcomes (win/loss)
  - BMA computes posterior weight: P(strategy | recent results)
  - Weight decays exponentially: older results matter less
  - Final signal = weighted vote of all strategies

Usage:
  bma = BayesianModelAveraging(half_life=20)
  # After each trade closes:
  bma.record_outcome("VWAP", True)   # win
  bma.record_outcome("kronos_antitrend", False)  # loss
  # Get current weights:
  weights = bma.get_weights()
"""

import numpy as np
from collections import defaultdict


class BayesianModelAveraging:
    """
    Adaptive model weighting using Bayesian posterior updates.

    Each strategy maintains a Beta posterior over its "true win rate".
    Weight = expected win rate from posterior, decayed by recency.
    """

    def __init__(self, half_life: int = 20, min_weight: float = 0.05):
        """
        Args:
            half_life: number of trades after which prior weight halves
            min_weight: minimum weight any strategy can have (prevents starvation)
        """
        self.half_life = half_life
        self.min_weight = min_weight
        # Beta prior: alpha=1, beta=1 (uniform)
        self.alphas = defaultdict(lambda: 1.0)
        self.betas = defaultdict(lambda: 1.0)
        self._outcomes = defaultdict(list)  # strategy -> [(was_win, decay_factor)]
        self._total_trades = 0

    def record_outcome(self, strategy_name: str, was_win: bool, age: int = 0):
        """
        Record a trade outcome for a strategy.

        Args:
            strategy_name: name matching the strategy
            was_win: True if profitable, False if loss
            age: how many trades ago this happened (0 = just now)
        """
        # Decay factor: older outcomes matter less
        decay = np.exp(-age / self.half_life) if self.half_life > 0 else 1.0

        self._outcomes[strategy_name].append((was_win, decay))
        self._total_trades += 1

        # Update Beta posterior
        if was_win:
            self.alphas[strategy_name] += 1.0 * decay
        else:
            self.betas[strategy_name] += 1.0 * decay

    def get_weight(self, strategy_name: str) -> float:
        """Get current posterior weight for a strategy."""
        expected_wr = self.alphas[strategy_name] / (
            self.alphas[strategy_name] + self.betas[strategy_name]
        )
        # Add small prior pull to 0.5 for strategies with few trades
        n = self.alphas[strategy_name] + self.betas[strategy_name] - 2  # subtract prior
        if n < 10:
            # Pull toward 0.5 (Jeffreys prior)
            expected_wr = (expected_wr * (n / 10)) + (0.5 * (1 - n / 10))
        return max(self.min_weight, min(1.0, expected_wr))

    def get_weights(self, strategies: list[str] = None) -> dict:
        """
        Get normalized weights for all (or specified) strategies.

        Returns:
            {strategy_name: weight} summing to 1.0
        """
        if strategies is None:
            strategies = list(self._outcomes.keys())
        if not strategies:
            return {}

        raw = {s: self.get_weight(s) for s in strategies}
        total = sum(raw.values())
        if total == 0:
            return {s: 1.0 / len(strategies) for s in strategies}

        return {s: w / total for s, w in raw.items()}

    def get_weighted_signal(self, strategy_signals: dict) -> tuple:
        """
        Weighted ensemble vote.

        Args:
            strategy_signals: {name: (signal_int, confidence)}

            signal_int: -1 (SELL), 0 (HOLD), 1 (BUY)

        Returns:
            (weighted_vote, avg_confidence, details_dict)
        """
        weights = self.get_weights(list(strategy_signals.keys()))

        total_vote = 0.0
        total_conf = 0.0
        details = {}

        for name, (sig_int, conf) in strategy_signals.items():
            w = weights.get(name, 0)
            total_vote += w * sig_int * conf
            total_conf += w * conf
            details[name] = {
                "weight": round(w, 3),
                "signal": sig_int,
                "conf": conf,
                "contribution": round(w * sig_int * conf, 3),
            }

        # Normalize final vote [-1, 1]
        norm_vote = max(-1.0, min(1.0, total_vote))
        avg_conf = total_conf / max(sum(weights.values()), 1e-9)

        return norm_vote, avg_conf, details

    def summary(self) -> str:
        """Printable summary of current weights."""
        weights = self.get_weights()
        lines = ["Bayesian Model Averaging Weights:"]
        for s, w in sorted(weights.items(), key=lambda x: -x[1]):
            wr = self.alphas[s] / (self.alphas[s] + self.betas[s])
            n = self._outcomes[s]
            lines.append(
                f"  {s:25s} weight={w:.3f}  est_WR={wr:.1%}  n={len(n)}"
            )
        return "\n".join(lines)
