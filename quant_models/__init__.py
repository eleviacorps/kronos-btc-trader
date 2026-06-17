"""
quant_models — Quant layer for Kronos hybrid trading system.

Six models feeding into a fusion layer for final signal:
  1. HMM         → regime detection (trending/mean-reverting/high-vol/low-vol)
  2. Kalman      → price smoothing + divergence detection
  3. Hurst       → trending vs mean-reverting classification
  4. GARCH       → volatility forecast for adaptive TP/SL
  5. Bayesian    → adaptive model weighting by recent accuracy
  6. Kelly       → optimal position sizing

Call order: HMM → Kalman → Hurst → GARCH → Bayesian → Kelly
Then fusion_layer.py combines everything into a final signal.
"""

from .hmm_regime import HMMRegimeDetector
from .kalman_filter import KalmanPriceSmoother
from .hurst_exponent import HurstExponent
from .garch_vol import GARCHVolForecast
from .bayesian_averaging import BayesianModelAveraging
from .kelly_sizing import KellyPositionSizer
