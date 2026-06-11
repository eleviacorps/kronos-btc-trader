"""
Kronos Trader — Multi-strategy BTC trading system using Kronos foundation model.

Architecture:
  1. Data layer: fetch live BTC/USDT 5m OHLCV from Binance
  2. Kronos prediction layer: run Kronos-base (~100M params) for 6-candle forecasts
  3. Strategy layer: 5+ strategies interpreting Kronos predictions into trade signals
  4. Backtest engine: evaluate strategies on sliding windows of recent data
  5. Hermes integration: cron-friendly output for agent-in-the-loop decision making

Usage:
  python kronos_trader.py                    # Run live prediction + all strategies
  python kronos_trader.py --backtest          # Run backtest on recent data
  python kronos_trader.py --hermes-summary    # Generate compact summary for Hermes agent
"""

import ccxt
import pandas as pd
import numpy as np
import sys, os, json, time, argparse
from typing import Optional, Literal
from dataclasses import dataclass, field
from enum import Enum

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import Kronos, KronosTokenizer, KronosPredictor

# =========================================================================
# CONFIG
# =========================================================================
TIMEFRAME = "5m"
LOOKBACK = 400          # candles of history for Kronos context
PRED_LEN = 6            # candles to predict (30 min)
FETCH_LIMIT = 600       # total candles to fetch from exchange
MAX_CONTEXT = 512
MODEL_ID = "NeoQuasar/Kronos-base"
TOKENIZER_ID = "NeoQuasar/Kronos-Tokenizer-base"

# =========================================================================
# SCALP / HFT CONFIG — lower thresholds for high-frequency trading
# =========================================================================
# When --scalp is active, these override the main config.
# Designed to generate 200+ trades/day on 5m BTC data.
SCALP_TIMEFRAME = "5m"          # Keep 5m — Kronos was trained on this
SCALP_LOOKBACK = 200            # Shorter context for faster inference
SCALP_PRED_LEN = 4              # Predict fewer candles (20 min vs 30)
SCALP_FETCH_LIMIT = 400

# Antitrend thresholds — ultra-sensitive for maximum signal frequency
SCALP_ANTITREND_EXTREME = 0.04  # Trigger on tiny predicted moves
SCALP_ANTITREND_CONSERVATIVE = 0.08
SCALP_MIN_RANGE = 0.08          # Lower noise floor for scalp entries

# Other strategy thresholds (scaled down from swing)
SCALP_TREND_THRESHOLD = 0.06
SCALP_SLOPE_THRESHOLD = 0.008
SCALP_MIN_CONSENSUS = 4         # Easier candle consensus (4/6 instead of 5/6)
SCALP_EXTENSION_PCT = 0.10
SCALP_VOL_FILTER = 0.12
SCALP_VOL_MULTIPLIER = 1.2      # Lower vol expansion threshold

# Trade management for scalp mode
SCALP_TP_PCT = 0.003            # 0.3% take profit
SCALP_SL_PCT = 0.002           # 0.2% stop loss
SCALP_POSITION_PCT = 0.24       # 24% of buying power (~0.75 BTC at $63k with $1k capital)
SCALP_TIMESTOP_MIN = 20         # Auto-close after 20 min (was 10 — trades need room to hit TP)
SCALP_COOLDOWN_MIN = 3          # Min minutes between same-direction trades (was 2)

# =========================================================================
# DATA TYPES
# =========================================================================
class Signal(Enum):
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"

@dataclass
class StrategyResult:
    name: str
    signal: Signal
    confidence: float        # 0.0 - 1.0
    reason: str = ""
    metrics: dict = field(default_factory=dict)

@dataclass
class KronosPrediction:
    current_close: float
    current_time: str
    pred_df: pd.DataFrame
    direction: Literal["BULLISH", "BEARISH", "NEUTRAL"]
    net_change_pct: float
    range_pct: float
    volatility: float
    candle_changes: list

# =========================================================================
# DATA LAYER
# =========================================================================
class BinanceData:
    def __init__(self):
        self.exchange = ccxt.binance({'options': {'defaultType': 'spot'}})
        self.exchange.timeout = 30000
        self.exchange.load_markets()

    def fetch_ohlcv(self, limit: int = FETCH_LIMIT) -> pd.DataFrame:
        ohlcv = self.exchange.fetch_ohlcv("BTC/USDT", timeframe=TIMEFRAME, limit=limit)
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamps"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["amount"] = 0.0
        return df

    def get_latest_price(self) -> float:
        ticker = self.exchange.fetch_ticker("BTC/USDT")
        return ticker["last"]

# =========================================================================
# KRONOS PREDICTION LAYER
# =========================================================================
class KronosEngine:
    """Singleton wrapper around Kronos model inference."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        import torch
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"[KronosEngine] Loading models on {self.device}...", file=sys.stderr)
        self.tokenizer = KronosTokenizer.from_pretrained(TOKENIZER_ID)
        self.model = Kronos.from_pretrained(MODEL_ID)
        self.predictor = KronosPredictor(self.model, self.tokenizer, device=self.device, max_context=MAX_CONTEXT)
        print(f"[KronosEngine] Models loaded.", file=sys.stderr)

    def predict(self, df: pd.DataFrame, sample_count: int = 5, T: float = 1.0, top_p: float = 0.9,
                lookback: int = None, pred_len: int = None) -> KronosPrediction:
        """Run Kronos prediction on the latest LOOKBACK candles.
        Accepts optional lookback/pred_len override for scalp mode."""
        lb = lookback if lookback is not None else LOOKBACK
        pl = pred_len if pred_len is not None else PRED_LEN
        x_df = df.iloc[-lb:][["open", "high", "low", "close", "volume", "amount"]]
        x_ts = df.iloc[-lb:]["timestamps"]
        candle_minutes = int(TIMEFRAME.replace("m", ""))
        y_ts = pd.Series(pd.date_range(
            start=x_ts.iloc[-1] + pd.Timedelta(minutes=candle_minutes),
            periods=pl, freq=TIMEFRAME
        ))

        pred_df = self.predictor.predict(
            df=x_df, x_timestamp=x_ts, y_timestamp=y_ts,
            pred_len=pl, T=T, top_p=top_p,
            sample_count=sample_count, verbose=False
        )

        current = float(x_df["close"].iloc[-1])
        closes = pred_df["close"].values
        net_change = ((closes[-1] - current) / current) * 100
        candle_changes = [round(float(x), 3) for x in np.round(np.diff(closes) / closes[:-1] * 100, 3)]
        range_pct = ((pred_df["high"].max() - pred_df["low"].min()) / current) * 100

        if net_change > 0.1:
            direction = "BULLISH"
        elif net_change < -0.1:
            direction = "BEARISH"
        else:
            direction = "NEUTRAL"

        return KronosPrediction(
            current_close=current,
            current_time=str(pd.Timestamp.now()),
            pred_df=pred_df,
            direction=direction,
            net_change_pct=net_change,
            range_pct=range_pct,
            volatility=range_pct,
            candle_changes=candle_changes,
        )

# =========================================================================
# STRATEGIES
# =========================================================================
class Strategy:
    """Base strategy — override analyze()."""
    name: str = "base"

    def analyze(self, pred: KronosPrediction) -> StrategyResult:
        raise NotImplementedError

# -- Strategy 1: Trend Momentum --
class TrendMomentumStrategy(Strategy):
    """Follow the direction of the net predicted change if strong enough.
    Only triggers when predicted move exceeds recent avg noise floor."""
    name = "trend_momentum"

    def __init__(self, buy_threshold: float = 0.15, sell_threshold: float = -0.15,
                 min_range_pct: float = 0.25):
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.min_range_pct = min_range_pct

    def analyze(self, pred: KronosPrediction) -> StrategyResult:
        # Skip if too low vol — predictions are noise
        if pred.range_pct < self.min_range_pct:
            return StrategyResult(self.name, Signal.HOLD, 0.0,
                                  f"Range {pred.range_pct:.2f}% < {self.min_range_pct}% noise floor")

        net = pred.net_change_pct
        if net > self.buy_threshold:
            conf = min(net / 0.4, 1.0)
            return StrategyResult(self.name, Signal.BUY, conf,
                                  f"Net +{net:.2f}% > {self.buy_threshold}%")
        elif net < self.sell_threshold:
            conf = min(abs(net) / 0.4, 1.0)
            return StrategyResult(self.name, Signal.SELL, conf,
                                  f"Net {net:.2f}% < {self.sell_threshold}%")
        return StrategyResult(self.name, Signal.HOLD, 0.0,
                              f"Net {net:.2f}% within range")


class MomentumSlopeStrategy(Strategy):
    """Linear regression slope of predicted closes — only triggers on strong persistent trends."""
    name = "momentum_slope"

    def __init__(self, slope_threshold: float = 0.02, min_range_pct: float = 0.25):
        self.slope_threshold = slope_threshold
        self.min_range_pct = min_range_pct

    def analyze(self, pred: KronosPrediction) -> StrategyResult:
        if pred.range_pct < self.min_range_pct:
            return StrategyResult(self.name, Signal.HOLD, 0.0,
                                  f"Range {pred.range_pct:.2f}% below noise floor")

        closes = pred.pred_df["close"].values
        x = np.arange(len(closes))
        slope = np.polyfit(x, closes, 1)[0]
        slope_pct = (slope / closes[0]) * 100

        if slope_pct > self.slope_threshold:
            conf = min(slope_pct / 0.08, 1.0)
            return StrategyResult(self.name, Signal.BUY, conf,
                                  f"Slope +{slope_pct:.3f}%/step")
        elif slope_pct < -self.slope_threshold:
            conf = min(abs(slope_pct) / 0.08, 1.0)
            return StrategyResult(self.name, Signal.SELL, conf,
                                  f"Slope {slope_pct:.3f}%/step")
        return StrategyResult(self.name, Signal.HOLD, 0.0,
                              f"Slope {slope_pct:.3f}%/step flat")


class VolatilityBreakoutStrategy(Strategy):
    """
    Trade vol expansion events. In low-vol regimes, Kronos range predictions
    may be more reliable than direction. Only triggers when:
      - Predicted range > 1.5x recent avg range (vol expansion event)
      - Direction is confirmed by first candle
    """
    name = "vol_breakout"

    def __init__(self, vol_multiplier: float = 1.5, recent_window: int = 20):
        self.vol_multiplier = vol_multiplier
        self.recent_window = recent_window

    def _get_recent_avg_range(self, df: pd.DataFrame) -> float:
        if df is None or len(df) < self.recent_window + 1:
            return 0.25
        recent = df.iloc[-(self.recent_window+1):-1]
        ranges = ((recent["high"] - recent["low"]) / recent["close"]) * 100
        return float(ranges.mean())

    def analyze(self, pred: KronosPrediction, df=None) -> StrategyResult:
        avg_range = self._get_recent_avg_range(df)
        vol_ratio = pred.range_pct / (avg_range + 1e-6)

        if vol_ratio < self.vol_multiplier:
            return StrategyResult(self.name, Signal.HOLD, 0.0,
                                  f"Vol ratio {vol_ratio:.2f}x < {self.vol_multiplier}x")

        # Direction from first candle
        first_close = pred.pred_df["close"].iloc[0]
        if first_close > pred.current_close:
            conf = min(vol_ratio / 3.0, 1.0)
            return StrategyResult(self.name, Signal.BUY, conf,
                                  f"Vol breakout {vol_ratio:.2f}x, first bullish")
        else:
            conf = min(vol_ratio / 3.0, 1.0)
            return StrategyResult(self.name, Signal.SELL, conf,
                                  f"Vol breakout {vol_ratio:.2f}x, first bearish")


class CandleProgressionStrategy(Strategy):
    """
    Requires strong consensus in candle progression.
    5+ of 6 candles in same direction → high conviction signal.
    """
    name = "candle_progression"

    def __init__(self, min_consensus: int = 5, min_range_pct: float = 0.25):
        self.min_consensus = min_consensus
        self.min_range_pct = min_range_pct

    def analyze(self, pred: KronosPrediction) -> StrategyResult:
        if pred.range_pct < self.min_range_pct:
            return StrategyResult(self.name, Signal.HOLD, 0.0,
                                  f"Range {pred.range_pct:.2f}% below noise floor")

        closes = pred.pred_df["close"].values
        diffs = np.diff(closes)
        bulls = int(np.sum(diffs > 0))
        bears = int(np.sum(diffs < 0))
        total = len(diffs)

        if bulls >= self.min_consensus:
            conf = bulls / total
            return StrategyResult(self.name, Signal.BUY, conf,
                                  f"{bulls}/{total} bullish candles (consensus)")
        elif bears >= self.min_consensus:
            conf = bears / total
            return StrategyResult(self.name, Signal.SELL, conf,
                                  f"{bears}/{total} bearish candles (consensus)")

        return StrategyResult(self.name, Signal.HOLD, 0.0,
                              f"Mixed: {bulls}B/{bears}S (need {self.min_consensus})")


class RangeExtensionStrategy(Strategy):
    """
    If predicted high/low extends beyond current + threshold, trade breakout.
    Uses absolute price levels — more reliable than direction in low vol.
    """
    name = "range_extension"

    def __init__(self, extension_pct: float = 0.3, min_range_pct: float = 0.25):
        self.extension_pct = extension_pct
        self.min_range_pct = min_range_pct

    def analyze(self, pred: KronosPrediction) -> StrategyResult:
        if pred.range_pct < self.min_range_pct:
            return StrategyResult(self.name, Signal.HOLD, 0.0,
                                  f"Range {pred.range_pct:.2f}% below noise floor")

        high_ext = ((pred.pred_df["high"].max() - pred.current_close) / pred.current_close) * 100
        low_ext = ((pred.pred_df["low"].min() - pred.current_close) / pred.current_close) * 100

        signals = []
        if high_ext > self.extension_pct:
            signals.append(("BUY", high_ext))
        if low_ext < -self.extension_pct:
            signals.append(("SELL", abs(low_ext)))

        if not signals:
            return StrategyResult(self.name, Signal.HOLD, 0.0,
                                  f"High ext {high_ext:.2f}%, Low ext {low_ext:.2f}%")

        best = max(signals, key=lambda x: x[1])
        sig = Signal.BUY if best[0] == "BUY" else Signal.SELL
        conf = min(best[1] / 0.8, 1.0)
        return StrategyResult(self.name, sig, conf,
                              f"High ext {high_ext:.2f}%, Low ext {low_ext:.2f}%")



class AntitrendStrategy(Strategy):
    """
    Mean reversion: exploits Kronos's bearish bias and ~40% direction accuracy.
    When Kronos predicts a strong move, bet the opposite — its extreme
    predictions are where it's most wrong (over-extrapolating recent drift).
    
    Optimal params found via sweep: threshold=0.10, min_range_pct=0.15
    Gives ~87% trade frequency at +1.7% over 15 windows.
    """
    name = "antitrend"

    def __init__(self, extreme_pct: float = 0.10, min_range_pct: float = 0.15):
        self.extreme_pct = extreme_pct
        self.min_range_pct = min_range_pct

    def analyze(self, pred: KronosPrediction) -> StrategyResult:
        if pred.range_pct < self.min_range_pct:
            return StrategyResult(self.name, Signal.HOLD, 0.0,
                                  f"Range {pred.range_pct:.2f}% < {self.min_range_pct}% noise floor")

        net = pred.net_change_pct
        # Kronos bearish bias: extreme bearish predictions are typically wrong → BUY
        if net >= self.extreme_pct:
            conf = min(net / 0.4, 0.85)
            return StrategyResult(self.name, Signal.SELL, conf,
                                  f"Kronos bullish {net:+.2f}%, antitrend SELL (40% acc)")
        elif net < -self.extreme_pct:
            conf = min(abs(net) / 0.4, 0.85)
            return StrategyResult(self.name, Signal.BUY, conf,
                                  f"Kronos bearish {net:+.2f}%, antitrend BUY (40% acc)")
        return StrategyResult(self.name, Signal.HOLD, 0.0,
                              f"Net {net:+.2f}%, below {self.extreme_pct}% threshold")


class VolFilteredMomentumStrategy(Strategy):
    """
    Hybrid: only trade when vol is elevated AND direction is consistent.
    Designed for the specific finding that Kronos is ~50% directional but
    its range predictions may be meaningful.
    """
    name = "vol_filtered_momentum"

    def __init__(self, min_range_pct: float = 0.3, min_bullish_ratio: float = 0.6):
        self.min_range_pct = min_range_pct
        self.min_bullish_ratio = min_bullish_ratio

    def analyze(self, pred: KronosPrediction) -> StrategyResult:
        if pred.range_pct < self.min_range_pct:
            return StrategyResult(self.name, Signal.HOLD, 0.0,
                                  f"Range {pred.range_pct:.2f}% < {self.min_range_pct}% vol filter")

        closes = pred.pred_df["close"].values
        diffs = np.diff(closes)
        bulls = int(np.sum(diffs > 0))
        total = len(diffs)
        bullish_ratio = bulls / total

        net = pred.net_change_pct

        if bullish_ratio >= self.min_bullish_ratio and net > 0:
            conf = min(bullish_ratio, 1.0) * min(abs(net) / 0.3, 1.0)
            return StrategyResult(self.name, Signal.BUY, conf,
                                  f"Vol {pred.range_pct:.2f}%, bullish ratio {bullish_ratio:.0%}")
        elif (1 - bullish_ratio) >= self.min_bullish_ratio and net < 0:
            bearish_ratio = 1 - bullish_ratio
            conf = min(bearish_ratio, 1.0) * min(abs(net) / 0.3, 1.0)
            return StrategyResult(self.name, Signal.SELL, conf,
                                  f"Vol {pred.range_pct:.2f}%, bearish ratio {bearish_ratio:.0%}")

        return StrategyResult(self.name, Signal.HOLD, 0.0,
                              f"Vol {pred.range_pct:.2f}%, ratio {bullish_ratio:.0%} mixed")

# =========================================================================
# ENSEMBLE — combine all strategies
# =========================================================================

# -- VWAP Strategy (price-based: score from HFT eval: +15.33%, 62.5% WR) --
class VWAPStrategy(Strategy):
    """VWAP bias: price above VWAP = bullish, below = bearish.
    Best standalone HFT strategy — 62.5% win rate on 10-min scalp."""
    name = "vwap"

    def __init__(self, threshold_pct: float = 0.1):
        self.threshold = threshold_pct

    def analyze(self, pred: KronosPrediction, df=None) -> StrategyResult:
        if df is None or len(df) < 20:
            return StrategyResult(self.name, Signal.HOLD, 0.0, "No data")
        typical = (df["high"] + df["low"] + df["close"]) / 3
        vwap = (typical * df["volume"]).sum() / df["volume"].sum()
        price = float(df["close"].iloc[-1])
        dist = (price - vwap) / vwap * 100
        if dist > self.threshold:
            conf = min(dist / 0.5, 0.8)
            return StrategyResult(self.name, Signal.BUY, conf, f"Price +{dist:.2f}% above VWAP")
        elif dist < -self.threshold:
            conf = min(abs(dist) / 0.5, 0.8)
            return StrategyResult(self.name, Signal.SELL, conf, f"Price {dist:.2f}% below VWAP")
        return StrategyResult(self.name, Signal.HOLD, 0.0, f"Price {dist:.2f}% near VWAP")


class EMAStrategy(Strategy):
    """EMA 9/21 crossover: fast above slow = bullish, below = bearish.
    Second-best HFT strategy — +10.51%, 56.5% WR."""
    name = "ema_cross"

    def __init__(self, fast: int = 9, slow: int = 21):
        self.fast = fast
        self.slow = slow

    def analyze(self, pred: KronosPrediction, df=None) -> StrategyResult:
        if df is None or len(df) < self.slow + 1:
            return StrategyResult(self.name, Signal.HOLD, 0.0, "No data")
        closes = df["close"].values
        ema_f = pd.Series(closes).ewm(span=self.fast).mean().values[-1]
        ema_s = pd.Series(closes).ewm(span=self.slow).mean().values[-1]
        prev_f = pd.Series(closes[:-1]).ewm(span=self.fast).mean().values[-1]
        prev_s = pd.Series(closes[:-1]).ewm(span=self.slow).mean().values[-1]
        spread = (ema_f - ema_s) / ema_s * 100
        # Crossover just happened
        if prev_f <= prev_s and ema_f > ema_s:
            conf = min(abs(spread) * 5, 0.8)
            return StrategyResult(self.name, Signal.BUY, conf, f"EMA9 crossed above EMA21 ({spread:+.3f}%)")
        if prev_f >= prev_s and ema_f < ema_s:
            conf = min(abs(spread) * 5, 0.8)
            return StrategyResult(self.name, Signal.SELL, conf, f"EMA9 crossed below EMA21 ({spread:+.3f}%)")
        # Sustained bias
        if ema_f > ema_s:
            return StrategyResult(self.name, Signal.BUY, 0.2, f"EMA9({ema_f:.0f}) > EMA21({ema_s:.0f})")
        if ema_f < ema_s:
            return StrategyResult(self.name, Signal.SELL, 0.2, f"EMA9({ema_f:.0f}) < EMA21({ema_s:.0f})")
        return StrategyResult(self.name, Signal.HOLD, 0.0, "EMAs flat")


class StrategyEnsemble:
    """
    Runs all strategies and produces a consensus signal.
    Weighted by confidence, with boost for strategies that agree.
    """

    def __init__(self):
        self.strategies: list[Strategy] = [
            AntitrendStrategy(extreme_pct=0.10, min_range_pct=0.15),       # Primary: mean reversion
            AntitrendStrategy(extreme_pct=0.20, min_range_pct=0.20),       # Conservative variant
            TrendMomentumStrategy(buy_threshold=0.15, sell_threshold=-0.15, min_range_pct=0.25),
            TrendMomentumStrategy(buy_threshold=0.30, sell_threshold=-0.30, min_range_pct=0.30),
            MomentumSlopeStrategy(slope_threshold=0.015, min_range_pct=0.25),
            CandleProgressionStrategy(min_consensus=5, min_range_pct=0.25),
            RangeExtensionStrategy(extension_pct=0.30, min_range_pct=0.25),
            VolFilteredMomentumStrategy(min_range_pct=0.30, min_bullish_ratio=0.6),
        ]
        self.vol_strategy = VolatilityBreakoutStrategy(vol_multiplier=1.5)

    def analyze(self, pred: KronosPrediction, df: Optional[pd.DataFrame] = None) -> list[StrategyResult]:
        results = []
        for s in self.strategies:
            try:
                results.append(s.analyze(pred))
            except Exception as e:
                results.append(StrategyResult(s.name, Signal.HOLD, 0.0, f"Error: {e}"))
        try:
            results.append(self.vol_strategy.analyze(pred, df))
        except Exception as e:
            results.append(StrategyResult(self.vol_strategy.name, Signal.HOLD, 0.0, f"Error: {e}"))
        return results


# -- RSI Strategy (highest WR at 54.1% over 30 months, 30 trades/day) --
class RSIStrategy(Strategy):
    """RSI overbought/oversold. Highest win rate of any strategy tested."""
    name = "rsi"

    def __init__(self, period=14, oversold=30, overbought=70):
        self.period = period
        self.oversold = oversold
        self.overbought = overbought

    def analyze(self, pred: KronosPrediction, df=None) -> StrategyResult:
        if df is None or len(df) < self.period + 1:
            return StrategyResult(self.name, Signal.HOLD, 0.0, "No data")
        closes = df["close"].values[-(self.period + 1):]
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        if avg_loss == 0:
            return StrategyResult(self.name, Signal.HOLD, 0.0, "RSI flat")
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        if rsi < self.oversold:
            conf = min((self.oversold - rsi) / self.oversold, 1.0)
            return StrategyResult(self.name, Signal.BUY, conf, f"RSI {rsi:.1f} oversold")
        elif rsi > self.overbought:
            conf = min((rsi - self.overbought) / (100 - self.overbought), 1.0)
            return StrategyResult(self.name, Signal.SELL, conf, f"RSI {rsi:.1f} overbought")
        return StrategyResult(self.name, Signal.HOLD, 0.0, f"RSI {rsi:.1f} neutral")


class ScalpStrategyEnsemble(StrategyEnsemble):
    """HFT ensemble optimized from 30-month backtest (129k windows).
    Primary: VWAP (+179%), EMA (+171%), RSI (+113%, highest WR 54%).
    Kronos range_ext supplements when breakout signals align."""

    def __init__(self):
        self.strategies: list[Strategy] = [
            # Price-based — proven across 30 months of BTC
            VWAPStrategy(threshold_pct=0.10),
            VWAPStrategy(threshold_pct=0.20),  # Wider threshold = fewer, higher-conf trades
            EMAStrategy(fast=9, slow=21),
            RSIStrategy(period=14, oversold=30, overbought=70),
            # Kronos-based — supplements when model predicts breakouts
            RangeExtensionStrategy(extension_pct=SCALP_EXTENSION_PCT, min_range_pct=SCALP_MIN_RANGE),
        ]
        self.vol_strategy = VolatilityBreakoutStrategy(vol_multiplier=SCALP_VOL_MULTIPLIER)

    def analyze(self, pred: KronosPrediction, df=None) -> list[StrategyResult]:
        results = []
        for s in self.strategies:
            try:
                results.append(s.analyze(pred, df))
            except Exception as e:
                results.append(StrategyResult(s.name, Signal.HOLD, 0.0, f"Error: {e}"))
        try:
            results.append(self.vol_strategy.analyze(pred, df))
        except Exception as e:
            results.append(StrategyResult(self.vol_strategy.name, Signal.HOLD, 0.0, f"Error: {e}"))
        return results

    def consensus(self, results: list[StrategyResult]) -> StrategyResult:
        """
        Weighted consensus from 30-month backtest data.
        VWAP: proven best (+179%, 119 trades/day)
        EMA: strong runner-up (+171%)
        RSI: highest WR (54.1%)
        Range_ext / vol_breakout: secondary supplements
        """
        signal_map = {
            Signal.STRONG_BUY: 2, Signal.BUY: 1, Signal.HOLD: 0,
            Signal.SELL: -1, Signal.STRONG_SELL: -2
        }

        # Tier 1 — highest proven edge (VWAP = 2.5x, EMA = 2x)
        vwap_results = [r for r in results if r.name == "vwap"]
        ema_results = [r for r in results if r.name == "ema_cross"]
        # Tier 2 — strong performers (RSI = 1.5x, range_ext = 1.5x)
        rsi_results = [r for r in results if r.name == "rsi"]
        range_results = [r for r in results if r.name == "range_extension"]
        # Tier 3 — the rest (1x)
        other_results = [r for r in results if r.name not in {"vwap", "ema_cross", "rsi", "range_extension"}]

        total_weight = 0.0
        weighted_sum = 0.0

        for r in vwap_results:
            w = r.confidence * 2.5
            weighted_sum += signal_map[r.signal] * w
            total_weight += w
        for r in ema_results:
            w = r.confidence * 2.0
            weighted_sum += signal_map[r.signal] * w
            total_weight += w
        for r in rsi_results:
            w = r.confidence * 1.5
            weighted_sum += signal_map[r.signal] * w
            total_weight += w
        for r in range_results:
            w = r.confidence * 1.5
            weighted_sum += signal_map[r.signal] * w
            total_weight += w
        for r in other_results:
            w = r.confidence
            weighted_sum += signal_map[r.signal] * w
            total_weight += w

        if total_weight == 0:
            return StrategyResult("ensemble", Signal.HOLD, 0.0, "No confident signals")

        avg = weighted_sum / total_weight
        max_possible = sum(signal_map[r.signal] * r.confidence for r in results
                          if signal_map[r.signal] > 0)
        max_neg_possible = abs(sum(signal_map[r.signal] * r.confidence for r in results
                                if signal_map[r.signal] < 0))

        if avg > 0.5:
            sig = Signal.BUY
            conf = min(avg / 2.0, 1.0)
        elif avg > 0.15:
            sig = Signal.BUY
            conf = 0.4
        elif avg < -0.5:
            sig = Signal.SELL
            conf = min(abs(avg) / 2.0, 1.0)
        elif avg < -0.15:
            sig = Signal.SELL
            conf = 0.4
        else:
            sig = Signal.HOLD
            conf = 0.0

        # Count high-confidence agreement
        non_hold = [r for r in results if r.signal != Signal.HOLD]
        if non_hold:
            agree_count = sum(1 for r in non_hold
                            if (sig == Signal.BUY and r.signal in (Signal.BUY, Signal.STRONG_BUY))
                            or (sig == Signal.SELL and r.signal in (Signal.SELL, Signal.STRONG_SELL)))
            agreement_ratio = agree_count / len(non_hold)
        else:
            agreement_ratio = 0.0

        metrics = {
            "weighted_avg": round(avg, 3),
            "num_strategies": len(results),
            "non_hold": sum(1 for r in results if r.signal != Signal.HOLD),
            "agreement_ratio": round(agreement_ratio, 2),
        }

        return StrategyResult("ensemble", sig, conf,
                              f"avg={avg:.2f}, agree={agreement_ratio:.0%}",
                              metrics)


# =========================================================================
# BACKTEST ENGINE
# =========================================================================
@dataclass
class BacktestTrade:
    timestamp: str
    predicted_signal: Signal
    predicted_change_pct: float
    actual_change_pct: float  # what actually happened in the following 30 min
    entry_price: float
    exit_price: float
    pnl_pct: float
    strategy_name: str

class Backtester:
    """
    Runs sliding-window backtest: for each prediction point, run Kronos,
    evaluate all strategies, then check what actually happened.
    """

    def __init__(self, num_windows: int = 10):
        self.num_windows = num_windows

    def run(self, df: pd.DataFrame) -> dict:
        """
        df must have at least (LOOKBACK + PRED_LEN + num_windows) rows.
        Slides forward by 1 candle each window.
        """
        engine = KronosEngine()
        total_needed = LOOKBACK + PRED_LEN + self.num_windows
        if len(df) < total_needed:
            print(f"[Backtester] Need {total_needed} rows, have {len(df)}", file=sys.stderr)
            return {"error": "insufficient data"}

        ensemble = ScalpStrategyEnsemble()
        all_trades: list[BacktestTrade] = []
        strategy_performance: dict[str, list[float]] = {}

        for i in range(self.num_windows):
            start = len(df) - total_needed + i
            window_df = df.iloc[start:start + LOOKBACK + PRED_LEN + 1].copy()

            # The "actual" is what happened in the PRED_LEN candles after the context
            context_df = window_df.iloc[:LOOKBACK]
            actual_df = window_df.iloc[LOOKBACK:LOOKBACK + PRED_LEN]

            actual_close_start = float(context_df["close"].iloc[-1])
            actual_close_end = float(actual_df["close"].iloc[-1])
            actual_change = ((actual_close_end - actual_close_start) / actual_close_start) * 100

            # Full df + context for the prediction call
            full_for_pred = pd.concat([context_df, actual_df], ignore_index=True)

            try:
                pred = engine.predict(full_for_pred, sample_count=5)
            except Exception as e:
                print(f"[Backtester] Window {i} pred failed: {e}", file=sys.stderr)
                continue

            results = ensemble.analyze(pred, full_for_pred)
            consensus = ensemble.consensus(results)

            # Record trade for the ensemble
            trade = BacktestTrade(
                timestamp=str(context_df["timestamps"].iloc[-1]),
                predicted_signal=consensus.signal,
                predicted_change_pct=pred.net_change_pct,
                actual_change_pct=actual_change,
                entry_price=actual_close_start,
                exit_price=actual_close_end,
                pnl_pct=actual_change if consensus.signal == Signal.BUY else
                       -actual_change if consensus.signal == Signal.SELL else 0.0,
                strategy_name="ensemble",
            )
            all_trades.append(trade)

            # Record per-strategy
            for r in results:
                if r.name not in strategy_performance:
                    strategy_performance[r.name] = []
                if r.signal == Signal.BUY:
                    strategy_performance[r.name].append(actual_change)
                elif r.signal == Signal.SELL:
                    strategy_performance[r.name].append(-actual_change)
                else:
                    strategy_performance[r.name].append(0.0)

        return self._summarize(all_trades, strategy_performance)

    def _summarize(self, trades: list[BacktestTrade],
                   strategy_perf: dict[str, list[float]]) -> dict:
        summary = {
            "windows_tested": len(trades),
            "ensemble": {
                "total_trades": sum(1 for t in trades if t.predicted_signal != Signal.HOLD),
                "total_pnl_pct": round(sum(t.pnl_pct for t in trades), 2),
                "avg_pnl_per_trade": round(
                    np.mean([t.pnl_pct for t in trades if t.pnl_pct != 0]), 3
                    ) if any(t.pnl_pct != 0 for t in trades) else 0,
                "win_rate": round(
                    sum(1 for t in trades if t.pnl_pct > 0) /
                    max(sum(1 for t in trades if t.pnl_pct != 0), 1), 3
                ),
                "max_win": round(max((t.pnl_pct for t in trades), default=0), 2),
                "max_loss": round(min((t.pnl_pct for t in trades), default=0), 2),
            },
            "strategies": {}
        }

        for name, pnls in strategy_perf.items():
            non_zero = [p for p in pnls if p != 0]
            summary["strategies"][name] = {
                "total_pnl_pct": round(sum(pnls), 2),
                "avg_pnl": round(np.mean(pnls), 3),
                "win_rate": round(sum(1 for p in pnls if p > 0) / max(len(pnls), 1), 3) if pnls else 0,
                "trade_count": len(non_zero),
                "sharpe_approx": round(np.mean(pnls) / max(np.std(pnls), 1e-6), 3) if pnls and np.std(pnls) > 0 else 0,
            }

        # Rank strategies by total PnL
        ranked = sorted(summary["strategies"].items(),
                        key=lambda x: x[1]["total_pnl_pct"], reverse=True)
        summary["ranking"] = [{"strategy": name, "pnl": data["total_pnl_pct"],
                               "win_rate": data["win_rate"], "trades": data["trade_count"]}
                              for name, data in ranked]

        summary["best_strategy"] = ranked[0][0] if ranked else "none"
        return summary


# =========================================================================
# HERMES INTEGRATION — produce a compact summary for the agent
# =========================================================================
def hermes_summary(pred: KronosPrediction, results: list[StrategyResult],
                   consensus: StrategyResult, scalp: bool = False) -> str:
    """Generate a compact, parseable summary for Hermes Agent to make the final call.
    When scalp=True, uses HFT-optimized thresholds for higher frequency."""

    mode_tag = " [SCALP HFT]" if scalp else ""
    horizon_min = PRED_LEN * 5 if not scalp else SCALP_PRED_LEN * 5

    lines = [
        f"=== KRONOS TRADING SIGNAL{mode_tag} ===",
        f"Time: {pred.current_time}",
        f"BTC: ${pred.current_close:,.2f}",
        f"Kronos: {pred.direction} ({pred.net_change_pct:+.2f}% over {horizon_min}m)",
        f"Predicted range: {pred.range_pct:.2f}%",
        f"Candle progression: {pred.candle_changes}",
        "",
        "--- Strategy Signals ---",
    ]

    for r in results:
        icon = {"STRONG_BUY": "🟢", "BUY": "🟢", "HOLD": "⚪", "SELL": "🔴", "STRONG_SELL": "🔴"}
        ic = icon.get(r.signal.value, "⚪")
        lines.append(f"  {ic} {r.name:<25s} {r.signal.value:<12s} ({r.confidence:.2f}) {r.reason}")

    lines.append("")
    sig_icon = {"STRONG_BUY": "🟢🟢", "BUY": "🟢", "HOLD": "⚪", "SELL": "🔴", "STRONG_SELL": "🔴🔴"}
    lines.append(f"--- Consensus: {sig_icon.get(consensus.signal.value, '⚪')} {consensus.signal.value} "
                 f"(conf={consensus.confidence:.2f}) ---")
    lines.append(f"Reason: {consensus.reason}")
    lines.append("")
    lines.append("ACTION REQUIRED: Review and say BUY, SELL, or HOLD to execute.")

    return "\n".join(lines)


# =========================================================================
# MAIN
# =========================================================================
def main():
    parser = argparse.ArgumentParser(description="Kronos Trader")
    parser.add_argument("--backtest", action="store_true", help="Run backtest")
    parser.add_argument("--windows", type=int, default=10, help="Backtest windows")
    parser.add_argument("--hermes-summary", action="store_true",
                        help="Output compact Hermes-agent summary")
    parser.add_argument("--forecast-only", action="store_true",
                        help="Only show forecast, no trading signals")
    parser.add_argument("--scalp", action="store_true",
                        help="Scalp/HFT mode: lower thresholds, tighter TP/SL, ~200+ trades/day")
    args = parser.parse_args()

    # Use scalp config if requested
    lb = SCALP_LOOKBACK if args.scalp else LOOKBACK
    pl = SCALP_PRED_LEN if args.scalp else PRED_LEN
    fl = SCALP_FETCH_LIMIT if args.scalp else FETCH_LIMIT

    # Fetch data
    data = BinanceData()
    df = data.fetch_ohlcv(limit=fl)

    if args.backtest:
        print(f"\n=== KRONOS BACKTEST ({args.windows} windows) ===")
        bt = Backtester(num_windows=args.windows)
        result = bt.run(df)
        if "error" in result:
            print(f"Error: {result['error']}")
            return
        print(f"\nWindows tested: {result['windows_tested']}")
        print(f"Ensemble total PnL: {result['ensemble']['total_pnl_pct']:+.2f}%")
        print(f"Win rate: {result['ensemble']['win_rate']:.1%}")
        print(f"Max win/loss: {result['ensemble']['max_win']:+.2f}% / {result['ensemble']['max_loss']:+.2f}%")
        print(f"\n--- Strategy Rankings ---")
        for s in result["ranking"]:
            print(f"  #{s['strategy']:<25s} PnL: {s['pnl']:+.2f}%  WR: {s['win_rate']:.0%}  Trades: {s['trades']}")
        print(f"\nBest strategy: {result['best_strategy']}")
        return

    # Live prediction
    engine = KronosEngine()
    pred = engine.predict(df, sample_count=5, lookback=lb, pred_len=pl)

    horizon = pl * 5  # minutes

    if args.forecast_only:
        print(f"\nBTC/USDT: ${pred.current_close:,.2f}")
        print(f"Kronos forecast ({horizon}min): {pred.direction} ({pred.net_change_pct:+.2f}%)")
        print(f"Range: {pred.range_pct:.2f}%")
        print(f"Candles: {[f'{c:+.2f}%' for c in pred.candle_changes]}")
        return

    # Run all strategies (scalp ensemble if in HFT mode)
    if args.scalp:
        ensemble = ScalpStrategyEnsemble()
        mode_label = "SCALP HFT"
    else:
        ensemble = StrategyEnsemble()
        mode_label = "SWING"
    results = ensemble.analyze(pred, df)
    consensus = ensemble.consensus(results)

    if args.hermes_summary:
        print(hermes_summary(pred, results, consensus, scalp=args.scalp))
        return

    # Full output
    print(f"\n{'='*60}")
    print(f"  KRONOS TRADER — Live BTC/USDT [{mode_label}]")
    print(f"{'='*60}")
    print(f"  Price: ${pred.current_close:,.2f}")
    print(f"  Time:  {pred.current_time}")
    print(f"  Kronos: {pred.direction} ({pred.net_change_pct:+.2f}% in {horizon}m)")
    print(f"  Range:  {pred.range_pct:.2f}%")
    print(f"{'='*60}")
    print(f"\n--- Predicted Candles ---")
    for i, (_, row) in enumerate(pred.pred_df.iterrows()):
        chg = ((row["close"] - pred.current_close) / pred.current_close) * 100
        print(f"  +{i+1}: ${row['close']:,.2f} ({chg:+.2f}%)  H:{row['high']:,.2f} L:{row['low']:,.2f}")

    print(f"\n--- Strategy Results ---")
    for r in results:
        icon = {"STRONG_BUY": "🟢🟢", "BUY": "🟢", "HOLD": "⚪", "SELL": "🔴", "STRONG_SELL": "🔴🔴"}
        print(f"  {icon.get(r.signal.value, '⚪')} {r.name:<25s} {r.signal.value:<12s} "
              f"(conf={r.confidence:.2f})  {r.reason}")

    sig_icon = {"STRONG_BUY": "🟢🟢", "BUY": "🟢", "HOLD": "⚪", "SELL": "🔴", "STRONG_SELL": "🔴🔴"}
    print(f"\n  >>> Consensus: {sig_icon.get(consensus.signal.value, '⚪')} "
          f"{consensus.signal.value} (conf={consensus.confidence:.2f}) <<<")
    print(f"      {consensus.reason}")
    print()

    # Suggested scalp TP/SL info when in HFT mode
    if args.scalp:
        tp_price = pred.current_close * (1 + SCALP_TP_PCT)
        sl_price = pred.current_close * (1 - SCALP_SL_PCT)
        print(f"  [SCALP] TP: ${tp_price:,.2f} (+{SCALP_TP_PCT*100:.1f}%)  "
              f"SL: ${sl_price:,.2f} (-{SCALP_SL_PCT*100:.1f}%)")
        print(f"  [SCALP] Position: {SCALP_POSITION_PCT:.0%} of buying power  "
              f"Time-stop: {SCALP_TIMESTOP_MIN}min  Cooldown: {SCALP_COOLDOWN_MIN}min")

    # Suggested action for Hermes
    if consensus.signal in (Signal.BUY, Signal.STRONG_BUY):
        print(f"  → Hermes decision: BUY BTC/USDT")
        print(f"  → Expected move: +{pred.net_change_pct:.2f}% (${pred.current_close * (1+pred.net_change_pct/100):,.2f})")
        print(f"  → Stop-loss: ${pred.pred_df['low'].min():,.2f} ({((pred.pred_df['low'].min()/pred.current_close)-1)*100:.2f}%)")
    elif consensus.signal in (Signal.SELL, Signal.STRONG_SELL):
        print(f"  → Hermes decision: SELL BTC/USDT")
        print(f"  → Expected move: {pred.net_change_pct:.2f}%")
        print(f"  → Stop-loss: ${pred.pred_df['high'].max():,.2f}")
    else:
        print(f"  → Hermes decision: HOLD — no clear signal")


if __name__ == "__main__":
    main()
