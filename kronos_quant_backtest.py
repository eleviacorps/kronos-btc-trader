"""
Kronos Quant Hybrid Backtest — Tests the full fusion pipeline vs vanilla antitrend.

Usage:
  python kronos_quant_backtest.py                    # Quick test (100 windows)
  python kronos_quant_backtest.py --windows 500      # Full test
  python kronos_quant_backtest.py --symbol XAU/USDT:USDT  # Gold test
  python kronos_quant_backtest.py --no-kronos        # Price-based only (faster)

Pipeline:
  1. Fetch historical data from Binance/Yahoo
  2. Run Kronos predictions (or skip with --no-kronos)
  3. Run all 6 quant models on each window
  4. Fusion layer produces final signal with adaptive TP/SL + Kelly sizing
  5. Record trade outcomes
  6. Compare quant-hybrid vs vanilla antitrend
"""

import sys, os, json, time, argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import quant models
from quant_models.fusion_layer import QuantFusionEngine

# For Kronos predictions
try:
    from model import Kronos, KronosTokenizer, KronosPredictor
    HAS_KRONOS = True
except ImportError:
    HAS_KRONOS = False
    print("[WARN] Kronos model not available — run with --no-kronos for price-only test")

# =========================================================================
# CONFIG
# =========================================================================
DEFAULT_CAPITAL = 5000.0
DEFAULT_LEVERAGE = 200
DEFAULT_SIZE_BTC = 0.5
TP_PCT = 0.003  # 0.3%
SL_PCT = 0.002  # 0.2%
MAKER_FEE = 0.000  # paper — no fees


def fetch_binance_ohlcv(symbol: str = "BTC/USDT", limit: int = 1000) -> pd.DataFrame:
    """Fetch OHLCV from Binance."""
    import ccxt
    ex = ccxt.binance({"options": {"defaultType": "spot"}})
    ex.timeout = 30000
    ex.load_markets()
    ohlcv = ex.fetch_ohlcv(symbol, timeframe="5m", limit=limit)
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df


class SimplePortfolioSim:
    """Minimal portfolio simulator for hybrid comparison."""

    def __init__(self, capital: float, tp_pct: float, sl_pct: float, leverage: int = 200):
        self.capital = capital
        self.initial_capital = capital
        self.tp_pct = tp_pct
        self.sl_pct = sl_pct
        self.leverage = leverage
        self.position = None  # {"side", "entry", "size", "tp", "sl", "bar"}
        self.trades = []
        self.equity_curve = [capital]

    def open(self, side: str, price: float, size_btc: float, bar: int,
             tp_pct: float = None, sl_pct: float = None):
        """Open a position."""
        if self.position is not None:
            return
        tp = tp_pct or self.tp_pct
        sl = sl_pct or self.sl_pct
        self.position = {
            "side": side,
            "entry": price,
            "size": size_btc,
            "tp": price * (1 + tp) if side == "BUY" else price * (1 - tp),
            "sl": price * (1 - sl) if side == "BUY" else price * (1 + sl),
            "bar": bar,
        }

    def update(self, high: float, low: float, close: float, bar: int):
        """Walk one candle, check TP/SL/time-stop."""
        if self.position is None:
            self.equity_curve.append(self.capital)
            return

        pos = self.position
        exit_price = None
        reason = None

        if pos["side"] == "BUY":
            if high >= pos["tp"]:
                exit_price = pos["tp"]
                reason = "TP"
            elif low <= pos["sl"]:
                exit_price = pos["sl"]
                reason = "SL"
        else:  # SELL
            if low <= pos["tp"]:
                exit_price = pos["tp"]
                reason = "TP"
            elif high >= pos["sl"]:
                exit_price = pos["sl"]
                reason = "SL"

        # Time-stop after 4 candles (20 min)
        if exit_price is None and (bar - pos["bar"]) >= 4:
            exit_price = close
            reason = "EXPIRY"

        if exit_price is not None:
            self._close(exit_price, reason)
        self.equity_curve.append(self.capital)

    def _close(self, exit_price: float, reason: str):
        pos = self.position
        if pos["side"] == "BUY":
            pnl_pct = (exit_price - pos["entry"]) / pos["entry"]
        else:
            pnl_pct = (pos["entry"] - exit_price) / pos["entry"]

        pnl_usd = pnl_pct * pos["size"] * pos["entry"]

        # Apply leverage
        margin = pos["size"] * pos["entry"] / self.leverage
        capital_change = pnl_usd
        self.capital += capital_change

        self.trades.append({
            "side": pos["side"],
            "entry": round(pos["entry"], 2),
            "exit": round(exit_price, 2),
            "size": pos["size"],
            "pnl_usd": round(pnl_usd, 2),
            "pnl_pct": round(pnl_pct * 100, 2),
            "reason": reason,
            "bar": pos["bar"],
        })
        self.position = None

    def stats(self) -> dict:
        wins = [t for t in self.trades if t["pnl_usd"] > 0]
        losses = [t for t in self.trades if t["pnl_usd"] < 0]
        total_pnl = sum(t["pnl_usd"] for t in self.trades)
        return {
            "total_trades": len(self.trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": round(len(wins) / max(len(self.trades), 1) * 100, 1),
            "total_pnl": round(total_pnl, 2),
            "final_capital": round(self.capital, 2),
            "return_pct": round((self.capital - self.initial_capital) / self.initial_capital * 100, 2),
            "avg_win": round(sum(t["pnl_usd"] for t in wins) / max(len(wins), 1), 2),
            "avg_loss": round(abs(sum(t["pnl_usd"] for t in losses)) / max(len(losses), 1), 2),
        }

    def reset(self, capital: float = None):
        if capital is not None:
            self.capital = capital
            self.initial_capital = capital
        else:
            self.capital = self.initial_capital
        self.position = None
        self.trades = []
        self.equity_curve = [self.capital]


def main():
    parser = argparse.ArgumentParser(description="Kronos Quant Hybrid Backtest")
    parser.add_argument("--windows", type=int, default=100, help="Number of windows to test")
    parser.add_argument("--symbol", default="BTC/USDT", help="Trading pair")
    parser.add_argument("--capital", type=float, default=DEFAULT_CAPITAL)
    parser.add_argument("--no-kronos", action="store_true", help="Skip Kronos inference (price-only)")
    parser.add_argument("--save", action="store_true", help="Save results to JSON")
    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"  KRONOS QUANT HYBRID BACKTEST")
    print(f"  Symbol: {args.symbol}  |  Windows: {args.windows}  |  Capital: ${args.capital:.0f}")
    print(f"  Kronos: {'OFF (price-only)' if args.no_kronos else 'ON'}")
    print(f"{'='*60}")

    # ── Fetch data ──
    print("\n[1] Fetching data...")
    if "=X" in args.symbol or "=F" in args.symbol or "^" in args.symbol:
        import yfinance as yf
        ticker = yf.Ticker(args.symbol)
        df = ticker.history(period="2mo", interval="5m")
        df.columns = [c.lower() for c in df.columns]
        df.rename(columns={"open": "open", "high": "high", "low": "low", "close": "close", "volume": "volume"}, inplace=True)
    else:
        df = fetch_binance_ohlcv(args.symbol, limit=args.windows + 200)
    print(f"  Got {len(df)} candles")

    if len(df) < 100:
        print("  ❌ Need at least 100 candles")
        return

    # ── Init models ──
    print("\n[2] Initializing quant fusion engine...")
    engine = QuantFusionEngine(
        capital=args.capital, leverage=DEFAULT_LEVERAGE,
        base_tp_pct=TP_PCT * 100, base_sl_pct=SL_PCT * 100,
    )
    engine.warmup(df)
    engine.initialize(
        btc_price=float(df["close"].iloc[-1]),
        win_rate=0.48, avg_win=10.0, avg_loss=8.5,
    )
    print("  ✅ Fusion engine ready")

    # ── Init simulators ──
    sim_hybrid = SimplePortfolioSim(args.capital, TP_PCT, SL_PCT, DEFAULT_LEVERAGE)
    sim_baseline = SimplePortfolioSim(args.capital, TP_PCT, SL_PCT, DEFAULT_LEVERAGE)

    # ── Run windows ──
    print(f"\n[3] Running {args.windows} windows...")
    total = args.windows
    step = max(1, total // 10)

    for i in range(total):
        if i % step == 0:
            pct = i * 100 // total
            bar_len = 20
            filled = i * bar_len // total
            prog = "█" * filled + "░" * (bar_len - filled)
            print(f"  [{prog}] {pct}%", end="\r", flush=True)

        # Sliding window
        end = len(df) - total + i
        start = max(0, end - 200)
        window_df = df.iloc[start:end]

        if len(window_df) < 50:
            continue

        prices = window_df["close"].values
        current_price = float(prices[-1])
        high = float(window_df["high"].iloc[-1])
        low = float(window_df["low"].iloc[-1])

        # ── Mock Kronos prediction ──
        # In real mode this uses Kronos inference. In no-kronos mode,
        # use a simple momentum proxy: last 3 candles direction.
        if args.no_kronos:
            # Simple price momentum as Kronos proxy
            rets = np.diff(np.log(prices[-10:])) if len(prices) > 10 else np.array([0])
            net_change = float(np.sum(rets)) * 100
            if net_change > 0.05:
                kronos_dir = "BULLISH"
            elif net_change < -0.05:
                kronos_dir = "BEARISH"
            else:
                kronos_dir = "NEUTRAL"
            kronos_net = net_change
            kronos_conf = min(abs(net_change) / 0.2, 1.0)

            # Antitrend baseline: invert when extreme
            if net_change > 0.15:
                base_dir = "SELL"  # overextended up, expect reversion
                base_conf = min(abs(net_change) / 0.3, 1.0)
            elif net_change < -0.15:
                base_dir = "BUY"
                base_conf = min(abs(net_change) / 0.3, 1.0)
            else:
                base_dir = "HOLD"
                base_conf = 0.0
        else:
            # Real Kronos — skip for now (GPU dependent)
            kronos_dir = "NEUTRAL"
            kronos_net = 0.0
            kronos_conf = 0.0
            base_dir = "HOLD"
            base_conf = 0.0

        # ── Run fusion engine ──
        fusion_result = engine.analyze(
            df=window_df,
            kronos_direction=kronos_dir,
            kronos_net_pct=kronos_net,
            kronos_confidence=kronos_conf,
            current_price=current_price,
            strategy_signals={
                "antitrend": (1 if base_dir == "BUY" else -1 if base_dir == "SELL" else 0, base_conf),
                "vwap": (1 if current_price > float(window_df["close"].rolling(20).mean().iloc[-1]) else -1, 0.5),
            },
        )

        # ── Execute ──
        # Hybrid (quant fusion)
        if fusion_result["decision"] in ("BUY", "SELL") and fusion_result["confidence"] > 0.3:
            sim_hybrid.open(
                fusion_result["decision"],
                current_price,
                fusion_result["size"]["size_btc"],
                i,
                tp_pct=fusion_result["tp_sl"]["final_tp_pct"] / 100,
                sl_pct=fusion_result["tp_sl"]["final_sl_pct"] / 100,
            )

        # Baseline (simple antitrend)
        if base_dir in ("BUY", "SELL") and base_conf > 0.3:
            sim_baseline.open(base_dir, current_price, DEFAULT_SIZE_BTC, i)

        # Update positions with this candle's data
        sim_hybrid.update(high, low, current_price, i)
        sim_baseline.update(high, low, current_price, i)

    print(f"  [{'█' * 20}] 100% ✅")

    # ── Results ──
    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")

    hybrid_stats = sim_hybrid.stats()
    baseline_stats = sim_baseline.stats()

    print(f"\n  {'Metric':20s} {'Quant Hybrid':>16s} {'Baseline':>12s}")
    print(f"  {'─'*20} {'─'*16} {'─'*12}")
    print(f"  {'Final capital':20s} ${hybrid_stats['final_capital']:>10.2f}  ${baseline_stats['final_capital']:>8.2f}")
    print(f"  {'Return':20s} {hybrid_stats['return_pct']:>10.2f}%  {baseline_stats['return_pct']:>8.2f}%")
    print(f"  {'Total trades':20s} {hybrid_stats['total_trades']:>10d}  {baseline_stats['total_trades']:>8d}")
    print(f"  {'Win rate':20s} {hybrid_stats['win_rate']:>10.1f}%  {baseline_stats['win_rate']:>8.1f}%")
    print(f"  {'Avg win':20s} ${hybrid_stats['avg_win']:>10.2f}  ${baseline_stats['avg_win']:>8.2f}")
    print(f"  {'Avg loss':20s} ${hybrid_stats['avg_loss']:>10.2f}  ${baseline_stats['avg_loss']:>8.2f}")
    print(f"  {'Total PnL':20s} ${hybrid_stats['total_pnl']:>10.2f}  ${baseline_stats['total_pnl']:>8.2f}")

    diff = hybrid_stats["final_capital"] - baseline_stats["final_capital"]
    print(f"\n  Quant vs Baseline: ${diff:+.2f} ({diff/baseline_stats['final_capital']*100:+.1f}%)")

    # ── Save ──
    if args.save:
        results = {
            "config": {"symbol": args.symbol, "windows": args.windows, "capital": args.capital},
            "hybrid": hybrid_stats,
            "baseline": baseline_stats,
            "improvement": round(diff, 2),
        }
        path = f"quant_backtest_results_{args.symbol.replace('/', '_')}.json"
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), path), "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Saved to {path}")


if __name__ == "__main__":
    main()
