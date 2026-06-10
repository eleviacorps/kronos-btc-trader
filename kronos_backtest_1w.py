"""
Kronos 1-Week Backtest — $100 capital, 1:200 leverage, antitrend strategy.

Fetches 1+ week of BTC 5m OHLCV from Binance, runs Kronos predictions
every 30 minutes, and simulates antitrend mean-reversion trades with
1:200 leverage on $100 capital.

Usage:
  python kronos_backtest_1w.py              (BTC/USDT)
  python kronos_backtest_1w.py --symbol XAUUSDT  (Gold)
"""

import ccxt, pandas as pd, numpy as np, sys, time, json, os, argparse
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, '.')
from model import Kronos, KronosTokenizer, KronosPredictor

# ============================================================
# CONFIG
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument('--symbol', default='BTC/USDT', help='Trading pair (e.g. BTC/USDT, XAUUSDT)')
args = parser.parse_args()

# Yahoo Finance symbols (forex, commodities, indices) — never append /USDT
YF_SYMBOLS = {'XAUUSD', 'XAU/USD', 'GC=F', 'XAGUSD', 'XAG/USD', 'SI=F',
              'EURUSD', 'EUR/USD', 'EURUSD=X', 'GBPUSD', 'GBP/USD', 'GBPUSD=X',
              'JPYUSD', 'JPY/USD', 'USDJPY', 'USDJPY=X',
              'SPY', 'QQQ', 'TSLA', 'AAPL', '^GSPC', '^VIX'}

# Detect if raw symbol is YF-style (no auto-append /USDT)
raw_sym = args.symbol
if any(s in raw_sym for s in ['=F', '=X', '^']) or raw_sym.upper() in {s.upper() for s in YF_SYMBOLS}:
    SYMBOL = raw_sym  # Use as-is for Yahoo Finance
    IS_YF = True
else:
    SYMBOL = raw_sym if '/' in raw_sym else raw_sym + '/USDT'
    IS_YF = False

IS_FUTURES = ':USDT' in SYMBOL
CAPITAL = 100.0          # $100 initial capital
LEVERAGE = 200           # 1:200 leverage
POSITION_PCT = 0.10      # Use 10% of buying power per trade
BUYING_POWER = CAPITAL * LEVERAGE  # $20,000 max position
TRADE_SIZE = BUYING_POWER * POSITION_PCT  # $2,000 per trade
PREDICTION_INTERVAL = 30  # minutes between predictions
CONTEXT_CANDLES = 400     # candles of context for Kronos
PRED_LEN = 6              # predict 6 candles ahead (30 min)
TIMEFRAME = '5m'          # 5-minute candles

# Antitrend parameters (from optimization)
ANTITREND_THRESHOLD = 0.10  # predicted move % that triggers mean reversion
MIN_RANGE_PCT = 0.15       # skip if predicted range below this

# Take profit / stop loss
TP_PCT = 0.5    # Take profit at 0.5% move (5x threshold, realistic)
SL_PCT = 0.3    # Stop loss at 0.3% (risk management)

def fetch_data() -> pd.DataFrame:
    """Fetch ~2 weeks of 5m data from Binance or Yahoo Finance."""
    
    if IS_YF:
        # Use Yahoo Finance
        import yfinance as yf
        print(f"Fetching {SYMBOL} from Yahoo Finance...", flush=True)
        raw = yf.download(SYMBOL, period='1mo', interval='5m', progress=False)
        # yfinance returns multi-level columns: (Close, SYM), (High, SYM), etc.
        # Flatten by taking first level of column MultiIndex
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        df = raw.reset_index()
        df.columns = [c.lower() for c in df.columns]
        df = df.rename(columns={'datatime' if 'datatime' in df.columns else 'datetime': 'timestamps',
                                'date' if 'date' in df.columns else 'datetime': 'timestamps'})
        # yfinance DatetimeIndex
        if 'timestamps' not in df.columns:
            df['timestamps'] = raw.index if isinstance(raw.index, pd.DatetimeIndex) else pd.to_datetime(raw.index)
        df = df.sort_values('timestamps').reset_index(drop=True)
        df['amount'] = 0.0
        print(f"Fetched {len(df)} candles from Yahoo Finance", flush=True)
    else:
        # Use Binance
        opts = {'options': {'defaultType': 'future'}} if IS_FUTURES else {'options': {'defaultType': 'spot'}}
        exchange = ccxt.binance(opts)
        exchange.timeout = 30000
        exchange.load_markets()
        all_candles = []
        since = exchange.parse8601((datetime.utcnow() - timedelta(days=10)).isoformat())
        while True:
            try:
                ohlcv = exchange.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME,
                                              since=since, limit=1000)
                if not ohlcv:
                    break
                all_candles.extend(ohlcv)
                since = ohlcv[-1][0] + 1
                time.sleep(0.5)
                if len(all_candles) >= 2500:
                    break
            except Exception as e:
                print(f"Fetch error: {e}, retrying...")
                time.sleep(2)
        df = pd.DataFrame(all_candles, columns=['timestamp','open','high','low','close','volume'])
        df['timestamps'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['amount'] = 0.0
        df = df.drop_duplicates(subset='timestamp').sort_values('timestamp').reset_index(drop=True)
        print(f"Fetched {len(df)} candles from Binance", flush=True)
    
    print(f"  Date range: {df['timestamps'].iloc[0]} → {df['timestamps'].iloc[-1]}", flush=True)
    print(f"  Price range: ${df['low'].min():,.2f} - ${df['high'].max():,.2f}", flush=True)
    return df

def compute_recent_avg_range(df: pd.DataFrame, idx: int, window: int = 20) -> float:
    """Average H-L range % over the last `window` candles before idx."""
    if idx < window + 1:
        return 0.25
    recent = df.iloc[idx-window:idx]
    ranges = ((recent['high'] - recent['low']) / recent['close']) * 100
    return float(ranges.mean())

def run_backtest(df: pd.DataFrame):
    """Run the 1-week backtest with 1:200 leverage."""
    
    print(f"\n{'='*60}")
    print(f"KRONOS 1-WEEK BACKTEST — {SYMBOL}")
    print(f"{'='*60}")
    print(f"Capital:     ${CAPITAL:.2f}")
    print(f"Leverage:    1:{LEVERAGE}")
    print(f"Trade size:  ${TRADE_SIZE:.2f} per trade ($2,000 position)")
    print(f"Interval:    {PREDICTION_INTERVAL} min")
    print(f"Strategy:    Antitrend mean reversion (threshold={ANTITREND_THRESHOLD}%)")
    print(f"TP/SL:       +{TP_PCT}% / -{SL_PCT}%")
    print(f"{'='*60}\n")

    # Load models
    print("Loading Kronos...")
    tok = KronosTokenizer.from_pretrained('NeoQuasar/Kronos-Tokenizer-base')
    model = Kronos.from_pretrained('NeoQuasar/Kronos-base')
    device = 'cuda:0'
    tok = tok.to(device)
    model = model.to(device)
    predictor = KronosPredictor(model, tok, max_context=512)
    print(f"Kronos loaded on {device}")

    # Determine prediction points: every PREDICTION_INTERVAL minutes
    first_pred = CONTEXT_CANDLES + 1  # need 400 context + 1 to start
    step = PREDICTION_INTERVAL // 5  # convert minutes to 5m candles
    pred_indices = list(range(first_pred, len(df) - PRED_LEN - 1, step))
    
    print(f"Prediction points: {len(pred_indices)} ({(len(pred_indices) * PREDICTION_INTERVAL) / 60:.1f} hours of trading)")

    # Backtest loop
    capital = CAPITAL
    trades = []
    running_balance = [capital]
    running_timestamps = [df['timestamps'].iloc[0]]
    
    start_time = time.time()
    
    for batch_num, i in enumerate(pred_indices):
        # Progress
        if batch_num % 20 == 0:
            elapsed = time.time() - start_time
            pct = (batch_num / len(pred_indices)) * 100
            eta = (elapsed / (batch_num + 1)) * (len(pred_indices) - batch_num) if batch_num > 0 else 0
            print(f"  [{batch_num}/{len(pred_indices)}] {pct:.0f}% — capital: ${capital:.2f} — ETA: {eta:.0f}s")
        
        # ---- STEP 1: Prepare context ----
        ctx_df = df.iloc[i-CONTEXT_CANDLES:i].copy()
        x_df = ctx_df[['open','high','low','close','volume','amount']].copy()
        x_ts = ctx_df['timestamps'].copy()
        current_price = float(ctx_df['close'].iloc[-1])
        current_time = ctx_df['timestamps'].iloc[-1]
        
        # Future timestamps for prediction
        y_ts = pd.Series(pd.date_range(
            start=x_ts.iloc[-1] + pd.Timedelta(minutes=5),
            periods=PRED_LEN, freq='5min'))
        
        # Actual future data for verification
        actual_df = df.iloc[i:i+PRED_LEN]
        actual_return = ((actual_df['close'].iloc[-1] - current_price) / current_price) * 100 if len(actual_df) >= PRED_LEN else None
        
        # ---- STEP 2: Kronos prediction ----
        try:
            with torch.no_grad():
                pred_df = predictor.predict(
                    df=x_df, x_timestamp=x_ts, y_timestamp=y_ts,
                    pred_len=PRED_LEN, T=1.0, top_p=0.9,
                    sample_count=5, verbose=False)
        except Exception as e:
            print(f"  ⚠ Prediction error at idx {i}: {e}")
            continue
        
        # ---- STEP 3: Antitrend signal ----
        pred_close = float(pred_df['close'].iloc[-1])
        net_change = ((pred_close - current_price) / current_price) * 100
        pred_range = ((pred_df['high'].max() - pred_df['low'].min()) / current_price) * 100
        
        # Range filter
        if pred_range < MIN_RANGE_PCT:
            running_balance.append(capital)
            running_timestamps.append(current_time)
            continue
        
        # Antitrend logic (mean reversion)
        signal = None
        signal_confidence = 0.0
        
        if net_change >= ANTITREND_THRESHOLD:
            # Kronos bullish → antitrend says SELL
            signal = 'SELL'
            signal_confidence = min(abs(net_change) / 0.4, 0.85)
        elif net_change <= -ANTITREND_THRESHOLD:
            # Kronos bearish → antitrend says BUY
            signal = 'BUY'
            signal_confidence = min(abs(net_change) / 0.4, 0.85)
        
        if signal is None or signal_confidence < 0.3:
            running_balance.append(capital)
            running_timestamps.append(current_time)
            continue
        
        # ---- STEP 4: Execute trade with 1:200 leverage ----
        # Entry at close of last context candle (current price)
        entry_price = current_price
        exit_idx = i + PRED_LEN - 1  # close of last predicted candle
        
        if exit_idx >= len(df):
            running_balance.append(capital)
            running_timestamps.append(current_time)
            continue
        
        actual_exit_price = float(df.iloc[exit_idx]['close'])
        
        # Position size
        position_btc = TRADE_SIZE / entry_price
        
        # Actual return over the period
        if signal == 'BUY':
            # Long: profit if price goes up
            pnl_pct = ((actual_exit_price - entry_price) / entry_price) * 100
        else:
            # Short: profit if price goes down
            pnl_pct = ((entry_price - actual_exit_price) / entry_price) * 100
        
        # Leveraged PnL
        leveraged_pnl_pct = pnl_pct * LEVERAGE
        trade_pnl_usd = capital * (leveraged_pnl_pct / 100 * POSITION_PCT)
        # Actually, more accurately:
        # Position value = $2,000 (fixed per trade)
        # PnL USD = position_value * pnl_pct / 100
        trade_pnl_usd = TRADE_SIZE * pnl_pct / 100
        
        # Apply TP/SL
        hit_tp = pnl_pct >= TP_PCT
        hit_sl = pnl_pct <= -SL_PCT
        
        if hit_tp:
            trade_pnl_usd = TRADE_SIZE * TP_PCT / 100
        elif hit_sl:
            trade_pnl_usd = -(TRADE_SIZE * SL_PCT / 100)
        
        # Update capital
        capital += trade_pnl_usd
        
        trade_record = {
            'idx': i,
            'time': str(current_time),
            'price': round(entry_price, 2),
            'signal': signal,
            'confidence': round(signal_confidence, 3),
            'predicted_change': round(net_change, 3),
            'predicted_range': round(pred_range, 3),
            'actual_return': round(actual_return, 3) if actual_return is not None else None,
            'pnl_pct': round(pnl_pct, 3),
            'leveraged_pnl_pct': round(leveraged_pnl_pct, 3),
            'trade_pnl_usd': round(trade_pnl_usd, 2),
            'capital_after': round(capital, 2),
            'hit_tp': hit_tp,
            'hit_sl': hit_sl,
        }
        trades.append(trade_record)
        running_balance.append(capital)
        running_timestamps.append(current_time)
    
    # ---- RESULTS ----
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"BACKTEST COMPLETE — {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"{'='*60}")
    
    total_pnl = capital - CAPITAL
    total_pnl_pct = (capital / CAPITAL - 1) * 100
    num_trades = len(trades)
    wins = [t for t in trades if t['trade_pnl_usd'] > 0]
    losses = [t for t in trades if t['trade_pnl_usd'] <= 0]
    win_rate = (len(wins) / num_trades * 100) if num_trades > 0 else 0
    
    # Profit factor
    gross_profit = sum(t['trade_pnl_usd'] for t in trades if t['trade_pnl_usd'] > 0)
    gross_loss = abs(sum(t['trade_pnl_usd'] for t in trades if t['trade_pnl_usd'] < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    # Max drawdown
    peak = CAPITAL
    max_dd = 0
    max_dd_pct = 0
    for bal in running_balance:
        if bal > peak:
            peak = bal
        dd = peak - bal
        dd_pct = (dd / peak) * 100 if peak > 0 else 0
        if dd_pct > max_dd_pct:
            max_dd = dd
            max_dd_pct = dd_pct
    
    # Sharpe ratio (rough: assume 0% risk-free)
    if num_trades > 1:
        returns = [t['pnl_pct'] for t in trades if t['pnl_pct'] is not None]
        if returns:
            sharpe = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(num_trades / (len(pred_indices) / (24 * 60 / PREDICTION_INTERVAL) * 365))
        else:
            sharpe = 0
    else:
        sharpe = 0

    print(f"\n📊 FINAL RESULTS")
    print(f"  Starting capital: ${CAPITAL:.2f}")
    print(f"  Final capital:    ${capital:.2f}")
    print(f"  Total PnL:        ${total_pnl:+.2f} ({total_pnl_pct:+.2f}%)")
    print(f"  Trades:           {num_trades}")
    print(f"  Win rate:         {win_rate:.1f}%")
    print(f"  Profit factor:    {profit_factor:.2f}")
    print(f"  Max drawdown:     ${max_dd:.2f} ({max_dd_pct:.2f}%)")
    print(f"  Best trade:       ${max([t['trade_pnl_usd'] for t in trades], default=0):+.2f}")
    print(f"  Worst trade:      ${min([t['trade_pnl_usd'] for t in trades], default=0):+.2f}")
    print(f"\n📈 TRADE BREAKDOWN")
    print(f"  Avg win:          ${np.mean([t['trade_pnl_usd'] for t in wins]):+.2f}" if wins else "  Avg win:          N/A")
    print(f"  Avg loss:         ${np.mean([t['trade_pnl_usd'] for t in losses]):+.2f}" if losses else "  Avg loss:         N/A")
    print(f"  Avg trade PnL:    ${np.mean([t['trade_pnl_usd'] for t in trades]):+.2f}" if trades else "  Avg trade PnL:    N/A")

    # Summary of trades
    print(f"\n📋 TRADE LOG")
    print(f"{'#':>4} {'Signal':>6} {'Entry':>10} {'Pred%':>8} {'Act%':>8} {'PnL$':>8} {'Capital':>9}")
    print(f"{'-'*60}")
    for j, t in enumerate(trades[-50:]):  # last 50 trades
        signal_mark = '🟢' if t['signal'] == 'BUY' else '🔴'
        print(f"{j+1:>4} {signal_mark + ' ' + t['signal']:>6} ${t['price']:>8,.0f} {t['predicted_change']:>+7.2f}% {t['actual_return'] if t['actual_return'] is not None else '?':>+7}% ${t['trade_pnl_usd']:>+7.2f} ${t['capital_after']:>8,.2f}")

    print(f"\n🏆 FINAL PORTFOLIO VALUE: ${capital:.2f}")
    print(f"   Return: {total_pnl_pct:+.2f}%")
    
    return trades, running_balance

if __name__ == '__main__':
    import torch
    print(f"Step 1: Fetching 1 week {SYMBOL} data...", flush=True)
    df = fetch_data()
    print(f"\nDataset: {len(df)} candles, {df['timestamps'].iloc[0]} → {df['timestamps'].iloc[-1]}")
    
    trades, balance = run_backtest(df)
    
    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'capital': CAPITAL,
        'leverage': LEVERAGE,
        'total_trades': len(trades),
        'final_capital': round(CAPITAL + sum(t['trade_pnl_usd'] for t in trades), 2),
        'total_pnl': round(sum(t['trade_pnl_usd'] for t in trades), 2),
    }
    with open('backtest_1w_results.json', 'w') as f:
        json.dump({'summary': results, 'trades': trades}, f, indent=2, default=str)
    print(f"\nResults saved to backtest_1w_results.json")
