import ccxt
ex = ccxt.binance()
t = ex.fetch_ticker('BTC/USDT')
live = t['last']
signal = 64155.99
diff_pct = abs(live - signal) / signal * 100
print(f"Live BTC: {live}")
print(f"Signal BTC: {signal}")
print(f"Diff %: {diff_pct:.4f}%")
if diff_pct > 0.15:
    print("SIGNAL STALE - price moved > 0.15%")
else:
    print("Signal fresh - price stable")
