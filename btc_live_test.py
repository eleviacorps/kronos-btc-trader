# Testing live BTC with Kronos

import ccxt
import pandas as pd
import matplotlib.pyplot as plt

from model import Kronos, KronosTokenizer, KronosPredictor

# =========================
# CONFIG
# =========================

TIMEFRAME = "5m"
LOOKBACK = 400
PRED_LEN = 6  # 3 x 5min = 15 minutes

# =========================
# DOWNLOAD BTC DATA
# =========================

exchange = ccxt.binance()

ohlcv = exchange.fetch_ohlcv(
    "BTC/USDT",
    timeframe=TIMEFRAME,
    limit=600
)

df = pd.DataFrame(
    ohlcv,
    columns=[
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]
)

df["timestamps"] = pd.to_datetime(df["timestamp"], unit="ms")
df["amount"] = 0

df = df[
    [
        "timestamps",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount"
    ]
]

# =========================
# LOAD KRONOS
# =========================

tokenizer = KronosTokenizer.from_pretrained(
    "NeoQuasar/Kronos-Tokenizer-base"
)

model = Kronos.from_pretrained(
    "NeoQuasar/Kronos-base"
)

predictor = KronosPredictor(
    model,
    tokenizer,
    max_context=512
)

# =========================
# USE LATEST 400 CANDLES
# =========================

x_df = df.iloc[-LOOKBACK:][
    ["open", "high", "low", "close", "volume", "amount"]
]

x_timestamp = df.iloc[-LOOKBACK:]["timestamps"]

future_ts = pd.date_range(
    start=x_timestamp.iloc[-1] + pd.Timedelta(minutes=5),
    periods=PRED_LEN,
    freq="5min"
)

y_timestamp = pd.Series(future_ts)

# =========================
# PREDICT
# =========================

pred_df = predictor.predict(
    df=x_df,
    x_timestamp=x_timestamp,
    y_timestamp=y_timestamp,
    pred_len=PRED_LEN,
    T=1.0,
    top_p=0.9,
    sample_count=1,
    verbose=True
)

# =========================
# PRINT RESULTS
# =========================

current_close = x_df["close"].iloc[-1]

print("\n========================")
print("BTC FORECAST")
print("========================")
print(f"Current Close: {current_close:.2f}")
print()

for idx, (_, row) in enumerate(pred_df.iterrows(), start=1):
    print(
        f"Candle {idx}: "
        f"{row['close']:.2f}"
    )

print("\nPrediction Data:")
print(pred_df)

# =========================
# PLOT
# =========================

hist_close = x_df["close"]

plt.figure(figsize=(14, 7))

# Historical
plt.plot(
    range(len(hist_close)),
    hist_close.values,
    label="Historical",
    linewidth=2
)

# Prediction
plt.plot(
    range(
        len(hist_close),
        len(hist_close) + PRED_LEN
    ),
    pred_df["close"].values,
    label="Prediction",
    linewidth=3
)

# Forecast separator
plt.axvline(
    x=len(hist_close) - 1,
    linestyle="--",
    alpha=0.7
)

plt.title(
    f"BTC/USDT Forecast ({TIMEFRAME})"
)
plt.xlabel("Candles")
plt.ylabel("Price (USDT)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

print("\nPredicted closes:")
print(pred_df["close"])