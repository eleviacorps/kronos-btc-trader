# Kronos BTC — Backtest Results Summary

## 1. BTC VWAP — Portfolio Sim (30 months, $5K → 0.15 BTC, 1:200)
| Variant | Final | Return | Trades | WR | PF | Max DD |
|---------|-------|--------|--------|----|----|--------|
| **Bot (VWAP only)** | **$87,685** | **+1,654%** | 107,233 | 42.5% | 1.06 | 2.6% |
| Agent (VWAP+EMA) | $54,398 | +988% | 82,874 | 42.0% | 1.04 | 8.2% |

- Profitable months: 23/31 (Bot), 22/31 (Agent)
- Built-in expiry fix (trades close at market if TP/SL not triggered in 2h)

## 2. BTC Kronos Model — Portfolio Sim (10 days, $5K, 0.5 BTC, 1:200)
| Final | Return | Trades | WR | PF | TP/SL/Exp |
|-------|--------|--------|----|----|-----------|
| **$5,547** | **+10.94%** | 107 | 43.0% | 1.14 | 38/52/17 |

- Entry: Kronos range_ext on 5m
- Positive but small sample (10 days)

## 3. XAU/USDT VWAP — Portfolio Sim (31 days, $5K, 0.75 oz, 1:200)
| Final | Return | Trades | WR | PF | TP/SL/Exp |
|-------|--------|--------|----|----|-----------|
| **$5,721** | **+14.43%** | 2,892 | 45.8% | 1.08 | 706/1,094/1,092 |

- Gold wins: 706 TP hits, consistent intraday vol, lower margin ($15/trade vs $232 BTC)

## 4. Live Bot PnL (current)
- Agent: +$302.83
- Bot: +$295.09

## Key Findings
- VWAP is the dominant strategy across all assets (+1,654% on BTC, +14% on Gold in 1 mo)
- Kronos model adds marginal value (1.14 PF vs VWAP's 1.06)
- Gold is more capital-efficient (lower margin, more TP hits)
- Expiry fix was critical — earlier version forced all non-TP/SL trades to register as SL losses
- TradersPost can bridge our Python strategy to prop firms (GoatFunded, FTMO, etc.) via webhooks
