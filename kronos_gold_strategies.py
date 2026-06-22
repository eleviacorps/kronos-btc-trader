"""
kronos_gold_strategies.py — V3: Properly implemented strategies.
- ADX uses Wilder's RMA (not SMA)
- Kronos is PRIMARY signal generator, tech indicators are CONFIRMATION
- All major gold strategies with correct implementations
"""
import json, sys, os, time, argparse
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import torch, ccxt

sys.path.insert(0, '.')
from model import Kronos, KronosTokenizer, KronosPredictor

PROJECT_DIR = Path(__file__).parent
parser = argparse.ArgumentParser()
parser.add_argument('--windows', type=int, default=2000)
parser.add_argument('--capital', type=float, default=5000)
parser.add_argument('--size', type=float, default=0.75)
parser.add_argument('--save', action='store_true')
parser.add_argument('--days', type=int, default=45)
parser.add_argument('--no-kronos', action='store_true')
parser.add_argument('--quick', action='store_true', help='Run fewer windows for testing')
args = parser.parse_args()

CAPITAL = args.capital; POS_SIZE = args.size; LEV = 200
TP_PCT = 0.3; SL_PCT = 0.2

# ============================================================
# DATA
# ============================================================
def fetch_gold_data(days=45):
    print("Fetching XAU/USDT:USDT...", flush=True)
    ex = ccxt.binance({'options': {'defaultType': 'future'}, 'timeout': 30000})
    ex.load_markets()
    all_c = []
    since = ex.parse8601((datetime.utcnow() - timedelta(days=days)).isoformat())
    while True:
        o = ex.fetch_ohlcv('XAU/USDT:USDT', '5m', since=since, limit=1000)
        if not o: break
        all_c.extend(o); since = o[-1][0] + 1; time.sleep(0.15)
        if len(all_c) >= 13000: break
    df = pd.DataFrame(all_c, columns=['ts','o','h','l','c','v'])
    df['t'] = pd.to_datetime(df['ts'], unit='ms')
    df = df.drop_duplicates('ts').sort_values('ts').reset_index(drop=True)
    print(f"  {len(df)} candles ({df['t'].iloc[0]} -> {df['t'].iloc[-1]})", flush=True)
    return df

# ============================================================
# KRONOS CACHE
# ============================================================
predictor = None
kronos_cache = {}

def load_kronos():
    global predictor
    if args.no_kronos: return
    print("Loading Kronos...", flush=True)
    device = 'cuda:0'
    tok = KronosTokenizer.from_pretrained('NeoQuasar/Kronos-Tokenizer-base').to(device)
    model = Kronos.from_pretrained('NeoQuasar/Kronos-base').to(device)
    predictor = KronosPredictor(model, tok, max_context=512)
    print("  Done", flush=True)

def kronos_pred(df5, lb=200, pl=4):
    if predictor is None or len(df5) < lb: return None, 0
    ulb = min(lb, len(df5)-1)
    x = df5.iloc[-ulb:][['o','h','l','c','v']].copy()
    x.columns = ['open','high','low','close','volume']; x['amount']=0.0
    ts = df5['t'].iloc[-ulb:].reset_index(drop=True)
    yt = pd.Series(pd.date_range(start=ts.iloc[-1]+pd.Timedelta(minutes=5), periods=pl, freq='5min'))
    try:
        with torch.no_grad():
            p = predictor.predict(df=x, x_timestamp=ts, y_timestamp=yt, pred_len=pl, T=1.0, top_p=0.9, sample_count=5, verbose=False)
        pc = float(p['close'].iloc[-1]); cp = float(df5['c'].iloc[-1])
        return ((pc-cp)/cp)*100, ((p['high'].max()-p['low'].min())/cp)*100
    except: return None, 0

def precompute_kronos(df, indices):
    print(f"Pre-computing {len(indices)} Kronos predictions...", flush=True)
    start = time.time()
    for batch, idx in enumerate(indices):
        if batch % 50 == 0 and batch > 0:
            eta = (time.time()-start)/(batch+1)*(len(indices)-batch)
            print(f"  [{batch}/{len(indices)}] {batch/len(indices)*100:.0f}% ETA: {eta:.0f}s", flush=True)
        ctx = df.iloc[max(0, idx-205):idx].copy()
        net, rng = kronos_pred(ctx)
        kronos_cache[idx] = (net, rng)
    print(f"  Cache: {len(kronos_cache)} preds in {time.time()-start:.0f}s", flush=True)

def get_kronos_signal(idx):
    """Get Kronos direction signal from cache."""
    kn = kronos_cache.get(idx, (None, 0))[0]
    if kn is None: return 'HOLD', 0
    if kn > 0.08: return 'BUY', min(kn/0.3, 0.85)
    if kn < -0.08: return 'SELL', min(abs(kn)/0.3, 0.85)
    return 'HOLD', 0

def get_kronos_antitrend(idx):
    """ANTITREND: trade AGAINST Kronos predictions."""
    kn = kronos_cache.get(idx, (None, 0))[0]
    if kn is None: return 'HOLD', 0
    if kn > 0.10: return 'SELL', min(kn/0.4, 0.85)  # Kronos UP → SELL
    if kn < -0.10: return 'BUY', min(abs(kn)/0.4, 0.85)  # Kronos DOWN → BUY
    return 'HOLD', 0

# ============================================================
# SIMULATION
# ============================================================
def sim_trade(side, entry, idx, df, mc=24, tp_pct=None, sl_pct=None):
    """Simulate a position with TP/SL/EXPIRY trigger fills.
    tp_pct/sl_pct override defaults for adaptive TP/SL."""
    tp_pct = tp_pct if tp_pct is not None else TP_PCT
    sl_pct = sl_pct if sl_pct is not None else SL_PCT
    tp = entry*(1+tp_pct/100) if side=='BUY' else entry*(1-tp_pct/100)
    sl = entry*(1-sl_pct/100) if side=='BUY' else entry*(1+sl_pct/100)
    outcome='EXPIRY'; ex_p=entry
    for o in range(1, mc+1):
        fi=idx+o
        if fi>=len(df): break
        cp=float(df.iloc[fi]['c'])
        if (side=='BUY' and cp>=tp) or (side=='SELL' and cp<=tp): ex_p=tp; outcome='TP'; break
        if (side=='BUY' and cp<=sl) or (side=='SELL' and cp>=sl): ex_p=sl; outcome='SL'; break
    if outcome=='EXPIRY': ex_p=float(df.iloc[min(idx+mc, len(df)-1)]['c'])
    pnl_pct=((ex_p-entry)/entry)*100
    if side=='SELL': pnl_pct=-pnl_pct
    pnl=pnl_pct/100*POS_SIZE*entry
    if outcome=='TP': pnl=tp_pct/100*POS_SIZE*entry
    if outcome=='SL': pnl=-(sl_pct/100)*POS_SIZE*entry
    return round(pnl,2), outcome

def run_trade(sig, conf, price, idx, df, balance, tp_pct=None, sl_pct=None):
    """Execute a single trade, return (new_balance, trade_dict or None)."""
    if sig == 'HOLD' or conf < 0.2: return balance, None
    margin = POS_SIZE * price / LEV
    if margin > balance: return balance, None
    balance -= margin
    pnl, outcome = sim_trade(sig, price, idx, df, tp_pct=tp_pct, sl_pct=sl_pct)
    balance += margin + pnl
    return balance, {'entry': round(price,2), 'side': sig, 'pnl': pnl, 'outcome': outcome, 'bal': round(balance,2)}

# ============================================================
# PROPERLY IMPLEMENTED INDICATORS
# ============================================================

def wilders_rma(series, period):
    """Wilder's RMA (Running Moving Average). Same as: RMA = (prev * (p-1) + current) / p"""
    return series.ewm(alpha=1/period, adjust=False).mean()

def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_g = wilders_rma(gain, period)  # Wilder's RMA on gains
    avg_l = wilders_rma(loss, period)  # Wilder's RMA on losses
    rs = avg_g / avg_l.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calc_adx_wilder(ctx, period=14):
    """Wilder's ADX using proper RMA smoothing."""
    h = ctx['h'].values; l = ctx['l'].values; c = ctx['c'].values
    n = len(h)
    tr = np.zeros(n); pdm = np.zeros(n); ndm = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1]))
        up = h[i] - h[i-1]
        dn = l[i-1] - l[i]
        if up > dn and up > 0: pdm[i] = up
        if dn > up and dn > 0: ndm[i] = dn

    # Use Wilder's RMA (EWM alpha=1/period) for smoothing
    tr_s = pd.Series(tr).ewm(alpha=1/period, adjust=False).mean()
    pdm_s = pd.Series(pdm).ewm(alpha=1/period, adjust=False).mean()
    ndm_s = pd.Series(ndm).ewm(alpha=1/period, adjust=False).mean()

    # +DI and -DI
    pdi = 100 * pdm_s / tr_s.replace(0, np.nan)
    ndi = 100 * ndm_s / tr_s.replace(0, np.nan)

    # DX
    dx = 100 * abs(pdi - ndi) / (pdi + ndi).replace(0, np.nan)

    # ADX = RMA of DX
    adx = dx.ewm(alpha=1/period, adjust=False).mean()

    return float(adx.iloc[-1]), float(pdi.iloc[-1]), float(ndi.iloc[-1])

def calc_vwap(ctx):
    typ = (ctx['h'] + ctx['l'] + ctx['c']) / 3
    return float((typ * ctx['v']).sum() / ctx['v'].sum())

def calc_macd(series, fast=12, slow=26, signal=9):
    e1 = series.ewm(span=fast).mean()
    e2 = series.ewm(span=slow).mean()
    m = e1 - e2
    s = m.ewm(span=signal).mean()
    return m, s, m - s

def calc_bb(ctx, period=20, std_dev=2):
    c = ctx['c']; sm = c.rolling(period).mean(); st = c.rolling(period).std()
    return float(sm.iloc[-1]), float(sm.iloc[-1]+std_dev*st.iloc[-1]), float(sm.iloc[-1]-std_dev*st.iloc[-1])

def calc_stoch(ctx, period=14, k_smooth=3, d_smooth=3):
    l14 = ctx['l'].rolling(period).min(); h14 = ctx['h'].rolling(period).max()
    k = 100 * (ctx['c'] - l14) / (h14 - l14).replace(0, np.nan)
    k = k.rolling(k_smooth).mean(); d = k.rolling(d_smooth).mean()
    return float(k.iloc[-1]), float(d.iloc[-1])

def calc_keltner(ctx, period=20, atr_mult=1.5):
    c = ctx['c']; e = c.ewm(span=period).mean()
    tr = pd.concat([ctx['h']-ctx['l'], abs(ctx['h']-c.shift(1)), abs(ctx['l']-c.shift(1))], axis=1).max(axis=1)
    at = tr.rolling(period).mean()
    return float(e.iloc[-1]), float((e+at_mult*at).iloc[-1]), float((e-at_mult*at).iloc[-1])

def calc_ichimoku(ctx):
    h,l,c = ctx['h'], ctx['l'], ctx['c']
    tk = (h.rolling(9).max() + l.rolling(9).min()) / 2
    kj = (h.rolling(26).max() + l.rolling(26).min()) / 2
    sa = (tk + kj) / 2
    sb = (h.rolling(52).max() + l.rolling(52).min()) / 2
    pr = float(c.iloc[-1])
    ct = max(sa.iloc[-26] if len(sa)>26 else sa.iloc[-1], sb.iloc[-26] if len(sb)>26 else sb.iloc[-1])
    cb = min(sa.iloc[-26] if len(sa)>26 else sa.iloc[-1], sb.iloc[-26] if len(sb)>26 else sb.iloc[-1])
    return pr, float(ct), float(cb), float(tk.iloc[-1]-kj.iloc[-1]), float(tk.iloc[-2]-kj.iloc[-2]) if len(tk)>1 else 0

# ============================================================
# STRATEGY DEFINITIONS
# Each strategy returns (signal, confidence) where signal is BUY/SELL/HOLD
# ============================================================

def run_strategy(name, signal_fn, df, indices, use_kronos=False, adaptive_tp_sl=False):
    """Run a strategy over all windows.
    If adaptive_tp_sl=True, TP/SL scale with Kronos prediction magnitude."""
    balance = CAPITAL
    trades = []
    start = time.time()

    for batch, idx in enumerate(indices):
        if batch % 200 == 0 and batch > 0:
            eta = (time.time()-start)/(batch+1)*(len(indices)-batch)
            print(f"  {name}: [{batch}/{len(indices)}] ETA: {eta:.0f}s", flush=True)

        ctx = df.iloc[idx-100:idx].copy()
        price = float(ctx['c'].iloc[-1])

        # Get signal
        sig, conf = signal_fn(ctx, df, idx)
        if sig == 'HOLD' or conf < 0.2: continue

        # Kronos confirmation (only trade when Kronos CONFIRMS)
        if use_kronos:
            ksig, kconf = get_kronos_signal(idx)
            if ksig == 'HOLD' or ksig != sig:
                continue

        # Determine TP/SL for this trade (adaptive for Kronos, fixed for others)
        tp_pct = TP_PCT
        sl_pct = SL_PCT
        if adaptive_tp_sl:
            kn = kronos_cache.get(idx, (None, 0))[0]
            if kn is not None and abs(kn) > 0.02:
                # Scale TP/SL to Kronos prediction magnitude
                pred_mag = abs(kn)
                tp_pct = max(pred_mag * 1.5, 0.05)  # TP = 1.5x predicted move, min 0.05%
                sl_pct = max(pred_mag * 0.5, 0.03)  # SL = 0.5x predicted move, min 0.03%

        balance, trade = run_trade(sig, conf, price, idx, df, balance, tp_pct=tp_pct, sl_pct=sl_pct)
        if trade:
            trades.append(trade)
        if balance <= 0:
            break

    elapsed = time.time() - start
    pnls = [t['pnl'] for t in trades]; wins = [p for p in pnls if p > 0]; losses = [p for p in pnls if p < 0]
    wr = len(wins)/len(pnls)*100 if pnls else 0
    tc = sum(1 for t in trades if t['outcome']=='TP')
    sc = sum(1 for t in trades if t['outcome']=='SL')
    ec = sum(1 for t in trades if t['outcome']=='EXPIRY')
    pf = abs(sum(wins)/sum(losses)) if losses else float('inf')
    return {'trades':trades,'final':balance,'pnl':balance-CAPITAL,'return_pct':(balance/CAPITAL-1)*100,
            'trade_count':len(trades),'win_rate':wr,'profit_factor':pf,'tp':tc,'sl':sc,'exp':ec,
            'avg_win':round(np.mean(wins),2) if wins else 0,'avg_loss':round(np.mean(losses),2) if losses else 0}

# ============================================================
# STRATEGY 1: KRONOS PRIMARY — Kronos generates the signal
# ============================================================
def s_kronos_only(ctx, df, idx):
    """Kronos predicts future price direction. Primary strategy."""
    ksig, kconf = get_kronos_signal(idx)
    return ksig, kconf

def s_kronos_antitrend(ctx, df, idx):
    """Kronos ANTITREND: trade AGAINST strong Kronos predictions."""
    ksig, kconf = get_kronos_antitrend(idx)
    return ksig, kconf

def s_kronos_adx(ctx, df, idx):
    """Kronos signal + ADX trend filter (ADX>25 required)."""
    if len(ctx) < 30: return 'HOLD', 0
    ksig, kconf = get_kronos_signal(idx)
    if ksig == 'HOLD': return 'HOLD', 0
    try:
        adx, pdi, ndi = calc_adx_wilder(ctx, 14)
        if adx < 25: return 'HOLD', 0  # No trend -> skip
    except:
        return 'HOLD', 0
    return ksig, kconf * 0.9  # Slightly reduce confidence

def s_kronos_vwap(ctx, df, idx):
    """Kronos signal + VWAP confirmation."""
    if len(ctx) < 20: return 'HOLD', 0
    ksig, kconf = get_kronos_signal(idx)
    if ksig == 'HOLD': return 'HOLD', 0
    vwap = calc_vwap(ctx)
    price = float(ctx['c'].iloc[-1])
    dist = (price - vwap) / vwap * 100
    # VWAP must agree with direction
    if ksig == 'BUY' and dist < -0.05: return ksig, kconf  # Price below VWAP, Kronos says BUY -> agree
    if ksig == 'SELL' and dist > 0.05: return ksig, kconf  # Price above VWAP, Kronos says SELL -> agree
    # VWAP disagrees or neutral -> still trade but with lower confidence
    if abs(dist) > 0.1: return 'HOLD', 0  # Strong VWAP disagreement -> skip
    return ksig, kconf * 0.7

def s_kronos_macd(ctx, df, idx):
    """Kronos signal + MACD confirmation."""
    if len(ctx) < 27: return 'HOLD', 0
    ksig, kconf = get_kronos_signal(idx)
    if ksig == 'HOLD': return 'HOLD', 0
    m, s, _ = calc_macd(ctx['c'])
    macd_bull = float(m.iloc[-1]) > float(s.iloc[-1])
    macd_bear = float(m.iloc[-1]) < float(s.iloc[-1])
    if (ksig == 'BUY' and macd_bull) or (ksig == 'SELL' and macd_bear):
        return ksig, min(kconf * 1.1, 0.9)  # Boost confidence
    return 'HOLD', 0  # MACD disagrees

def s_kronos_rsi(ctx, df, idx):
    """Kronos signal + RSI extreme filter."""
    if len(ctx) < 15: return 'HOLD', 0
    ksig, kconf = get_kronos_signal(idx)
    if ksig == 'HOLD': return 'HOLD', 0
    rsi = calc_rsi(ctx['c'], 14)
    rv = float(rsi.iloc[-1])
    # RSI must not be extreme in opposite direction
    if ksig == 'BUY' and rv < 70: return ksig, kconf  # Not overbought -> OK
    if ksig == 'SELL' and rv > 30: return ksig, kconf  # Not oversold -> OK
    return 'HOLD', 0  # RSI extreme in wrong direction

# ============================================================
# STRATEGY 2: ADX PRIMARY — Wilder's ADX generates signal
# ============================================================
def s_adx_only(ctx, df, idx):
    """ADX+DI: Trade when ADX>25 in DI direction."""
    if len(ctx) < 30: return 'HOLD', 0
    try:
        adx, pdi, ndi = calc_adx_wilder(ctx, 14)
        if adx > 25:
            if pdi > ndi: return 'BUY', min((pdi-ndi)/100*3, 0.7)
            if ndi > pdi: return 'SELL', min((ndi-pdi)/100*3, 0.7)
    except:
        pass
    return 'HOLD', 0

def s_adx_kronos(ctx, df, idx):
    """ADX filter + Kronos direction."""
    if len(ctx) < 30: return 'HOLD', 0
    try:
        adx, pdi, ndi = calc_adx_wilder(ctx, 14)
    except:
        return 'HOLD', 0
    if adx < 25: return 'HOLD', 0
    ksig, kconf = get_kronos_signal(idx)
    if ksig == 'HOLD': return 'HOLD', 0
    return ksig, kconf * 0.85

# ============================================================
# STRATEGY 3: CLASSIC TECHNICAL (properly implemented)
# ============================================================
def s_vwap_trend(ctx, df, idx):
    if len(ctx) < 20: return 'HOLD', 0
    v = calc_vwap(ctx); p = float(ctx['c'].iloc[-1]); d = (p-v)/v*100
    if d > 0.1: return 'BUY', min(d/0.5, 0.8)
    if d < -0.1: return 'SELL', min(abs(d)/0.5, 0.8)
    return 'HOLD', 0

def s_vwap_mr(ctx, df, idx):
    if len(ctx) < 20: return 'HOLD', 0
    v = calc_vwap(ctx); p = float(ctx['c'].iloc[-1]); d = (p-v)/v*100
    if d < -0.1: return 'BUY', min(abs(d)/0.5, 0.8)
    if d > 0.1: return 'SELL', min(d/0.5, 0.8)
    return 'HOLD', 0

def s_rsi_extreme(ctx, df, idx):
    """Proper RSI with Wilder's smoothing: extreme entries only."""
    if len(ctx) < 15: return 'HOLD', 0
    r = calc_rsi(ctx['c'], 14)
    rv = float(r.iloc[-1])
    if rv < 25: return 'BUY', min((25-rv)/25, 0.85)
    if rv > 75: return 'SELL', min((rv-75)/25, 0.85)
    return 'HOLD', 0

def s_macd_cross(ctx, df, idx):
    if len(ctx) < 27: return 'HOLD', 0
    m, s, _ = calc_macd(ctx['c'])
    mv = float(m.iloc[-1]); sv = float(s.iloc[-1])
    pm = float(m.iloc[-2]) if len(m) > 1 else 0; ps = float(s.iloc[-2]) if len(s) > 1 else 0
    if mv > sv and pm <= ps: return 'BUY', min(abs(mv-sv)/abs(sv+1e-10)*10, 0.8)
    if mv < sv and pm >= ps: return 'SELL', min(abs(mv-sv)/abs(sv+1e-10)*10, 0.8)
    return 'HOLD', 0

def s_bollinger_mr(ctx, df, idx):
    if len(ctx) < 20: return 'HOLD', 0
    _, u, l = calc_bb(ctx); p = float(ctx['c'].iloc[-1])
    # %B indicator: (price - lower) / (upper - lower)
    bb_width = u - l
    if bb_width < 0.01: return 'HOLD', 0
    pct_b = (p - l) / bb_width
    if pct_b < 0.05: return 'BUY', min((0.05-pct_b)/0.05, 0.85)
    if pct_b > 0.95: return 'SELL', min((pct_b-0.95)/0.05, 0.85)
    return 'HOLD', 0

def s_stoch_extreme(ctx, df, idx):
    if len(ctx) < 20: return 'HOLD', 0
    k, d = calc_stoch(ctx)
    pk, pd_ = calc_stoch(ctx.iloc[:-1]) if len(ctx) > 15 else (k, d)
    if k < 15 and k > d and pk <= pd_: return 'BUY', min((15-k)/15, 0.8)
    if k > 85 and k < d and pk >= pd_: return 'SELL', min((k-85)/15, 0.8)
    return 'HOLD', 0

def s_macd_rsi_combo(ctx, df, idx):
    """MACD + RSI combo: MACD crossover + RSI non-extreme = trending entry."""
    if len(ctx) < 27: return 'HOLD', 0
    m, s, _ = calc_macd(ctx['c'])
    mv = float(m.iloc[-1]); sv = float(s.iloc[-1])
    pm = float(m.iloc[-2]) if len(m) > 1 else 0
    if not ((mv > sv and pm <= sv) or (mv < sv and pm >= sv)):
        return 'HOLD', 0
    r = calc_rsi(ctx['c'], 14); rv = float(r.iloc[-1])
    if mv > sv and 40 <= rv <= 70: return 'BUY', min((rv-40)/30*0.5+0.3, 0.8)
    if mv < sv and 30 <= rv <= 60: return 'SELL', min((60-rv)/30*0.5+0.3, 0.8)
    return 'HOLD', 0

def s_ichimoku_tk(ctx, df, idx):
    """Ichimoku TK cross with cloud filter."""
    if len(ctx) < 53: return 'HOLD', 0
    try:
        pr, ct, cb, tk, ptk, _, _ = calc_ichimoku(ctx)
        if pr > ct and tk > 0 and ptk <= 0: return 'BUY', min(abs(tk)/pr*100*5, 0.8)
        if pr < cb and tk < 0 and ptk >= 0: return 'SELL', min(abs(tk)/pr*100*5, 0.8)
    except: pass
    return 'HOLD', 0

# ============================================================
# STRATEGY 4: ICT / SMC / ORB — Institutional Strategies
# ============================================================

def find_swing_highs_lows(ctx, lookback=10):
    """Find swing highs and lows in the context."""
    h = ctx['h'].values; l = ctx['l'].values
    n = len(h)
    highs = []
    lows = []
    for i in range(lookback, n - lookback):
        # Swing high: center candle has highest high
        if all(h[i] >= h[i-j] for j in range(1, lookback+1)) and \
           all(h[i] >= h[i+j] for j in range(1, lookback+1)):
            highs.append((i, h[i]))
        # Swing low: center candle has lowest low
        if all(l[i] <= l[i-j] for j in range(1, lookback+1)) and \
           all(l[i] <= l[i+j] for j in range(1, lookback+1)):
            lows.append((i, l[i]))
    return highs, lows

def find_fvg(ctx):
    """Find Fair Value Gaps (imbalances between consecutive candles)."""
    h = ctx['h'].values; l = ctx['l'].values; c = ctx['c'].values
    fvgs = []
    for i in range(2, len(h)):
        # Bullish FVG: current candle's low > previous candle's high
        if l[i] > h[i-1]:
            gap_top = l[i]
            gap_bot = h[i-1]
            midpoint = (gap_top + gap_bot) / 2
            fvgs.append(('BULL', gap_bot, gap_top, midpoint, i))
        # Bearish FVG: previous candle's low > current candle's high
        if l[i-1] > h[i]:
            gap_top = l[i-1]
            gap_bot = h[i]
            midpoint = (gap_top + gap_bot) / 2
            fvgs.append(('BEAR', gap_bot, gap_top, midpoint, i))
    return fvgs

def find_order_blocks(ctx):
    """Find Order Blocks (last candle before a strong move)."""
    h = ctx['h'].values; l = ctx['l'].values; c = ctx['c'].values
    obs = []
    for i in range(3, len(h)-3):
        # Bullish OB: bearish candle that precedes 3 consecutive bullish candles
        if c[i] < c[i-1]:  # bearish candle
            if all(c[i+j] > c[i+j-1] for j in range(1, 4) if i+j < len(c)):
                obs.append(('BULL', i, float(l[i]), float(h[i])))
        # Bearish OB: bullish candle that precedes 3 consecutive bearish candles
        if c[i] > c[i-1]:  # bullish candle
            if all(c[i+j] < c[i+j-1] for j in range(1, 4) if i+j < len(c)):
                obs.append(('BEAR', i, float(l[i]), float(h[i])))
    return obs

def find_liquidity_sweeps(ctx, df, idx):
    """Detect liquidity sweeps: price breaks recent swing high/low then reverses."""
    if len(ctx) < 20: return None
    highs, lows = find_swing_highs_lows(ctx, lookback=5)
    if not highs and not lows: return None
    current_price = float(ctx['c'].iloc[-1])
    prev_price = float(ctx['c'].iloc[-2]) if len(ctx) > 1 else current_price
    recent_highs = [hp for _, hp in highs[-3:]] if highs else []
    recent_lows = [lp for _, lp in lows[-3:]] if lows else []

    # Liquidity sweep UP: price broke above recent high then closed back below
    if recent_highs:
        sh = max(recent_highs)
        # Check if price swept above the swing high and reversed
        max_last_3 = float(ctx['h'].iloc[-3:].max())
        if max_last_3 > sh and current_price < sh and prev_price > sh:
            return 'SELL', 0.65  # Sweep high -> expect reversal DOWN

    # Liquidity sweep DOWN: price broke below recent low then closed back above
    if recent_lows:
        sl = min(recent_lows)
        min_last_3 = float(ctx['l'].iloc[-3:].min())
        if min_last_3 < sl and current_price > sl and prev_price < sl:
            return 'BUY', 0.65  # Sweep low -> expect reversal UP

    return None

def s_orb(ctx, df, idx):
    """Opening Range Breakout: trade breakouts of daily opening range."""
    # Define opening range as first 6 candles of each day
    current_time = ctx['t'].iloc[-1]
    # Find today's candles in full df up to idx
    day_start = pd.Timestamp(current_time).normalize()  # midnight
    # Get candles from today within df up to idx
    today_mask = (df['t'] >= day_start) & (df['t'] <= df['t'].iloc[idx]) if idx < len(df) else pd.Series([False])
    today_indices = today_mask[today_mask].index if today_mask.any() else []
    if len(today_indices) < 2:
        return 'HOLD', 0

    today_df = df.loc[today_indices[0]:idx]
    if len(today_df) < 8:  # Need at least 8 candles to have opening range
        return 'HOLD', 0

    # Opening range = first 6 candles
    open_range = today_df.iloc[:6]
    if len(open_range) < 3:
        return 'HOLD', 0

    range_high = float(open_range['h'].max())
    range_low = float(open_range['l'].min())
    range_size = range_high - range_low
    if range_size < 1:  # Min range for gold
        return 'HOLD', 0

    current_price = float(ctx['c'].iloc[-1])

    # Breakout above opening range high
    if current_price > range_high:
        return 'BUY', min((current_price - range_high) / range_size * 2 + 0.3, 0.8)

    # Breakout below opening range low
    if current_price < range_low:
        return 'SELL', min((range_low - current_price) / range_size * 2 + 0.3, 0.8)

    return 'HOLD', 0

def s_ict_fvg(ctx, df, idx):
    """ICT Fair Value Gap: trade when price returns to an unfilled FVG."""
    if len(ctx) < 20: return 'HOLD', 0
    fvgs = find_fvg(ctx.iloc[:-5])  # Look at FVGs formed before recent candles
    if not fvgs: return 'HOLD', 0
    current_price = float(ctx['c'].iloc[-1])
    # Find the most recent FVG
    for fvg_type, gap_bot, gap_top, midpoint, fvg_idx in reversed(fvgs):
        if fvg_idx >= len(ctx) - 3:  # Too recent
            continue
        # Price returned to FVG zone
        if fvg_type == 'BULL' and gap_bot <= current_price <= gap_top:
            return 'BUY', min((gap_top - current_price) / (gap_top - gap_bot + 1e-10) * 0.5 + 0.3, 0.8)
        if fvg_type == 'BEAR' and gap_bot <= current_price <= gap_top:
            return 'SELL', min((current_price - gap_bot) / (gap_top - gap_bot + 1e-10) * 0.5 + 0.3, 0.8)
    return 'HOLD', 0

def s_ict_order_block(ctx, df, idx):
    """ICT Order Block: trade when price returns to an identified OB zone."""
    if len(ctx) < 30: return 'HOLD', 0
    obs = find_order_blocks(ctx.iloc[:-5])
    if not obs: return 'HOLD', 0
    current_price = float(ctx['c'].iloc[-1])
    for ob_type, ob_idx, ob_low, ob_high in reversed(obs):
        if ob_idx >= len(ctx) - 5:
            continue
        if ob_type == 'BULL' and ob_low <= current_price <= ob_high:
            return 'BUY', min((ob_high - current_price) / (ob_high - ob_low + 1e-10) * 0.5 + 0.3, 0.75)
        if ob_type == 'BEAR' and ob_low <= current_price <= ob_high:
            return 'SELL', min((current_price - ob_low) / (ob_high - ob_low + 1e-10) * 0.5 + 0.3, 0.75)
    return 'HOLD', 0

def s_ict_liquidity_sweep(ctx, df, idx):
    """ICT Liquidity Sweep: price sweeps a swing high/low then reverses."""
    result = find_liquidity_sweeps(ctx, df, idx)
    if result:
        sig, conf = result
        return sig, conf
    return 'HOLD', 0

def s_ict_bos(ctx, df, idx):
    """ICT Break of Structure: trade breakouts of swing points with confirmation."""
    if len(ctx) < 20: return 'HOLD', 0
    highs, lows = find_swing_highs_lows(ctx, lookback=5)
    if not highs or not lows: return 'HOLD', 0
    current_price = float(ctx['c'].iloc[-1])
    prev_price = float(ctx['c'].iloc[-2]) if len(ctx) > 1 else current_price

    # BOS UP: price breaks above most recent swing high
    if highs:
        last_high = highs[-1][1]
        if current_price > last_high and prev_price <= last_high:
            # Confirm with momentum: last candle closed strong
            return 'BUY', min((current_price - last_high) / last_high * 100 * 3, 0.7)

    # BOS DOWN: price breaks below most recent swing low
    if lows:
        last_low = lows[-1][1]
        if current_price < last_low and prev_price >= last_low:
            return 'SELL', min((last_low - current_price) / last_low * 100 * 3, 0.7)

    return 'HOLD', 0

# ============================================================
# REGISTRY
# ============================================================
# Format: (name, signal_func, use_kronos_filter)

STRATEGIES = [
    # Kronos-Primary family
    ("Kronos Only", s_kronos_only, False),
    ("Krono AntiTr", s_kronos_antitrend, False),
    ("Kronos+ADX", s_kronos_adx, False),
    ("Kronos+VWAP", s_kronos_vwap, False),
    ("Kronos+MACD", s_kronos_macd, False),
    ("Kronos+RSI", s_kronos_rsi, False),

    # Kronos with adaptive TP/SL (scales to prediction magnitude)
    ("Krono Adapt", s_kronos_only, False),  # adaptive_tp_sl used below
    ("Kron AntiAda", s_kronos_antitrend, False),  # adaptive_tp_sl used below

    # ADX-Primary family (Wilder's proper)
    ("ADX Wilder", s_adx_only, False),

    # Classic Technical (properly implemented)
    ("VWAP Trend", s_vwap_trend, False),
    ("VWAP MeanRev", s_vwap_mr, False),
    ("RSI Extreme", s_rsi_extreme, False),
    ("MACD Cross", s_macd_cross, False),
    ("Bollinger %B", s_bollinger_mr, False),
    ("Stoch Extreme", s_stoch_extreme, False),
    ("MACD+RSI Combo", s_macd_rsi_combo, False),
    ("Ichimoku TK", s_ichimoku_tk, False),

    # ICT/SMC/ORB Institutional
    ("ORB Breakout", s_orb, False),
    ("ICT FVG", s_ict_fvg, False),
    ("ICT OrderBlock", s_ict_order_block, False),
    ("ICT LiqSweep", s_ict_liquidity_sweep, False),
    ("ICT BOS", s_ict_bos, False),
]

# Kronos-hybrid ICT variants
ICT_HYBRIDS = [
    ("Kronos+ORB", s_orb, True),
    ("Kronos+FVG", s_ict_fvg, True),
    ("Kronos+OB", s_ict_order_block, True),
    ("Kronos+LiqSwp", s_ict_liquidity_sweep, True),
    ("Kronos+BOS", s_ict_bos, True),
]

# ADX+Kronos combo uses BOTH (ADX filter + Kronos direction)
HYBRIDS = [
    ("ADX+Kronos", s_adx_kronos, False),
]

ALL_STRATS = STRATEGIES + ICT_HYBRIDS + HYBRIDS

# ============================================================
# MAIN
# ============================================================
def main():
    print("="*60)
    print("  GOLD STRATEGY BACKTEST V3")
    print(f"  Capital: ${CAPITAL} | Size: {POS_SIZE} oz | Lev: 1:{LEV}")
    print(f"  TP: {TP_PCT}% | SL: {SL_PCT}% | Windows: {args.windows}")
    print("="*60, flush=True)

    df = fetch_gold_data(args.days)
    LOOKBACK = 100
    indices = list(range(LOOKBACK+5, min(len(df)-5, args.windows+LOOKBACK+5), 2))
    if args.quick:
        indices = indices[:200]
    print(f"Windows: {len(indices)} ({len(indices)*10/60:.0f}h)", flush=True)

    # Pre-compute Kronos
    if not args.no_kronos:
        load_kronos()
        precompute_kronos(df, indices)

    print(f"\nStrategies: {len(ALL_STRATS)}\n", flush=True)

    ADAPTIVE_STRATS = {"Krono Adapt", "Kron AntiAda"}

    results = {}
    for i, (name, sf, uk) in enumerate(ALL_STRATS):
        print(f"\n--- [{i+1}/{len(ALL_STRATS)}] {name} ---", flush=True)
        adaptive = name in ADAPTIVE_STRATS
        r = run_strategy(name, sf, df, indices, use_kronos=uk, adaptive_tp_sl=adaptive)
        results[name] = r
        print(f"  -> ${r['final']:.2f} ({r['return_pct']:+.2f}%) | {r['trade_count']}t | WR:{r['win_rate']:.1f}% | PF:{r['profit_factor']:.2f}", flush=True)

    # RANKING
    print("\n"+"="*90); print("  FINAL RANKING"); print("="*90)
    print(f"{'Rk':>3} {'Strategy':<22} {'Final':>10} {'Return':>9} {'Trades':>6} {'WR':>5} {'PF':>6} {'TP/SL/E':>11}")
    print("-"*90)
    sorted_r = sorted(results.items(), key=lambda x: x[1]['return_pct'], reverse=True)
    for rk, (n, r) in enumerate(sorted_r, 1):
        print(f"{rk:>3} {n:<22} ${r['final']:>8.2f} {r['return_pct']:>+8.2f}% {r['trade_count']:>6} {r['win_rate']:>4.1f}% {r['profit_factor']:>5.2f} {r['tp']}/{r['sl']}/{r['exp']}", flush=True)

    if args.save:
        out = {}
        for n, r in results.items():
            out[n] = {k: v for k, v in r.items() if k != 'trades'}
            out[n]['_trades_sample'] = r['trades'][-5:] if r['trades'] else []
        json.dump(out, open(PROJECT_DIR/'gold_v3_results.json', 'w'), indent=2)
        print(f"\nSaved to gold_v3_results.json", flush=True)
    print("="*90, flush=True)

if __name__ == '__main__':
    main()
