import { NextResponse } from 'next/server';
import { readFileSync, existsSync } from 'fs';
import { join } from 'path';

const PROJECT_DIR = join('D:', 'Programming', 'AiProjects', 'Kronos(reviewing)', 'Kronos');
const BINANCE_API = 'https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT';

function readLedger(name: string) {
  const path = join(PROJECT_DIR, name);
  if (!existsSync(path)) return null;
  return JSON.parse(readFileSync(path, 'utf-8'));
}

function enrich(ledger: any, label: string, btcPrice: number | null) {
  if (!ledger) return null;
  const trades = (ledger.trades || []).slice().reverse();
  const wins = trades.filter((t: any) => (t.pnl_usdt || 0) > 0);
  const losses = trades.filter((t: any) => (t.pnl_usdt || 0) <= 0);
  const totalTrades = trades.length;

  const balance = ledger.balance ?? 0;
  const positions = ledger.positions || [];
  const initialCapital = ledger.initial_capital ?? 1000;

  let totalMargin = 0;
  let totalUnrealized = 0;
  const enrichedPositions = positions.map((p: any) => {
    const margin = (p.size * p.entry_price) / 200;
    totalMargin += margin;
    let upnl = 0;
    if (btcPrice) {
      if (p.side === 'buy') upnl = (btcPrice - p.entry_price) * p.size;
      else upnl = (p.entry_price - btcPrice) * p.size;
    }
    totalUnrealized += upnl;
    return {
      side: p.side, size: p.size, entry_price: p.entry_price,
      tp_pct: p.tp_pct ?? p.tp, sl_pct: p.sl_pct ?? p.sl,
      scalp: p.scalp ?? false, margin: Math.round(margin * 100) / 100,
      unrealized: Math.round(upnl * 100) / 100,
    };
  });

  const totalEquity = Math.round((balance + totalMargin + totalUnrealized) * 100) / 100;
  const pnl = Math.round((totalEquity - initialCapital) * 100) / 100;
  const winRate = totalTrades > 0 ? Math.round(wins.length / totalTrades * 100) : 0;

  let currentStreak = 0;
  for (let i = trades.length - 1; i >= 0; i--) {
    const isWin = (trades[i].pnl_usdt || 0) > 0;
    if (i === trades.length - 1) currentStreak = isWin ? 1 : -1;
    else if ((currentStreak > 0) === isWin) currentStreak += isWin ? 1 : -1;
    else break;
  }

  const grossProfit = wins.reduce((s: number, t: any) => s + Math.max(0, t.pnl_usdt), 0);
  const grossLoss = losses.reduce((s: number, t: any) => s + Math.max(0, -t.pnl_usdt), 0);
  const profitFactor = grossLoss > 0 ? Math.round(grossProfit / grossLoss * 100) / 100 : grossProfit > 0 ? 999 : 0;
  const avgWin = wins.length > 0 ? Math.round(grossProfit / wins.length * 100) / 100 : 0;
  const avgLoss = losses.length > 0 ? Math.round(-grossLoss / losses.length * 100) / 100 : 0;

  return {
    label, cashBalance: Math.round(balance * 100) / 100,
    marginInUse: Math.round(totalMargin * 100) / 100,
    unrealizedPnl: Math.round(totalUnrealized * 100) / 100,
    totalEquity, pnl, initialCapital,
    btcPrice,
    positions: enrichedPositions, trades,
    wins: wins.length, losses: losses.length, totalTrades, winRate,
    currentStreak, profitFactor, avgWin, avgLoss,
    largestWin: wins.length > 0 ? Math.round(Math.max(...wins.map((t: any) => t.pnl_usdt)) * 100) / 100 : 0,
    largestLoss: losses.length > 0 ? Math.round(Math.max(...losses.map((t: any) => -t.pnl_usdt)) * 100) / 100 : 0,
    timestamp: Date.now(),
  };
}

export async function GET() {
  let btcPrice: number | null = null;
  try {
    const res = await fetch(BINANCE_API, { signal: AbortSignal.timeout(5000) });
    const data = await res.json();
    btcPrice = parseFloat(data.price);
  } catch {}

  const bot = enrich(readLedger('paper_trades.json'), 'Bot-Only', btcPrice);
  const agent = enrich(readLedger('paper_trades_agent.json'), 'Agent-Assisted', btcPrice);
  return NextResponse.json({ bot, agent });
}
