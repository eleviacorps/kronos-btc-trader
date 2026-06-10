import { NextResponse } from 'next/server';
import { readFileSync, existsSync } from 'fs';
import { join } from 'path';

const PROJECT_DIR = join('D:', 'Programming', 'AiProjects', 'Kronos(reviewing)', 'Kronos');

function readLedger(name: string) {
  const path = join(PROJECT_DIR, name);
  if (!existsSync(path)) return null;
  return JSON.parse(readFileSync(path, 'utf-8'));
}

export async function GET() {
  const entries: string[] = [];
  for (const [label, file] of [['Bot-Only', 'paper_trades.json'], ['Agent-Assisted', 'paper_trades_agent.json']] as const) {
    const ledger = readLedger(file);
    if (!ledger) continue;
    entries.push(`# ${label}\n`);
    entries.push('Timestamp,Side,Close Reason,Entry Price,Exit Price,Size (BTC),PnL (USDT),PnL %,Balance After\n');
    for (const t of (ledger.trades || [])) {
      entries.push(`${t.timestamp},${t.side},${t.close_reason},${t.entry_price},${t.exit_price},${t.size_btc},${t.pnl_usdt},${t.pnl_pct},${t.balance_after}\n`);
    }
    entries.push('\n');
  }
  return new NextResponse(entries.join(''), {
    headers: {
      'Content-Type': 'text/csv',
      'Content-Disposition': 'attachment; filename="kronos_trades.csv"',
    },
  });
}
