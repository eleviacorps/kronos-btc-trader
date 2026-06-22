import { NextResponse } from 'next/server';
import { writeFileSync, readFileSync } from 'fs';
import { join } from 'path';

const PROJECT_DIR = join('D:', 'Programming', 'AiProjects', 'Kronos(reviewing)', 'Kronos');

function freshLedger() {
  return { balance: 1000.0, initial_capital: 1000.0, positions: [], trades: [], total_pnl: 0.0, cooldown: {} };
}

export async function POST(req: Request) {
  const { account } = await req.json();
  const allLedgers: Record<string, string> = {
    live: 'paper_trades_live.json',
  };
  if (account === 'all') {
    for (const file of Object.values(allLedgers))
      writeFileSync(join(PROJECT_DIR, file), JSON.stringify(freshLedger(), null, 2));
  } else {
    const ledgerFile = allLedgers[account as string];
    if (!ledgerFile) return NextResponse.json({ error: 'unknown account' }, { status: 400 });
    writeFileSync(join(PROJECT_DIR, ledgerFile), JSON.stringify(freshLedger(), null, 2));
  }
  return NextResponse.json({ success: true });
}
