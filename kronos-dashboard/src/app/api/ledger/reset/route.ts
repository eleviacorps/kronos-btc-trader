import { NextResponse } from 'next/server';
import { writeFileSync, readFileSync } from 'fs';
import { join } from 'path';

const PROJECT_DIR = join('D:', 'Programming', 'AiProjects', 'Kronos(reviewing)', 'Kronos');

function freshLedger() {
  return { balance: 1000.0, initial_capital: 1000.0, positions: [], trades: [], total_pnl: 0.0, cooldown: {} };
}

export async function POST(req: Request) {
  const { account } = await req.json();
  if (account === 'bot') writeFileSync(join(PROJECT_DIR, 'paper_trades.json'), JSON.stringify(freshLedger(), null, 2));
  else if (account === 'agent') writeFileSync(join(PROJECT_DIR, 'paper_trades_agent.json'), JSON.stringify(freshLedger(), null, 2));
  else if (account === 'all') {
    writeFileSync(join(PROJECT_DIR, 'paper_trades.json'), JSON.stringify(freshLedger(), null, 2));
    writeFileSync(join(PROJECT_DIR, 'paper_trades_agent.json'), JSON.stringify(freshLedger(), null, 2));
  } else return NextResponse.json({ error: 'unknown account' }, { status: 400 });
  return NextResponse.json({ success: true });
}
