import { NextResponse } from 'next/server';
import { execSync } from 'child_process';
import { join } from 'path';

const PROJECT_DIR = join('D:', 'Programming', 'AiProjects', 'Kronos(reviewing)', 'Kronos');
const VENV_PY = join(PROJECT_DIR, '.venv', 'Scripts', 'python.exe');
const EXEC = join(PROJECT_DIR, 'kronos_exec.py');

export async function POST(req: Request) {
  const { account, index } = await req.json();
  const ledgers: Record<string, string> = {
    live: 'paper_trades_live.json',
  };
  const ledger = ledgers[account as string];
  if (!ledger) return NextResponse.json({ error: 'unknown account' }, { status: 400 });
  try {
    if (index !== undefined) {
      execSync(`"${VENV_PY}" "${EXEC}" --scalp --close-trade ${index} --ledger ${ledger}`, { timeout: 20000, windowsHide: true });
    } else {
      execSync(`"${VENV_PY}" "${EXEC}" --scalp --paper close --ledger ${ledger}`, { timeout: 20000, windowsHide: true });
    }
    return NextResponse.json({ success: true });
  } catch (e: any) {
    return NextResponse.json({ error: e.message }, { status: 500 });
  }
}
