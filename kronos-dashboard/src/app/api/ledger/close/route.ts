import { NextResponse } from 'next/server';
import { execSync } from 'child_process';
import { join } from 'path';

const PROJECT_DIR = join('D:', 'Programming', 'AiProjects', 'Kronos(reviewing)', 'Kronos');
const VENV_PY = join(PROJECT_DIR, '.venv', 'Scripts', 'python.exe');
const EXEC = join(PROJECT_DIR, 'kronos_exec.py');

export async function POST(req: Request) {
  const { account, index } = await req.json();
  const ledger = account === 'agent' ? 'paper_trades_agent.json' : 'paper_trades.json';
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
