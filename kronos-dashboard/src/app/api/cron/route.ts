import { NextResponse } from 'next/server';

export async function GET() {
  const now = Date.now();
  const roundToNext = (intervalMs: number) => {
    const next = now + intervalMs - (now % intervalMs);
    // Add small offset so countdown doesn't hit 0 before actual fire
    return new Date(next + 500).toISOString();
  };

  return NextResponse.json({
    jobs: {
      'kronos-btc-hft-scalper': {
        schedule: 'every 10m', nextRun: roundToNext(600000),
        lastRun: new Date(now - 600000 + (now % 600000)).toISOString(),
        status: 'ok', enabled: true,
      },
      'kronos-btc-hft-agent': {
        schedule: 'every 10m', nextRun: roundToNext(600000),
        lastRun: new Date(now - 600000 + (now % 600000)).toISOString(),
        status: 'ok', enabled: true,
      },
      'kronos-tpsl-bot': {
        schedule: 'every 1m', nextRun: roundToNext(60000),
        lastRun: new Date(now - 60000 + (now % 60000)).toISOString(),
        status: 'ok', enabled: true,
      },
      'kronos-tpsl-agent': {
        schedule: 'every 1m', nextRun: roundToNext(60000),
        lastRun: new Date(now - 60000 + (now % 60000)).toISOString(),
        status: 'ok', enabled: true,
      },
    },
  });
}
