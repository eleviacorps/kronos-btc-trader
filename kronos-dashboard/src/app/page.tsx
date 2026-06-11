"use client";

import { useEffect, useState, useCallback } from "react";

interface Trade {
  timestamp: string; side: string; close_reason: string;
  entry_price: number; exit_price: number; size_btc: number;
  pnl_usdt: number; pnl_pct: number; balance_after: number;
}

interface Position {
  side: string; size: number; entry_price: number;
  tp_pct: number; sl_pct: number; scalp: boolean;
  margin: number; unrealized: number;
}

interface AccountData {
  label: string; cashBalance: number; marginInUse: number;
  unrealizedPnl: number; totalEquity: number; pnl: number;
  initialCapital: number; btcPrice: number | null;
  positions: Position[]; trades: Trade[];
  wins: number; losses: number; totalTrades: number;
  winRate: number; currentStreak: number; profitFactor: number;
  avgWin: number; avgLoss: number;
  largestWin: number; largestLoss: number; timestamp: number;
}

interface LedgerResponse { bot: AccountData; agent: AccountData; hc: AccountData; hc_agent: AccountData; }

function fmt(v: number | null | undefined): string {
  if (v == null || isNaN(v as number)) return "0.00";
  return v.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}
function pnlClass(v: number | null | undefined): string {
  if (v == null) return "text-gray-400";
  return v > 0 ? "text-emerald-400" : v < 0 ? "text-red-400" : "text-gray-400";
}
function sideClass(s: string): string {
  return s === "buy" ? "text-emerald-400" : "text-red-400";
}
function shortTime(ts: string): string {
  return ts?.slice(5, 16) || "?";
}

function Controls({ account, onAction }: { account: string; onAction: () => void }) {
  const [loading, setLoading] = useState(false);
  const doAction = async (action: string, idx?: number) => {
    setLoading(true);
    try {
      if (action === 'reset') await fetch('/api/ledger/reset', { method: 'POST', body: JSON.stringify({ account }) });
      else if (action === 'close') await fetch('/api/ledger/close', { method: 'POST', body: JSON.stringify({ account, index: idx }) });
      else if (action === 'close-all') await fetch('/api/ledger/close', { method: 'POST', body: JSON.stringify({ account }) });
      onAction();
    } finally { setLoading(false); }
  };
  return (
    <div className="flex flex-wrap gap-2 mt-3">
      <button onClick={() => doAction('reset')} disabled={loading}
        className="text-xs px-3 py-1.5 rounded-lg bg-red-900/40 border border-red-800/50 text-red-300 hover:bg-red-800/50 transition disabled:opacity-40">
        Reset {account}
      </button>
    </div>
  );
}

function FinBox({ label, value, cls }: { label: string; value: string; cls?: string }) {
  return (
    <div className="bg-gray-800/60 rounded-xl p-3">
      <div className="text-[10px] text-gray-500 uppercase tracking-wider mb-0.5">{label}</div>
      <div className={`text-sm font-bold tracking-tight ${cls || "text-gray-100"}`}>{value}</div>
    </div>
  );
}

function AccountCard({ data, onClose }: { data: AccountData; onClose?: () => void }) {
  const [showTrades, setShowTrades] = useState(false);
  const [refreshKey, setRefreshKey] = useState(0);
  const refresh = useCallback(() => setRefreshKey(k => k + 1), []);

  return (
    <div className="bg-gray-900/80 backdrop-blur-sm rounded-2xl border border-gray-800 p-5 shadow-lg">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-base font-semibold tracking-tight">{data.label}</h2>
        <div className="flex items-center gap-2">
          <span className="text-[10px] text-gray-500 px-2 py-1 rounded-full bg-gray-800/80">{data.totalTrades} trades</span>
          <a href="/api/ledger/export" className="text-[10px] text-gray-500 hover:text-gray-300 underline">CSV</a>
        </div>
      </div>

      {data.btcPrice && (
        <div className="text-xs text-gray-500 mb-3">BTC/USDT: <span className="text-gray-300 font-mono">${fmt(data.btcPrice)}</span></div>
      )}

      <div className="grid grid-cols-4 gap-2 mb-3">
        <FinBox label="Cash" value={`$${fmt(data.cashBalance)}`} />
        <FinBox label="Margin" value={`$${fmt(data.marginInUse)}`} cls="text-amber-400" />
        <FinBox label="Unreal." value={`$${fmt(data.unrealizedPnl)}`} cls={pnlClass(data.unrealizedPnl)} />
        <FinBox label="Equity" value={`$${fmt(data.totalEquity)}`} cls="text-blue-400" />
      </div>

      <div className="bg-gray-800/40 rounded-xl p-3 mb-3 text-center">
        <div className="text-[10px] text-gray-500 uppercase tracking-wider">PnL</div>
        <div className={`text-xl font-bold tracking-tight ${pnlClass(data.pnl)}`}>
          {data.pnl >= 0 ? "+" : ""}${fmt(data.pnl)}
        </div>
      </div>

      <div className="grid grid-cols-4 gap-2 mb-3 text-center">
        <div className="bg-gray-800/40 rounded-lg p-2">
          <div className="text-[10px] text-gray-500">Win Rate</div>
          <div className="text-xs font-semibold">{data.winRate}%</div>
        </div>
        <div className="bg-gray-800/40 rounded-lg p-2">
          <div className="text-[10px] text-gray-500">W/L</div>
          <div className="text-xs font-semibold">{data.wins}/{data.losses}</div>
        </div>
        <div className="bg-gray-800/40 rounded-lg p-2">
          <div className="text-[10px] text-gray-500">Streak</div>
          <div className={`text-xs font-semibold ${data.currentStreak > 0 ? "text-emerald-400" : data.currentStreak < 0 ? "text-red-400" : ""}`}>
            {data.currentStreak > 0 ? `+${data.currentStreak}` : data.currentStreak || "—"}
          </div>
        </div>
        <div className="bg-gray-800/40 rounded-lg p-2">
          <div className="text-[10px] text-gray-500">PF</div>
          <div className="text-xs font-semibold">{data.profitFactor > 999 ? "∞" : data.profitFactor}</div>
        </div>
      </div>

      {data.totalTrades > 0 && (
        <div className="grid grid-cols-4 gap-2 mb-3 text-center">
          <div className="bg-gray-800/30 rounded-lg p-1.5">
            <div className="text-[9px] text-gray-600">Avg Win</div>
            <div className="text-[11px] font-semibold text-emerald-400">${fmt(data.avgWin)}</div>
          </div>
          <div className="bg-gray-800/30 rounded-lg p-1.5">
            <div className="text-[9px] text-gray-600">Avg Loss</div>
            <div className="text-[11px] font-semibold text-red-400">${fmt(data.avgLoss)}</div>
          </div>
          <div className="bg-gray-800/30 rounded-lg p-1.5">
            <div className="text-[9px] text-gray-600">Best</div>
            <div className="text-[11px] font-semibold text-emerald-400">${fmt(data.largestWin)}</div>
          </div>
          <div className="bg-gray-800/30 rounded-lg p-1.5">
            <div className="text-[9px] text-gray-600">Worst</div>
            <div className="text-[11px] font-semibold text-red-400">${fmt(data.largestLoss)}</div>
          </div>
        </div>
      )}

      {data.positions.length > 0 && (
        <div className="mb-3">
          <div className="text-[10px] text-gray-500 uppercase tracking-wider mb-1.5">Open Positions</div>
          <div className="space-y-1.5">
            {data.positions.map((p, i) => (
              <div key={i} className="bg-gray-800/50 rounded-xl p-2.5 text-xs">
                <div className="flex items-center justify-between mb-1">
                  <span className={`font-semibold ${sideClass(p.side)}`}>{p.side.toUpperCase()}</span>
                  <span className="text-gray-300">{p.size} BTC @ ${fmt(p.entry_price)}</span>
                </div>
                <div className="flex items-center justify-between text-[10px] text-gray-500">
                  <span>TP {p.tp_pct}% / SL {p.sl_pct}%</span>
                  <span>Margin ${fmt(p.margin)}</span>
                  <span className={pnlClass(p.unrealized)}>Unr. ${fmt(p.unrealized)}</span>
                </div>
              </div>
            ))}
            <div className="flex gap-1.5 mt-1.5">
              <button onClick={() => { fetch('/api/ledger/close', { method: 'POST', body: JSON.stringify({ account: data.label === 'Bot-Only' ? 'bot' : 'agent' }) }) }}
                className="text-[10px] px-2.5 py-1 rounded-lg bg-red-900/30 border border-red-800/40 text-red-300 hover:bg-red-800/40 transition">
                Close All
              </button>
            </div>
          </div>
        </div>
      )}

      <Controls account={data.label === 'Bot-Only' ? 'bot' : 'agent'} onAction={refresh} />

      <button onClick={() => setShowTrades(!showTrades)}
        className="w-full mt-3 text-[10px] text-gray-500 hover:text-gray-300 transition text-center py-1 rounded-lg bg-gray-800/20 hover:bg-gray-800/40">
        {showTrades ? "Hide" : "Show"} Trade History ({data.totalTrades})
      </button>

      {showTrades && (
        <div className="mt-2 max-h-64 overflow-y-auto space-y-1">
          {data.trades.length === 0 ? (
            <div className="text-gray-600 text-xs text-center py-3">No trades yet</div>
          ) : (
            data.trades.map((t, i) => (
              <div key={i} className="bg-gray-800/30 rounded-lg px-2.5 py-1.5 flex items-center justify-between text-[11px]">
                <div className="flex items-center gap-2 min-w-0">
                  <span className={`w-1.5 h-1.5 rounded-full shrink-0 ${t.pnl_usdt > 0 ? "bg-emerald-400" : "bg-red-400"}`} />
                  <span className="text-gray-500 shrink-0">{shortTime(t.timestamp)}</span>
                  <span className={`font-medium shrink-0 ${sideClass(t.side)}`}>{t.side}</span>
                  <span className="text-gray-600 shrink-0">{t.close_reason}</span>
                  <span className="text-gray-600 truncate">{t.size_btc}B @ {fmt(t.entry_price)}</span>
                </div>
                <span className={`shrink-0 font-medium ${pnlClass(t.pnl_usdt)}`}>
                  {t.pnl_usdt > 0 ? "+" : ""}${fmt(t.pnl_usdt)}
                </span>
              </div>
            ))
          )}
        </div>
      )}
    </div>
  );
}

function SkeletonCard() {
  return (
    <div className="bg-gray-900/80 rounded-2xl border border-gray-800 p-5 shadow-lg animate-pulse">
      <div className="h-4 bg-gray-800 rounded w-24 mb-4" />
      <div className="grid grid-cols-4 gap-2 mb-3">
        {[...Array(4)].map((_, i) => <div key={i} className="bg-gray-800/60 rounded-xl p-3 h-12" />)}
      </div>
      <div className="bg-gray-800/40 rounded-xl p-3 h-14 mb-3" />
      <div className="grid grid-cols-4 gap-2 mb-3">
        {[...Array(4)].map((_, i) => <div key={i} className="bg-gray-800/40 rounded-lg p-2 h-8" />)}
      </div>
    </div>
  );
}

interface CronInfo {
  schedule: string;
  nextRun: string;
  enabled: boolean;
}

function CronBar({ jobs, now }: { jobs: Record<string, CronInfo>; now: number }) {
  const items = [
    { key: 'kronos-btc-hft-scalper', label: 'Bot Scalper' },
    { key: 'kronos-btc-hft-agent', label: 'Agent Judge' },
    { key: 'kronos-btc-hft-hc', label: 'HC Bot' },
    { key: 'kronos-btc-hft-hc-agent', label: 'HC Agent' },
    { key: 'kronos-watchdog-health', label: 'Watchdog' },
  ];
  return (
    <div className="flex gap-2 mb-5 flex-wrap">
      {items.map(({ key, label }) => {
        const job = jobs[key];
        if (!job) return null;
        const next = job.nextRun ? new Date(job.nextRun).getTime() : null;
        const countdown = next ? Math.max(0, Math.floor((next - now) / 1000)) : null;
        const mins = countdown !== null ? Math.floor(countdown / 60) : null;
        const secs = countdown !== null ? countdown % 60 : null;
        return (
          <div key={key} className={`bg-gray-800/60 rounded-xl px-3 py-2 text-xs ${job.enabled ? '' : 'opacity-40'}`}>
            <div className="text-gray-500">{label}</div>
            <div className="font-semibold text-gray-200">
              {countdown !== null && countdown > 0
                ? `${mins}m ${String(secs ?? 0).padStart(2, '0')}s`
                : countdown === 0 ? 'Firing...' : '—'}
            </div>
          </div>
        );
      })}
    </div>
  );
}

export default function Dashboard() {
  const [data, setData] = useState<LedgerResponse | null>(null);
  const [cron, setCron] = useState<Record<string, CronInfo> | null>(null);
  const [now, setNow] = useState(Date.now());
  const [key, setKey] = useState(0);

  useEffect(() => {
    async function fetchAll() {
      try {
        const [ledgerRes, cronRes] = await Promise.all([
          fetch("/api/ledger"),
          fetch("/api/cron"),
        ]);
        const ledgerJson = await ledgerRes.json();
        if (ledgerJson.bot) setData(ledgerJson);
        const cronJson = await cronRes.json();
        if (cronJson.jobs) setCron(cronJson.jobs);
      } catch {}
    }
    fetchAll();
    const interval = setInterval(fetchAll, 1000);
    const tick = setInterval(() => setNow(Date.now()), 1000);
    return () => { clearInterval(interval); clearInterval(tick); };
  }, [key]);

  return (
    <div className="min-h-screen p-4 md:p-8 max-w-6xl mx-auto">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h1 className="text-xl font-bold tracking-tight">Kronos Dashboard</h1>
          <p className="text-gray-500 text-xs mt-0.5">BTC/USDT Dual-Account Paper Trading</p>
        </div>
        <div className="text-right text-[10px] text-gray-500">
          <div className="font-mono">{new Date(now).toLocaleTimeString()}</div>
          <div className="text-gray-600">live 1s</div>
        </div>
      </div>

      {cron && <CronBar jobs={cron} now={now} />}

      <div className="flex gap-2 mb-5">
        <button onClick={async () => {
          await fetch('/api/ledger/reset', { method: 'POST', body: JSON.stringify({ account: 'all' }) });
          setKey(k => k + 1);
        }}
          className="text-[11px] px-3 py-1.5 rounded-lg bg-red-900/30 border border-red-800/40 text-red-300 hover:bg-red-800/40 transition">
          Reset Both Accounts
        </button>
      </div>

      <div className="grid md:grid-cols-2 xl:grid-cols-4 gap-5">
        {data ? (
          <>
            <AccountCard key={`bot-${key}`} data={data.bot} />
            <AccountCard key={`agent-${key}`} data={data.agent} />
            <AccountCard key={`hc-${key}`} data={data.hc} />
            <AccountCard key={`hc-agent-${key}`} data={data.hc_agent} />
          </>
        ) : (
          <>
            {[...Array(4)].map((_, i) => <SkeletonCard key={i} />)}
          </>
        )}
      </div>

      <div className="mt-8 text-center text-[10px] text-gray-700">
        Kronos BTC Trading System · 1:200 Leverage · Scalp Mode
      </div>
    </div>
  );
}
