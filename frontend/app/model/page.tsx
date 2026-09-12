'use client'

import { useEffect, useState } from 'react'
import { cn } from '@/lib/utils'
import { api, MetricsResponse, SeasonAccuracyEntry, Season, HistoryGame, EdgeSummaryResponse, RoiResponse } from '@/lib/api'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Legend, BarChart, Bar,
} from 'recharts'

type Tab = 'performance' | 'edge' | 'history'

function Skeleton({ className }: { className?: string }) {
  return <div className={cn('animate-pulse rounded-xl bg-white/5', className)} />
}

function StatCard({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="bg-card border border-border rounded-xl p-5">
      <p className="text-[10px] font-semibold uppercase tracking-widest text-muted-foreground">{label}</p>
      <p className="text-3xl font-black mt-1 tabular-nums">{value}</p>
      {sub && <p className="text-xs text-muted-foreground mt-1">{sub}</p>}
    </div>
  )
}

function ChartCard({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="bg-card border border-border rounded-xl overflow-hidden">
      <div className="px-5 py-4 border-b border-border">
        <p className="text-sm font-semibold">{title}</p>
      </div>
      <div className="p-5">{children}</div>
    </div>
  )
}

function PerformanceTab() {
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null)
  const [accuracy, setAccuracy] = useState<SeasonAccuracyEntry[] | null>(null)

  useEffect(() => {
    Promise.all([api.metrics(), api.seasonAccuracy()]).then(([m, a]) => {
      setMetrics(m)
      setAccuracy(a.seasons)
    })
  }, [])

  const cal = metrics?.xgboost?.test_2024_2025_calibrated
  const xgb = metrics?.xgboost

  const calibData = cal?.calibration.map((c) => ({
    name: `${(c.predicted * 100).toFixed(0)}%`,
    Model: parseFloat((c.actual * 100).toFixed(1)),
    Perfect: parseFloat((c.predicted * 100).toFixed(1)),
  }))

  return (
    <div className="space-y-6">
      {!metrics ? (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
          {Array.from({ length: 4 }).map((_, i) => <Skeleton key={i} className="h-24" />)}
        </div>
      ) : cal && xgb ? (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
          <StatCard label="Test AUC" value={cal.roc_auc.toFixed(4)} sub="2024–2025 seasons" />
          <StatCard label="Test Brier" value={cal.brier.toFixed(4)} sub="Lower is better" />
          <StatCard label="Val AUC" value={xgb.val_2023.roc_auc.toFixed(4)} sub="2023 season" />
          <StatCard label="Val Brier" value={xgb.val_2023.brier.toFixed(4)} sub="2023 season" />
        </div>
      ) : null}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {accuracy ? (
          <ChartCard title="Season-by-Season Accuracy">
            <ResponsiveContainer width="100%" height={260}>
              <LineChart data={accuracy} margin={{ top: 4, right: 8, bottom: 4, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e2d3f" />
                <XAxis dataKey="season" tick={{ fontSize: 11, fill: '#5b7491' }} tickLine={false} />
                <YAxis tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} tick={{ fontSize: 11, fill: '#5b7491' }} tickLine={false} axisLine={false} width={44} />
                <Tooltip formatter={(v: unknown) => typeof v === 'number' ? `${(v * 100).toFixed(1)}%` : ''} labelFormatter={(l) => `Season ${l}`} contentStyle={{ background: '#0f1923', border: '1px solid #1e2d3f', borderRadius: 8, fontSize: 12 }} />
                <Line type="monotone" dataKey="accuracy" stroke="#00e676" strokeWidth={2} dot={{ r: 3, fill: '#00e676' }} activeDot={{ r: 5 }} name="Accuracy" />
              </LineChart>
            </ResponsiveContainer>
          </ChartCard>
        ) : <Skeleton className="h-80" />}

        {calibData ? (
          <ChartCard title="Calibration — Reliability Diagram">
            <ResponsiveContainer width="100%" height={260}>
              <LineChart data={calibData} margin={{ top: 4, right: 8, bottom: 20, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e2d3f" />
                <XAxis dataKey="name" tick={{ fontSize: 11, fill: '#5b7491' }} tickLine={false} label={{ value: 'Predicted WP', position: 'insideBottom', offset: -8, fontSize: 11, fill: '#5b7491' }} height={36} />
                <YAxis tickFormatter={(v) => `${v}%`} tick={{ fontSize: 11, fill: '#5b7491' }} tickLine={false} axisLine={false} width={44} />
                <Tooltip formatter={(v: unknown) => typeof v === 'number' ? `${v}%` : ''} contentStyle={{ background: '#0f1923', border: '1px solid #1e2d3f', borderRadius: 8, fontSize: 12 }} />
                <Legend wrapperStyle={{ fontSize: 12, color: '#5b7491' }} />
                <Line type="monotone" dataKey="Perfect" stroke="#5b7491" strokeDasharray="5 4" strokeWidth={1.5} dot={false} />
                <Line type="monotone" dataKey="Model" stroke="#00e676" strokeWidth={2} dot={{ r: 4, fill: '#00e676' }} activeDot={{ r: 6 }} />
              </LineChart>
            </ResponsiveContainer>
          </ChartCard>
        ) : <Skeleton className="h-80" />}
      </div>
    </div>
  )
}

function EdgeTab() {
  const [edge, setEdge] = useState<EdgeSummaryResponse | null>(null)
  const [roi, setRoi] = useState<RoiResponse | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    Promise.all([api.edgeSummary(), api.roi()]).then(([e, r]) => {
      setEdge(e)
      setRoi(r)
    }).finally(() => setLoading(false))
  }, [])

  if (loading) return <div className="space-y-4">{Array.from({ length: 3 }).map((_, i) => <Skeleton key={i} className="h-64" />)}</div>
  if (!edge) return null

  const bucketData = edge.buckets.map((b) => ({
    bucket: b.bucket,
    'Model WP': parseFloat((b.model_wp_mean * 100).toFixed(1)),
    'Actual Win%': parseFloat((b.actual_win_rate * 100).toFixed(1)),
  }))

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 sm:grid-cols-3 gap-4">
        <StatCard label="Games Evaluated" value={edge.n_games.toString()} sub={`${edge.seasons[0]}–${edge.seasons[edge.seasons.length - 1]}`} />
        <StatCard label="Mean Edge" value={`${(edge.overall.mean_edge * 100).toFixed(1)}%`} sub={`σ = ${(edge.overall.std_edge * 100).toFixed(1)}%`} />
        {roi && <StatCard label="Breakeven Win Rate" value={`${(roi.breakeven_wr * 100).toFixed(1)}%`} sub={`at ${roi.odds} odds`} />}
      </div>

      <ChartCard title="Edge Bucket — Model WP vs Actual Win Rate">
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={bucketData} margin={{ top: 4, right: 8, bottom: 36, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e2d3f" vertical={false} />
            <XAxis dataKey="bucket" tick={{ fontSize: 10, fill: '#5b7491' }} tickLine={false} angle={-25} textAnchor="end" height={48} />
            <YAxis tickFormatter={(v) => `${v}%`} tick={{ fontSize: 11, fill: '#5b7491' }} tickLine={false} axisLine={false} width={40} />
            <Tooltip formatter={(v: unknown, name: unknown) => [typeof v === 'number' ? `${v}%` : '', String(name)]} contentStyle={{ background: '#0f1923', border: '1px solid #1e2d3f', borderRadius: 8, fontSize: 12 }} />
            <Legend wrapperStyle={{ fontSize: 12, color: '#5b7491' }} />
            <Bar dataKey="Model WP" fill="#00e676" radius={[3, 3, 0, 0]} opacity={0.8} />
            <Bar dataKey="Actual Win%" fill="#5b7491" radius={[3, 3, 0, 0]} opacity={0.7} />
          </BarChart>
        </ResponsiveContainer>
      </ChartCard>

      {roi && (
        <div className="bg-card border border-border rounded-xl overflow-hidden">
          <div className="px-5 py-4 border-b border-border">
            <p className="text-sm font-semibold">Flat-Bet ROI by Week Group</p>
            <p className="text-xs text-muted-foreground mt-0.5">threshold ≥ {(roi.edge_thresh * 100).toFixed(0)}% · {roi.odds} odds</p>
          </div>
          <div>
            <div className="grid grid-cols-4 px-5 py-2.5 text-[10px] font-semibold uppercase tracking-widest text-muted-foreground border-b border-border">
              <span>Week Group</span><span className="text-right">Bets</span><span className="text-right">Win Rate</span><span className="text-right">ROI</span>
            </div>
            {roi.overall.map((r) => (
              <div key={r.week_group} className="grid grid-cols-4 px-5 py-3 border-b border-border last:border-0 items-center">
                <span className="text-sm">{r.week_group}</span>
                <span className="text-right text-sm tabular-nums text-muted-foreground">{r.n}</span>
                <span className="text-right text-sm tabular-nums">{(r.win_rate * 100).toFixed(1)}%</span>
                <span className={cn('text-right text-sm font-bold tabular-nums', r.roi >= 0 ? 'text-primary' : 'text-[#ff4444]')}>
                  {r.roi >= 0 ? '+' : ''}{(r.roi * 100).toFixed(1)}%
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

function HistoryTab() {
  const [seasons, setSeasons] = useState<Season[]>([])
  const [season, setSeason] = useState<number | null>(null)
  const [week, setWeek] = useState<number | null>(null)
  const [games, setGames] = useState<HistoryGame[] | null>(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    api.seasons().then(({ seasons }) => {
      setSeasons(seasons)
      if (seasons.length) {
        const latest = seasons[seasons.length - 1]
        setSeason(latest.season)
        setWeek(latest.max_week)
      }
    })
  }, [])

  useEffect(() => {
    if (!season || !week) return
    setLoading(true)
    api.history(season, week).then(({ games }) => setGames(games)).catch(() => setGames([])).finally(() => setLoading(false))
  }, [season, week])

  const currentSeason = seasons.find((s) => s.season === season)
  const weeks = currentSeason ? Array.from({ length: currentSeason.max_week - currentSeason.min_week + 1 }, (_, i) => currentSeason.min_week + i) : []
  const correct = games?.filter((g) => g.model_correct === 1).length ?? 0
  const total = games?.length ?? 0

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3 flex-wrap">
        <select value={season ?? ''} onChange={(e) => { const s = +e.target.value; setSeason(s); const m = seasons.find((x) => x.season === s); setWeek(m?.max_week ?? null) }} className="bg-card border border-border rounded-lg px-3 py-1.5 text-sm text-foreground focus:outline-none focus:border-primary">
          {seasons.map((s) => <option key={s.season} value={s.season}>{s.season}</option>)}
        </select>
        <select value={week ?? ''} onChange={(e) => setWeek(+e.target.value)} disabled={!weeks.length} className="bg-card border border-border rounded-lg px-3 py-1.5 text-sm text-foreground focus:outline-none focus:border-primary disabled:opacity-50">
          {weeks.map((w) => <option key={w} value={w}>Week {w}</option>)}
        </select>
        {total > 0 && (
          <span className="text-sm text-muted-foreground">
            <span className="text-primary font-bold">{correct}/{total}</span> correct ({((correct / total) * 100).toFixed(0)}%)
          </span>
        )}
      </div>

      {loading ? (
        <div className="space-y-2">{Array.from({ length: 6 }).map((_, i) => <Skeleton key={i} className="h-14" />)}</div>
      ) : !games ? null : games.length === 0 ? (
        <p className="text-muted-foreground">No games found.</p>
      ) : (
        <div className="bg-card border border-border rounded-xl overflow-hidden">
          <div className="grid grid-cols-[1fr_auto_auto_auto_auto_auto] px-5 py-2.5 text-[10px] font-semibold uppercase tracking-widest text-muted-foreground border-b border-border gap-4">
            <span>Matchup</span><span>Score</span><span>Market WP</span><span>Model WP</span><span>Edge</span><span>Result</span>
          </div>
          {games.map((g) => {
            const homeWon = g.home_won === 1
            const correct = g.model_correct === 1
            const modelHome = g.model_wp > 0.5
            return (
              <div key={g.game_id} className="grid grid-cols-[1fr_auto_auto_auto_auto_auto] px-5 py-3.5 items-center gap-4 border-b border-border last:border-0">
                <p className="text-sm font-medium">
                  <span className={homeWon ? 'text-muted-foreground' : 'font-bold'}>{g.away_team}</span>
                  <span className="text-muted-foreground mx-1 font-normal">@</span>
                  <span className={homeWon ? 'font-bold' : 'text-muted-foreground'}>{g.home_team}</span>
                </p>
                <span className="text-sm tabular-nums text-muted-foreground">{g.away_score}–{g.home_score}</span>
                <span className="text-sm tabular-nums">{(g.market_wp * 100).toFixed(1)}%</span>
                <span className={cn('text-sm tabular-nums', modelHome !== homeWon ? 'text-muted-foreground' : '')}>{(g.model_wp * 100).toFixed(1)}%</span>
                <span className={cn('text-sm tabular-nums font-semibold', g.edge > 0 ? 'text-primary' : 'text-[#ff4444]')}>{g.edge > 0 ? '+' : ''}{(g.edge * 100).toFixed(1)}%</span>
                <span className={cn('text-xs font-bold px-2 py-1 rounded', correct ? 'bg-primary/15 text-primary' : 'bg-white/6 text-muted-foreground')}>
                  {correct ? '✓' : '✗'} {homeWon ? g.home_team : g.away_team}
                </span>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}

const TABS: { id: Tab; label: string }[] = [
  { id: 'performance', label: 'Performance' },
  { id: 'edge', label: 'Edge Analysis' },
  { id: 'history', label: 'History' },
]

export default function ModelPage() {
  const [tab, setTab] = useState<Tab>('performance')

  return (
    <div className="max-w-7xl mx-auto px-4 py-8">
      <div className="mb-6">
        <p className="text-xs font-semibold uppercase tracking-widest text-muted-foreground mb-1">
          XGBoost · Isotonic Calibration · 2016–2025
        </p>
        <h1 className="text-3xl font-black">Model</h1>
      </div>

      <div className="flex gap-1 mb-6 bg-white/4 p-1 rounded-xl w-fit">
        {TABS.map((t) => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            className={cn(
              'px-4 py-1.5 rounded-lg text-sm font-medium transition-colors',
              tab === t.id ? 'bg-card text-foreground shadow' : 'text-muted-foreground hover:text-foreground'
            )}
          >
            {t.label}
          </button>
        ))}
      </div>

      {tab === 'performance' && <PerformanceTab />}
      {tab === 'edge' && <EdgeTab />}
      {tab === 'history' && <HistoryTab />}
    </div>
  )
}
