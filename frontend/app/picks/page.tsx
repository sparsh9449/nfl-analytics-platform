'use client'

import { useEffect, useState } from 'react'
import { cn } from '@/lib/utils'
import { api, PreGamePick } from '@/lib/api'

function Skeleton({ className }: { className?: string }) {
  return <div className={cn('animate-pulse rounded-xl bg-white/5', className)} />
}

function ConfidenceBadge({ level }: { level: 'HIGH' | 'MEDIUM' | 'LOW' }) {
  return (
    <span className={cn(
      'text-[10px] font-bold uppercase tracking-widest px-2 py-0.5 rounded-md',
      level === 'HIGH'   && 'bg-primary/15 text-primary',
      level === 'MEDIUM' && 'bg-yellow-500/15 text-yellow-400',
      level === 'LOW'    && 'bg-white/8 text-muted-foreground',
    )}>
      {level}
    </span>
  )
}

function WpBar({ label, value, accent }: { label: string; value: number; accent?: boolean }) {
  return (
    <div>
      <div className="flex justify-between text-xs mb-1.5">
        <span className="text-muted-foreground">{label}</span>
        <span className="tabular-nums font-semibold">{(value * 100).toFixed(1)}%</span>
      </div>
      <div className="h-1.5 bg-white/8 rounded-full overflow-hidden">
        <div
          className={cn('h-full rounded-full transition-all', accent ? 'bg-primary' : 'bg-white/25')}
          style={{ width: `${value * 100}%` }}
        />
      </div>
    </div>
  )
}

function StatRow({ label, home, away }: { label: string; home: number | null; away: number | null }) {
  if (home == null && away == null) return null
  const fmt = (v: number | null) => v != null ? v.toFixed(1) : '—'
  const homeBetter = home != null && away != null && home > away
  const awayBetter = home != null && away != null && away > home
  return (
    <div className="flex items-center text-xs">
      <span className={cn('w-12 tabular-nums text-right font-medium', homeBetter && 'text-primary')}>{fmt(home)}</span>
      <span className="flex-1 text-center text-muted-foreground text-[10px] uppercase tracking-wide px-2">{label}</span>
      <span className={cn('w-12 tabular-nums font-medium', awayBetter && 'text-primary')}>{fmt(away)}</span>
    </div>
  )
}

function PickCard({ pick }: { pick: PreGamePick }) {
  const absEdge = Math.abs(pick.edge)
  const isSharp = absEdge >= 0.12
  const gameDate = new Date(pick.date)
  const dateStr = gameDate.toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' })

  const spreadLabel = pick.spread_line != null
    ? (pick.spread_line > 0
        ? `${pick.home_team} -${pick.spread_line}`
        : `${pick.away_team} -${Math.abs(pick.spread_line)}`)
    : 'No line'

  const hs = pick.home_stats
  const as = pick.away_stats

  return (
    <div className={cn(
      'bg-card border rounded-xl p-5 space-y-4 transition-colors',
      isSharp ? 'border-primary/40 hover:border-primary/70' : 'border-border hover:border-white/20'
    )}>
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          {isSharp && <span className="inline-block w-1.5 h-1.5 rounded-full bg-primary" />}
          <ConfidenceBadge level={pick.confidence} />
        </div>
        <span className="text-[10px] text-muted-foreground">{dateStr}</span>
      </div>

      {/* Matchup */}
      <div className="flex items-start justify-between gap-4">
        <div>
          <p className="font-bold text-base">
            {pick.away_team} <span className="text-muted-foreground font-normal text-sm">@</span> {pick.home_team}
          </p>
          <p className="text-xs text-muted-foreground mt-0.5">{spreadLabel} · O/U {pick.total_line ?? '—'}</p>
        </div>
        <div className="text-right shrink-0">
          <p className="text-[10px] text-muted-foreground uppercase tracking-widest">Model Edge</p>
          <p className="text-2xl font-black tabular-nums leading-tight text-primary">
            +{(pick.edge * 100).toFixed(1)}%
          </p>
        </div>
      </div>

      {/* WP bars */}
      <div className="space-y-2.5">
        <WpBar label="Market WP (Vegas)" value={pick.market_wp} />
        <WpBar label="Model WP" value={pick.model_wp} accent />
      </div>

      {/* Team stat comparison */}
      {(hs.roll4_pts_scored != null || as.roll4_pts_scored != null) && (
        <div className="border-t border-border pt-3 space-y-1.5">
          <div className="flex items-center text-[10px] uppercase tracking-widest text-muted-foreground mb-1.5">
            <span className="w-12 text-right">{pick.home_team}</span>
            <span className="flex-1 text-center">L4 Avg</span>
            <span className="w-12">{pick.away_team}</span>
          </div>
          <StatRow label="Pts For" home={hs.roll4_pts_scored as number} away={as.roll4_pts_scored as number} />
          <StatRow label="Pts Vs" home={hs.roll4_pts_allowed as number} away={as.roll4_pts_allowed as number} />
          <StatRow label="Win %" home={hs.roll4_win_pct != null ? (hs.roll4_win_pct as number) * 100 : null} away={as.roll4_win_pct != null ? (as.roll4_win_pct as number) * 100 : null} />
        </div>
      )}

      {/* Pick recommendation */}
      <div className="flex items-center justify-between pt-2 border-t border-border">
        <p className="text-xs text-muted-foreground">Model pick</p>
        <span className="px-3 py-1 rounded-lg bg-primary/10 text-primary text-sm font-bold">
          {pick.pick_team} {pick.pick_side === 'home' ? '(home)' : '(away)'}
        </span>
      </div>
    </div>
  )
}

export default function PicksPage() {
  const [picks, setPicks] = useState<PreGamePick[] | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    api.upcomingPicks()
      .then((data) => setPicks(data.picks))
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  const sharpPicks = picks?.filter((p) => Math.abs(p.edge) >= 0.12) ?? []
  const otherPicks = picks?.filter((p) => Math.abs(p.edge) < 0.12) ?? []

  return (
    <div className="max-w-7xl mx-auto px-4 py-8">
      <div className="flex items-end justify-between mb-6 gap-4 flex-wrap">
        <div>
          <p className="text-xs font-semibold uppercase tracking-widest text-muted-foreground mb-1">
            Pre-Game Analysis
          </p>
          <h1 className="text-3xl font-black">Picks</h1>
        </div>
        <div className="text-right">
          <p className="text-xs text-muted-foreground">2026 Season · Week 1</p>
          <p className="text-[10px] text-muted-foreground/60">Lines via DraftKings</p>
        </div>
      </div>

      {/* Model description */}
      <div className="mb-6 p-4 bg-white/3 border border-border rounded-xl grid grid-cols-1 md:grid-cols-3 gap-4">
        <div>
          <p className="text-[10px] uppercase tracking-widest text-muted-foreground mb-1">Pre-game model</p>
          <p className="text-sm font-semibold">XGBoost · Game-level</p>
          <p className="text-xs text-muted-foreground mt-0.5">Trained on 2,751 games (2016–2025)</p>
        </div>
        <div>
          <p className="text-[10px] uppercase tracking-widest text-muted-foreground mb-1">In-game model</p>
          <p className="text-sm font-semibold">XGBoost · Play-level</p>
          <p className="text-xs text-muted-foreground mt-0.5">AUC 0.8495 · anchors pre-game WP</p>
        </div>
        <div>
          <p className="text-[10px] uppercase tracking-widest text-muted-foreground mb-1">Edge = Model − Market</p>
          <p className="text-sm font-semibold">Team form from 2025 season</p>
          <p className="text-xs text-muted-foreground mt-0.5">Last 4 games rolling avg</p>
        </div>
      </div>

      {loading ? (
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
          {Array.from({ length: 8 }).map((_, i) => <Skeleton key={i} className="h-72" />)}
        </div>
      ) : error ? (
        <div className="text-center py-20 text-muted-foreground">
          <p className="text-lg mb-2">Unable to load picks</p>
          <p className="text-xs">{error}</p>
        </div>
      ) : !picks || picks.length === 0 ? (
        <div className="text-center py-20 text-muted-foreground">No scheduled games found.</div>
      ) : (
        <div className="space-y-8">
          {sharpPicks.length > 0 && (
            <div>
              <p className="text-xs font-semibold uppercase tracking-widest text-primary mb-3">
                Strong Model Edge — ≥ 12%
              </p>
              <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
                {sharpPicks.map((p) => <PickCard key={p.game_id} pick={p} />)}
              </div>
            </div>
          )}
          {otherPicks.length > 0 && (
            <div>
              <p className="text-xs font-semibold uppercase tracking-widest text-muted-foreground mb-3">
                All This Week's Games
              </p>
              <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
                {otherPicks.map((p) => <PickCard key={p.game_id} pick={p} />)}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
