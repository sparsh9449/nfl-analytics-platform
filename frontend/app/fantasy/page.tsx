'use client'

import { useEffect, useState } from 'react'
import { cn } from '@/lib/utils'

type Position = 'QB' | 'RB' | 'WR' | 'TE'

interface FantasyTarget {
  player_id: string
  name: string
  position: Position
  team: string
  pts_ppr: number
  pass_yd?: number
  pass_td?: number
  rush_att?: number
  rush_yd?: number
  rush_td?: number
  rec?: number
  rec_yd?: number
  rec_td?: number
  injury_status?: string | null
}

const POSITIONS: Position[] = ['QB', 'RB', 'WR', 'TE']

const POS_COLOR: Record<Position, string> = {
  QB: '#ff9a3c',
  RB: '#00e676',
  WR: '#3b9eff',
  TE: '#b06dff',
}

function statLine(p: FantasyTarget): string {
  if (p.position === 'QB') {
    const parts = []
    if (p.pass_yd) parts.push(`${Math.round(p.pass_yd)} pass yds`)
    if (p.pass_td) parts.push(`${p.pass_td.toFixed(1)} TD`)
    if (p.rush_yd) parts.push(`${Math.round(p.rush_yd)} rush yds`)
    return parts.join(' · ')
  }
  if (p.position === 'RB') {
    const parts = []
    if (p.rush_yd) parts.push(`${Math.round(p.rush_yd)} rush yds`)
    if (p.rush_td) parts.push(`${p.rush_td.toFixed(1)} TD`)
    if (p.rec) parts.push(`${p.rec.toFixed(1)} rec`)
    if (p.rec_yd) parts.push(`${Math.round(p.rec_yd)} rec yds`)
    return parts.join(' · ')
  }
  const parts = []
  if (p.rec) parts.push(`${p.rec.toFixed(1)} rec`)
  if (p.rec_yd) parts.push(`${Math.round(p.rec_yd)} yds`)
  if (p.rec_td) parts.push(`${p.rec_td.toFixed(1)} TD`)
  return parts.join(' · ')
}

function InjuryBadge({ status }: { status?: string | null }) {
  if (!status) return null
  const color =
    status === 'Out' ? 'text-[#ff4444] bg-[#ff4444]/10' :
    status === 'Doubtful' ? 'text-orange-400 bg-orange-400/10' :
    'text-yellow-400 bg-yellow-400/10'
  return (
    <span className={cn('text-[10px] font-bold uppercase px-1.5 py-0.5 rounded', color)}>
      {status}
    </span>
  )
}

function Skeleton({ className }: { className?: string }) {
  return <div className={cn('animate-pulse rounded-xl bg-white/5', className)} />
}

export default function FantasyPage() {
  const [byPosition, setByPosition] = useState<Record<Position, FantasyTarget[]> | null>(null)
  const [activePos, setActivePos] = useState<Position>('RB')
  const [season, setSeason] = useState<string>('')
  const [week, setWeek] = useState<string>('')
  const [loading, setLoading] = useState(true)
  const [noData, setNoData] = useState(false)

  useEffect(() => {
    fetch('/api/espn/scoreboard')
      .then((r) => r.json())
      .then((d) => {
        const s = d.season ?? new Date().getFullYear()
        const w = d.week ?? 1
        setSeason(String(s))
        setWeek(String(w))
        return fetch(`/api/fantasy/targets?season=${s}&week=${w}`)
      })
      .then((r) => r.json())
      .then((d) => {
        const bp = d.byPosition as Record<Position, FantasyTarget[]>
        const hasData = Object.values(bp).some((arr) => arr.length > 0)
        if (!hasData) {
          setNoData(true)
        } else {
          setByPosition(bp)
        }
      })
      .finally(() => setLoading(false))
  }, [])

  const players = byPosition?.[activePos] ?? []

  return (
    <div className="max-w-7xl mx-auto px-4 py-8">
      <div className="mb-6">
        <p className="text-xs font-semibold uppercase tracking-widest text-muted-foreground mb-1">
          {season && week ? `Season ${season} · Week ${week}` : 'Current Week'}
        </p>
        <h1 className="text-3xl font-black">Fantasy Targets</h1>
      </div>

      <div className="flex gap-2 mb-6">
        {POSITIONS.map((pos) => (
          <button
            key={pos}
            onClick={() => setActivePos(pos)}
            style={activePos === pos ? { backgroundColor: POS_COLOR[pos] + '20', color: POS_COLOR[pos], borderColor: POS_COLOR[pos] + '40' } : {}}
            className={cn(
              'px-4 py-1.5 rounded-lg text-sm font-bold border transition-all',
              activePos === pos
                ? 'border-transparent'
                : 'border-border text-muted-foreground hover:text-foreground hover:bg-white/5'
            )}
          >
            {pos}
          </button>
        ))}
      </div>

      {loading ? (
        <div className="space-y-3">
          {Array.from({ length: 8 }).map((_, i) => <Skeleton key={i} className="h-16" />)}
        </div>
      ) : noData ? (
        <div className="text-center py-20 space-y-3">
          <p className="text-lg font-semibold text-muted-foreground">Projections unavailable</p>
          <p className="text-sm text-muted-foreground">
            Fantasy projections for the current week haven&apos;t been released yet.
          </p>
        </div>
      ) : players.length === 0 ? (
        <div className="text-center py-20 text-muted-foreground">No data for this position.</div>
      ) : (
        <div className="space-y-2">
          <div className="grid grid-cols-[2rem_1fr_auto] gap-4 px-4 pb-2">
            <span />
            <span className="text-xs font-semibold uppercase tracking-widest text-muted-foreground">Player</span>
            <span className="text-xs font-semibold uppercase tracking-widest text-muted-foreground">Proj</span>
          </div>
          {players.map((p, i) => (
            <div
              key={p.player_id}
              className="bg-card border border-border rounded-xl px-4 py-3 grid grid-cols-[2rem_1fr_auto] gap-4 items-center hover:border-white/20 transition-colors"
            >
              <span className="text-muted-foreground font-bold tabular-nums text-sm">{i + 1}</span>
              <div className="min-w-0">
                <div className="flex items-center gap-2 mb-0.5">
                  <span className="font-bold text-sm">{p.name}</span>
                  <span
                    className="text-[10px] font-bold uppercase px-1.5 py-0.5 rounded"
                    style={{ color: POS_COLOR[p.position], backgroundColor: POS_COLOR[p.position] + '20' }}
                  >
                    {p.position}
                  </span>
                  <InjuryBadge status={p.injury_status} />
                </div>
                <p className="text-xs text-muted-foreground truncate">
                  {p.team}
                  {statLine(p) ? <span className="ml-2">{statLine(p)}</span> : null}
                </p>
              </div>
              <div className="text-right">
                <span
                  className="text-lg font-black tabular-nums"
                  style={{ color: POS_COLOR[p.position] }}
                >
                  {p.pts_ppr}
                </span>
                <p className="text-[10px] text-muted-foreground">PPR</p>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
