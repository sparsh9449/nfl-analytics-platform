'use client'

import { useEffect, useState } from 'react'
import { cn } from '@/lib/utils'

type Tab = 'standings' | 'passing' | 'rushing' | 'receiving'

interface Standing {
  abbr: string
  name: string
  shortName: string
  logo: string
  color: string
  conference: string
  division: string
  wins: number
  losses: number
  ties: number
  pct: string
  pf: number
  pa: number
  streak: string
  divisionRecord: string
}

interface LeaderEntry {
  name: string
  team: string
  headshot: string | null
  value: string
}

interface LeaderCategory {
  key: string
  label: string
  leaders: LeaderEntry[]
}

function Skeleton({ className }: { className?: string }) {
  return <div className={cn('animate-pulse rounded-xl bg-white/5', className)} />
}

const TABS: { id: Tab; label: string }[] = [
  { id: 'standings', label: 'Standings' },
  { id: 'passing', label: 'Passing' },
  { id: 'rushing', label: 'Rushing' },
  { id: 'receiving', label: 'Receiving' },
]

const LEADER_MAP: Record<Tab, string[]> = {
  standings: [],
  passing: ['passingYards', 'passingTouchdowns'],
  rushing: ['rushingYards', 'rushingTouchdowns'],
  receiving: ['receivingYards'],
}

export default function StatsPage() {
  const [tab, setTab] = useState<Tab>('standings')
  const [standings, setStandings] = useState<Standing[]>([])
  const [leaders, setLeaders] = useState<LeaderCategory[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    Promise.all([
      fetch('/api/espn/standings').then((r) => r.json()),
      fetch('/api/espn/leaders').then((r) => r.json()),
    ]).then(([s, l]) => {
      setStandings(s.standings ?? [])
      setLeaders(l.categories ?? [])
    }).finally(() => setLoading(false))
  }, [])

  const conferences = ['AFC', 'NFC']
  const activeKeys = LEADER_MAP[tab]
  const activeCategories = leaders.filter((c) => activeKeys.includes(c.key))

  return (
    <div className="max-w-7xl mx-auto px-4 py-8">
      <div className="mb-6">
        <p className="text-xs font-semibold uppercase tracking-widest text-muted-foreground mb-1">
          2026 NFL Season
        </p>
        <h1 className="text-3xl font-black">Stats</h1>
      </div>

      <div className="flex gap-1 mb-6 bg-white/4 p-1 rounded-xl w-fit">
        {TABS.map((t) => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            className={cn(
              'px-4 py-1.5 rounded-lg text-sm font-medium transition-colors',
              tab === t.id
                ? 'bg-card text-foreground shadow'
                : 'text-muted-foreground hover:text-foreground'
            )}
          >
            {t.label}
          </button>
        ))}
      </div>

      {loading ? (
        <div className="space-y-4">
          {Array.from({ length: 4 }).map((_, i) => <Skeleton key={i} className="h-40" />)}
        </div>
      ) : tab === 'standings' ? (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {conferences.map((conf) => {
            const teams = standings
              .filter((s) => s.conference === conf)
              .sort((a, b) => parseFloat(b.pct) - parseFloat(a.pct))
            return (
              <div key={conf} className="bg-card border border-border rounded-xl overflow-hidden">
                <div className="px-5 py-3.5 bg-white/4 border-b border-border">
                  <p className="text-sm font-black uppercase tracking-widest">{conf}</p>
                </div>
                <div className="grid grid-cols-[1fr_3rem_3rem_3rem_4rem] px-5 py-2 text-[10px] font-semibold uppercase tracking-widest text-muted-foreground border-b border-border">
                  <span>Team</span>
                  <span className="text-center">W</span>
                  <span className="text-center">L</span>
                  <span className="text-center">PCT</span>
                  <span className="text-right">Streak</span>
                </div>
                {teams.map((team, i) => (
                  <div
                    key={team.abbr}
                    className={cn(
                      'grid grid-cols-[1fr_3rem_3rem_3rem_4rem] px-5 py-3 items-center',
                      i < teams.length - 1 ? 'border-b border-border' : ''
                    )}
                  >
                    <div className="flex items-center gap-2.5 min-w-0">
                      {team.logo && (
                        <img src={team.logo} alt={team.abbr} className="w-6 h-6 object-contain shrink-0" />
                      )}
                      <div>
                        <p className="font-semibold text-sm leading-tight">{team.shortName}</p>
                        <p className="text-[10px] text-muted-foreground">{team.divisionRecord} DIV</p>
                      </div>
                    </div>
                    <span className="text-center text-sm font-bold tabular-nums">{team.wins}</span>
                    <span className="text-center text-sm tabular-nums text-muted-foreground">{team.losses}</span>
                    <span className="text-center text-sm tabular-nums text-muted-foreground">{team.pct}</span>
                    <span className={cn('text-right text-xs font-semibold tabular-nums', team.streak.startsWith('W') ? 'text-primary' : 'text-[#ff4444]')}>
                      {team.streak}
                    </span>
                  </div>
                ))}
              </div>
            )
          })}
        </div>
      ) : activeCategories.length === 0 ? (
        <div className="text-center py-20 text-muted-foreground">No stat data available.</div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {activeCategories.map((cat) => (
            <div key={cat.key} className="bg-card border border-border rounded-xl overflow-hidden">
              <div className="px-5 py-3.5 bg-white/4 border-b border-border">
                <p className="text-xs font-semibold uppercase tracking-widest text-muted-foreground">{cat.label}</p>
              </div>
              <div>
                {cat.leaders.map((leader, i) => (
                  <div
                    key={i}
                    className={cn(
                      'flex items-center justify-between px-5 py-3',
                      i < cat.leaders.length - 1 ? 'border-b border-border' : ''
                    )}
                  >
                    <div className="flex items-center gap-3">
                      <span className="text-muted-foreground font-bold tabular-nums text-sm w-5">{i + 1}</span>
                      {leader.headshot ? (
                        <img
                          src={leader.headshot}
                          alt={leader.name}
                          className="w-8 h-8 rounded-full object-cover bg-white/10"
                        />
                      ) : (
                        <div className="w-8 h-8 rounded-full bg-white/10 shrink-0" />
                      )}
                      <div>
                        <p className="font-semibold text-sm">{leader.name}</p>
                        <p className="text-xs text-muted-foreground">{leader.team}</p>
                      </div>
                    </div>
                    <span className="font-black text-lg tabular-nums">{leader.value}</span>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
