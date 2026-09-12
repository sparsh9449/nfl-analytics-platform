'use client'

import { useEffect, useState } from 'react'
import { cn } from '@/lib/utils'

interface Game {
  id: string
  name: string
  date: string
  status: 'pre' | 'in' | 'post'
  statusText: string
  homeTeam: Team | null
  awayTeam: Team | null
  spread: string | null
  overUnder: number | null
  tv: string | null
}

interface Team {
  abbr: string
  name: string
  shortName: string
  color: string
  logo: string
  score: string | null
  winner: boolean
}

function Skeleton({ className }: { className?: string }) {
  return <div className={cn('animate-pulse rounded-xl bg-white/5', className)} />
}

function GameCard({ game }: { game: Game }) {
  const home = game.homeTeam
  const away = game.awayTeam
  const isLive = game.status === 'in'
  const isFinal = game.status === 'post'
  const isPre = game.status === 'pre'

  const kickoffTime = isPre
    ? new Date(game.date).toLocaleTimeString('en-US', {
        hour: 'numeric',
        minute: '2-digit',
        timeZoneName: 'short',
      })
    : null

  return (
    <div className="bg-card border border-border rounded-xl overflow-hidden hover:border-white/20 transition-colors">
      {(home?.color || away?.color) && (
        <div
          className="h-1"
          style={{
            background: `linear-gradient(to right, #${away?.color ?? '1e2d3f'}, #${home?.color ?? '1e2d3f'})`,
          }}
        />
      )}
      <div className="p-5">
        <div className="flex items-center justify-between mb-4">
          <span
            className={cn(
              'text-[10px] font-bold uppercase tracking-widest px-2 py-0.5 rounded-full',
              isLive
                ? 'bg-[#00e676]/15 text-[#00e676]'
                : isFinal
                ? 'bg-white/8 text-muted-foreground'
                : 'bg-white/8 text-muted-foreground'
            )}
          >
            {isLive ? '● ' : ''}
            {game.statusText}
          </span>
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            {game.tv && <span>{game.tv}</span>}
          </div>
        </div>

        <div className="space-y-3">
          {[away, home].map((team, i) =>
            team ? (
              <div key={i} className="flex items-center justify-between">
                <div className="flex items-center gap-3 min-w-0">
                  {team.logo ? (
                    <img src={team.logo} alt={team.abbr} className="w-8 h-8 object-contain shrink-0" />
                  ) : (
                    <div className="w-8 h-8 rounded-full bg-white/10 shrink-0" />
                  )}
                  <div className="min-w-0">
                    <p
                      className={cn(
                        'font-bold text-base leading-tight',
                        isFinal && !team.winner ? 'text-muted-foreground' : 'text-foreground'
                      )}
                    >
                      {team.shortName}
                    </p>
                    <p className="text-xs text-muted-foreground">{i === 0 ? 'Away' : 'Home'}</p>
                  </div>
                </div>
                {team.score !== null ? (
                  <span
                    className={cn(
                      'text-2xl font-black tabular-nums',
                      isFinal && !team.winner ? 'text-muted-foreground' : 'text-foreground'
                    )}
                  >
                    {team.score}
                  </span>
                ) : isPre && kickoffTime ? (
                  i === 0 ? (
                    <span className="text-sm text-muted-foreground">{kickoffTime}</span>
                  ) : null
                ) : null}
              </div>
            ) : null
          )}
        </div>

        {(game.spread || game.overUnder) && (
          <div className="mt-4 pt-3 border-t border-border flex gap-4 text-xs text-muted-foreground">
            {game.spread && <span>Spread: {game.spread}</span>}
            {game.overUnder && <span>O/U: {game.overUnder}</span>}
          </div>
        )}
      </div>
    </div>
  )
}

export default function GamesPage() {
  const [games, setGames] = useState<Game[]>([])
  const [currentWeek, setCurrentWeek] = useState<number>(1)
  const [selectedWeek, setSelectedWeek] = useState<number | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetch('/api/espn/scoreboard')
      .then((r) => r.json())
      .then((d) => {
        setGames(d.events ?? [])
        setCurrentWeek(d.week ?? 1)
        setSelectedWeek(d.week ?? 1)
      })
      .finally(() => setLoading(false))
  }, [])

  const fetchWeek = (week: number) => {
    setLoading(true)
    setSelectedWeek(week)
    fetch(`/api/espn/scoreboard?week=${week}`)
      .then((r) => r.json())
      .then((d) => setGames(d.events ?? []))
      .finally(() => setLoading(false))
  }

  const weeks = Array.from({ length: 18 }, (_, i) => i + 1)

  return (
    <div className="max-w-7xl mx-auto px-4 py-8">
      <div className="flex items-end justify-between mb-6">
        <div>
          <p className="text-xs font-semibold uppercase tracking-widest text-muted-foreground mb-1">
            NFL Schedule
          </p>
          <h1 className="text-3xl font-black">Games</h1>
        </div>
      </div>

      <div className="flex gap-2 overflow-x-auto pb-3 mb-6">
        {weeks.map((w) => (
          <button
            key={w}
            onClick={() => fetchWeek(w)}
            className={cn(
              'shrink-0 px-3 py-1.5 rounded-lg text-sm font-medium transition-colors',
              selectedWeek === w
                ? 'bg-primary text-black'
                : 'bg-white/5 text-muted-foreground hover:bg-white/10 hover:text-foreground'
            )}
          >
            Wk {w}
            {w === currentWeek && (
              <span className="ml-1 text-[10px] text-primary/70">●</span>
            )}
          </button>
        ))}
      </div>

      {loading ? (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {Array.from({ length: 8 }).map((_, i) => <Skeleton key={i} className="h-52" />)}
        </div>
      ) : games.length === 0 ? (
        <div className="text-center py-20 text-muted-foreground">No games found for this week.</div>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {games.map((g) => <GameCard key={g.id} game={g} />)}
        </div>
      )}
    </div>
  )
}
