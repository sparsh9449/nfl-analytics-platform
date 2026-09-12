'use client'

import { useEffect, useState } from 'react'
import Link from 'next/link'
import { cn } from '@/lib/utils'
import { api } from '@/lib/api'

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

interface NewsArticle {
  headline: string
  description: string
  published: string
  url: string
  image: string | null
}

interface FantasyTarget {
  player_id: string
  name: string
  position: string
  team: string
  pts_ppr: number
}

interface FeaturedPick {
  home: string
  away: string
  modelWp: number
  marketWp: number
  edge: number
  betTeam: string
}

function Skeleton({ className }: { className?: string }) {
  return <div className={cn('animate-pulse rounded-lg bg-white/5', className)} />
}

function GameMiniCard({ game }: { game: Game }) {
  const home = game.homeTeam
  const away = game.awayTeam
  const isLive = game.status === 'in'
  const isFinal = game.status === 'post'

  return (
    <div className="shrink-0 w-52 bg-card border border-border rounded-xl p-4 flex flex-col gap-3">
      <div className="flex items-center justify-between">
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
          {isLive ? '● LIVE' : game.statusText}
        </span>
        {game.tv && <span className="text-[10px] text-muted-foreground">{game.tv}</span>}
      </div>

      <div className="space-y-2">
        {[away, home].map((team, i) =>
          team ? (
            <div key={i} className="flex items-center justify-between gap-2">
              <div className="flex items-center gap-2 min-w-0">
                {team.logo && (
                  <img src={team.logo} alt={team.abbr} className="w-5 h-5 object-contain shrink-0" />
                )}
                <span
                  className={cn(
                    'text-sm font-semibold truncate',
                    team.winner ? 'text-foreground' : isFinal ? 'text-muted-foreground' : 'text-foreground'
                  )}
                >
                  {team.abbr}
                </span>
              </div>
              {team.score !== null && (
                <span
                  className={cn(
                    'text-sm font-bold tabular-nums',
                    team.winner ? 'text-foreground' : 'text-muted-foreground'
                  )}
                >
                  {team.score}
                </span>
              )}
            </div>
          ) : null
        )}
      </div>

      {game.spread && (
        <div className="text-[10px] text-muted-foreground border-t border-border pt-2">{game.spread}</div>
      )}
    </div>
  )
}

export default function HomePage() {
  const [games, setGames] = useState<Game[]>([])
  const [currentWeek, setCurrentWeek] = useState<number>(1)
  const [news, setNews] = useState<NewsArticle[]>([])
  const [fantasyTargets, setFantasyTargets] = useState<Record<string, FantasyTarget | null>>({})
  const [featuredPick, setFeaturedPick] = useState<FeaturedPick | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    Promise.all([
      fetch('/api/espn/scoreboard').then((r) => r.json()),
      fetch('/api/espn/news').then((r) => r.json()),
    ]).then(([scoreData, newsData]) => {
      setGames(scoreData.events ?? [])
      setCurrentWeek(scoreData.week ?? 1)
      setNews(newsData.articles ?? [])

      const season = scoreData.season ?? new Date().getFullYear()
      const week = scoreData.week ?? 1
      fetch(`/api/fantasy/targets?season=${season}&week=${week}`)
        .then((r) => r.json())
        .then((d) => setFantasyTargets(d.topPerPosition ?? {}))

      api
        .seasons()
        .then(({ seasons }) => {
          if (!seasons.length) return
          const latest = seasons[seasons.length - 1]
          return api.games(latest.season, latest.max_week)
        })
        .then((data) => {
          if (!data?.games?.length) return
          const top = data.games[0]
          setFeaturedPick({
            home: top.home_team,
            away: top.away_team,
            modelWp: top.model_wp,
            marketWp: top.market_wp,
            edge: top.edge,
            betTeam: top.bet_team,
          })
        })
        .catch(() => {})
    }).finally(() => setLoading(false))
  }, [])

  return (
    <div className="max-w-7xl mx-auto px-4 py-8 space-y-10">
      <div className="flex items-end justify-between">
        <div>
          <p className="text-xs font-semibold uppercase tracking-widest text-muted-foreground mb-1">
            NFL 2026
          </p>
          <h1 className="text-3xl font-black text-foreground">
            Week {currentWeek}
          </h1>
        </div>
        <Link href="/games" className="text-sm text-primary hover:underline">
          Full schedule →
        </Link>
      </div>

      <div className="overflow-x-auto -mx-4 px-4">
        <div className="flex gap-3 pb-2">
          {loading
            ? Array.from({ length: 5 }).map((_, i) => <Skeleton key={i} className="shrink-0 w-52 h-36" />)
            : games.length === 0
            ? <p className="text-sm text-muted-foreground">No games found.</p>
            : games.map((g) => <GameMiniCard key={g.id} game={g} />)}
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          <p className="text-xs font-semibold uppercase tracking-widest text-muted-foreground mb-3">
            Featured Pick
          </p>
          {loading ? (
            <Skeleton className="h-40" />
          ) : featuredPick ? (
            <div className="bg-card border border-border rounded-xl p-5 space-y-4">
              <p className="text-xs text-muted-foreground">
                {featuredPick.away} @ {featuredPick.home}
              </p>
              <div className="space-y-2">
                <div className="flex justify-between text-sm mb-1">
                  <span className="text-muted-foreground">Market WP</span>
                  <span className="tabular-nums">{(featuredPick.marketWp * 100).toFixed(1)}%</span>
                </div>
                <div className="h-1.5 bg-white/8 rounded-full">
                  <div
                    className="h-full bg-white/30 rounded-full"
                    style={{ width: `${featuredPick.marketWp * 100}%` }}
                  />
                </div>
                <div className="flex justify-between text-sm mb-1 mt-3">
                  <span className="text-muted-foreground">Model WP</span>
                  <span className="tabular-nums">{(featuredPick.modelWp * 100).toFixed(1)}%</span>
                </div>
                <div className="h-1.5 bg-white/8 rounded-full">
                  <div
                    className="h-full bg-primary rounded-full"
                    style={{ width: `${featuredPick.modelWp * 100}%` }}
                  />
                </div>
              </div>
              <div className="flex items-center justify-between pt-2 border-t border-border">
                <div>
                  <p className="text-xs text-muted-foreground uppercase tracking-widest">Edge</p>
                  <p
                    className={cn(
                      'text-2xl font-black tabular-nums',
                      featuredPick.edge > 0 ? 'text-primary' : 'text-[#ff4444]'
                    )}
                  >
                    {featuredPick.edge > 0 ? '+' : ''}
                    {(featuredPick.edge * 100).toFixed(1)}%
                  </p>
                </div>
                <div className="text-right">
                  <p className="text-xs text-muted-foreground uppercase tracking-widest">Bet on</p>
                  <p className="text-lg font-bold">{featuredPick.betTeam}</p>
                </div>
              </div>
            </div>
          ) : (
            <div className="bg-card border border-border rounded-xl p-5 flex items-center justify-center h-40">
              <p className="text-sm text-muted-foreground">No picks available</p>
            </div>
          )}
          <Link href="/picks" className="mt-2 block text-xs text-primary hover:underline">
            See all picks →
          </Link>
        </div>

        <div>
          <p className="text-xs font-semibold uppercase tracking-widest text-muted-foreground mb-3">
            Top Fantasy Targets
          </p>
          {loading ? (
            <Skeleton className="h-40" />
          ) : (
            <div className="bg-card border border-border rounded-xl divide-y divide-border">
              {(['QB', 'RB', 'WR', 'TE'] as const).map((pos) => {
                const p = fantasyTargets[pos]
                return (
                  <div key={pos} className="flex items-center justify-between px-5 py-3">
                    <div className="flex items-center gap-3">
                      <span className="text-[10px] font-bold uppercase tracking-widest text-muted-foreground w-6">
                        {pos}
                      </span>
                      {p ? (
                        <div>
                          <p className="text-sm font-semibold">{p.name}</p>
                          <p className="text-xs text-muted-foreground">{p.team}</p>
                        </div>
                      ) : (
                        <p className="text-sm text-muted-foreground">—</p>
                      )}
                    </div>
                    {p && (
                      <span className="text-sm font-bold text-primary tabular-nums">
                        {p.pts_ppr} pts
                      </span>
                    )}
                  </div>
                )
              })}
            </div>
          )}
          <Link href="/fantasy" className="mt-2 block text-xs text-primary hover:underline">
            Full fantasy targets →
          </Link>
        </div>
      </div>

      <div>
        <p className="text-xs font-semibold uppercase tracking-widest text-muted-foreground mb-4">
          NFL News
        </p>
        {loading ? (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {Array.from({ length: 6 }).map((_, i) => <Skeleton key={i} className="h-28" />)}
          </div>
        ) : (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {news.slice(0, 6).map((article, i) => (
              <a
                key={i}
                href={article.url}
                target="_blank"
                rel="noopener noreferrer"
                className="bg-card border border-border rounded-xl overflow-hidden flex gap-3 hover:border-white/20 transition-colors group"
              >
                {article.image && (
                  <img
                    src={article.image}
                    alt=""
                    className="w-24 h-full object-cover shrink-0"
                  />
                )}
                <div className="p-3 min-w-0">
                  <p className="text-sm font-semibold leading-snug line-clamp-2 group-hover:text-primary transition-colors">
                    {article.headline}
                  </p>
                  {article.description && (
                    <p className="text-xs text-muted-foreground mt-1 line-clamp-2">
                      {article.description}
                    </p>
                  )}
                </div>
              </a>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
