'use client'

import { useEffect, useState } from 'react'
import { api, Season, HistoryGame } from '@/lib/api'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'
import { Badge } from '@/components/ui/badge'

function pct(v: number | null | undefined) {
  if (v == null) return '—'
  return `${(v * 100).toFixed(1)}%`
}

function formatSpread(spread: number) {
  if (spread === 0) return 'Pick'
  const favored = spread > 0 ? 'Home' : 'Away'
  return `${favored} -${Math.abs(spread)}`
}

export default function HistoryPage() {
  const [seasons, setSeasons] = useState<Season[]>([])
  const [season, setSeason] = useState<number | null>(null)
  const [week, setWeek] = useState<number | null>(null)
  const [games, setGames] = useState<HistoryGame[] | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    api.seasons().then(({ seasons }) => {
      setSeasons(seasons)
      if (seasons.length > 0) {
        const latest = seasons[seasons.length - 1]
        setSeason(latest.season)
        setWeek(latest.max_week)
      }
    })
  }, [])

  useEffect(() => {
    if (!season || !week) return
    setLoading(true)
    setError(null)
    api
      .history(season, week)
      .then(({ games }) => setGames(games))
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false))
  }, [season, week])

  const currentSeason = seasons.find((s) => s.season === season)
  const weeks = currentSeason
    ? Array.from(
        { length: currentSeason.max_week - currentSeason.min_week + 1 },
        (_, i) => currentSeason.min_week + i
      )
    : []

  const correct = games?.filter((g) => g.model_correct === 1).length ?? 0
  const total = games?.length ?? 0

  return (
    <div className="max-w-6xl mx-auto px-4 py-8">
      <div className="flex items-center justify-between mb-6 gap-4">
        <h1 className="text-2xl font-semibold">History</h1>
        <div className="flex items-center gap-3">
          <Select
            value={season?.toString() ?? ''}
            onValueChange={(v) => {
              if (!v) return
              const s = +v
              setSeason(s)
              const meta = seasons.find((x) => x.season === s)
              setWeek(meta?.max_week ?? null)
            }}
          >
            <SelectTrigger className="w-28">
              <SelectValue placeholder="Season" />
            </SelectTrigger>
            <SelectContent>
              {seasons.map((s) => (
                <SelectItem key={s.season} value={s.season.toString()}>
                  {s.season}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Select
            value={week?.toString() ?? ''}
            onValueChange={(v) => { if (v) setWeek(+v) }}
            disabled={!season || weeks.length === 0}
          >
            <SelectTrigger className="w-28">
              <SelectValue placeholder="Week" />
            </SelectTrigger>
            <SelectContent>
              {weeks.map((w) => (
                <SelectItem key={w} value={w.toString()}>
                  Week {w}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>

      {loading && (
        <p className="text-sm text-muted-foreground">Loading results…</p>
      )}
      {error && (
        <p className="text-sm text-destructive">Error: {error}</p>
      )}

      {games && !loading && (
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-3">
              <span>
                Season {season} · Week {week}
              </span>
              {total > 0 && (
                <Badge variant="outline">
                  {correct}/{total} correct ({((correct / total) * 100).toFixed(0)}%)
                </Badge>
              )}
            </CardTitle>
          </CardHeader>
          <CardContent className="p-0 pb-2">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Matchup</TableHead>
                  <TableHead className="text-right">Score</TableHead>
                  <TableHead className="text-right">Spread</TableHead>
                  <TableHead className="text-right">Market WP</TableHead>
                  <TableHead className="text-right">Model WP</TableHead>
                  <TableHead className="text-right">Edge</TableHead>
                  <TableHead className="text-center">Result</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {games.map((g) => {
                  const homeWon = g.home_won === 1
                  const winner = homeWon ? g.home_team : g.away_team
                  const modelPredictedHome = g.model_wp > 0.5
                  const correct = g.model_correct === 1

                  return (
                    <TableRow key={g.game_id}>
                      <TableCell className="font-medium">
                        <span className={!homeWon ? 'font-semibold' : 'text-muted-foreground'}>
                          {g.away_team}
                        </span>
                        {' @ '}
                        <span className={homeWon ? 'font-semibold' : 'text-muted-foreground'}>
                          {g.home_team}
                        </span>
                      </TableCell>
                      <TableCell className="text-right tabular-nums text-sm">
                        {g.away_score}–{g.home_score}
                      </TableCell>
                      <TableCell className="text-right text-muted-foreground text-sm">
                        {formatSpread(g.spread_line)}
                      </TableCell>
                      <TableCell className="text-right tabular-nums">
                        {pct(g.market_wp)}
                      </TableCell>
                      <TableCell className="text-right tabular-nums">
                        <span className={modelPredictedHome === homeWon ? '' : 'text-muted-foreground'}>
                          {pct(g.model_wp)}
                        </span>
                      </TableCell>
                      <TableCell className="text-right tabular-nums">
                        <span className={g.edge > 0 ? 'text-green-600' : 'text-red-500'}>
                          {g.edge > 0 ? '+' : ''}
                          {pct(g.edge)}
                        </span>
                      </TableCell>
                      <TableCell className="text-center">
                        <Badge
                          variant={correct ? 'default' : 'secondary'}
                          className="text-xs"
                        >
                          {correct ? `✓ ${winner}` : `✗ ${winner}`}
                        </Badge>
                      </TableCell>
                    </TableRow>
                  )
                })}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
