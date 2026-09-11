'use client'

import { useEffect, useState } from 'react'
import { api, EdgeSummaryResponse, WeekBreakdownResponse, RoiResponse } from '@/lib/api'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  ReferenceLine,
} from 'recharts'

function pct(v: number) {
  return `${(v * 100).toFixed(1)}%`
}

export default function EdgePage() {
  const [edge, setEdge] = useState<EdgeSummaryResponse | null>(null)
  const [weekBreak, setWeekBreak] = useState<WeekBreakdownResponse | null>(null)
  const [roi, setRoi] = useState<RoiResponse | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    Promise.all([api.edgeSummary(), api.weekBreakdown(), api.roi()])
      .then(([e, w, r]) => {
        setEdge(e)
        setWeekBreak(w)
        setRoi(r)
      })
      .catch((e: Error) => setError(e.message))
  }, [])

  if (error) {
    return (
      <div className="max-w-6xl mx-auto px-4 py-8">
        <p className="text-sm text-destructive">Error: {error}</p>
      </div>
    )
  }

  const bucketChartData = edge?.buckets.map((b) => ({
    bucket: b.bucket,
    'Model WP': parseFloat((b.model_wp_mean * 100).toFixed(1)),
    'Actual Win %': parseFloat((b.actual_win_rate * 100).toFixed(1)),
    count: b.count,
  }))

  const weekGroupData = weekBreak?.pos_edge_gt10_by_week_group.map((e) => ({
    group: e.week_group.trim(),
    'Market WP': parseFloat((e.market_wp * 100).toFixed(1)),
    'Model WP': parseFloat((e.model_wp * 100).toFixed(1)),
    'Actual Win %': parseFloat((e.actual_wr * 100).toFixed(1)),
    n: e.n,
  }))

  return (
    <div className="max-w-6xl mx-auto px-4 py-8 space-y-8">
      <h1 className="text-2xl font-semibold">Edge Analysis</h1>

      {!edge && <p className="text-sm text-muted-foreground">Loading…</p>}

      {edge && (
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-4">
          <Card>
            <CardContent className="pt-5 pb-4">
              <p className="text-xs text-muted-foreground uppercase tracking-wide">
                Games evaluated
              </p>
              <p className="text-3xl font-semibold mt-1">{edge.n_games}</p>
              <p className="text-xs text-muted-foreground mt-1">
                {edge.seasons[0]}–{edge.seasons[edge.seasons.length - 1]} seasons
              </p>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="pt-5 pb-4">
              <p className="text-xs text-muted-foreground uppercase tracking-wide">
                Mean edge
              </p>
              <p className="text-3xl font-semibold mt-1">
                {pct(edge.overall.mean_edge)}
              </p>
              <p className="text-xs text-muted-foreground mt-1">
                σ = {pct(edge.overall.std_edge)}
              </p>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="pt-5 pb-4">
              <p className="text-xs text-muted-foreground uppercase tracking-wide">
                Breakeven win rate
              </p>
              <p className="text-3xl font-semibold mt-1">
                {roi ? pct(roi.breakeven_wr) : '—'}
              </p>
              <p className="text-xs text-muted-foreground mt-1">
                At {roi?.odds} odds
              </p>
            </CardContent>
          </Card>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {bucketChartData && (
          <Card>
            <CardHeader>
              <CardTitle className="text-base font-medium">
                Edge Bucket — Model WP vs Actual Win Rate
              </CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart
                  data={bucketChartData}
                  margin={{ top: 4, right: 8, bottom: 40, left: 0 }}
                >
                  <CartesianGrid
                    strokeDasharray="3 3"
                    stroke="hsl(var(--border))"
                    vertical={false}
                  />
                  <XAxis
                    dataKey="bucket"
                    tick={{ fontSize: 10 }}
                    tickLine={false}
                    angle={-30}
                    textAnchor="end"
                    height={50}
                  />
                  <YAxis
                    tickFormatter={(v) => `${v}%`}
                    tick={{ fontSize: 11 }}
                    tickLine={false}
                    axisLine={false}
                    width={40}
                    domain={[0, 100]}
                  />
                  <Tooltip
                    formatter={(v: unknown, name: unknown) => [typeof v === 'number' ? `${v}%` : '', String(name)]}
                    contentStyle={{
                      fontSize: 12,
                      borderColor: 'hsl(var(--border))',
                    }}
                  />
                  <Legend wrapperStyle={{ fontSize: 12 }} />
                  <Bar dataKey="Model WP" fill="hsl(var(--primary))" radius={[3, 3, 0, 0]} />
                  <Bar dataKey="Actual Win %" fill="hsl(var(--muted-foreground))" radius={[3, 3, 0, 0]} opacity={0.7} />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        )}

        {weekGroupData && (
          <Card>
            <CardHeader>
              <CardTitle className="text-base font-medium">
                {'> +10% Edge — Actual Win Rate by Week Group'}
              </CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart
                  data={weekGroupData}
                  margin={{ top: 4, right: 8, bottom: 8, left: 0 }}
                >
                  <CartesianGrid
                    strokeDasharray="3 3"
                    stroke="hsl(var(--border))"
                    vertical={false}
                  />
                  <XAxis
                    dataKey="group"
                    tick={{ fontSize: 11 }}
                    tickLine={false}
                  />
                  <YAxis
                    tickFormatter={(v) => `${v}%`}
                    tick={{ fontSize: 11 }}
                    tickLine={false}
                    axisLine={false}
                    width={40}
                    domain={[0, 60]}
                  />
                  <Tooltip
                    formatter={(v: unknown, name: unknown) => [typeof v === 'number' ? `${v}%` : '', String(name)]}
                    contentStyle={{
                      fontSize: 12,
                      borderColor: 'hsl(var(--border))',
                    }}
                  />
                  <Legend wrapperStyle={{ fontSize: 12 }} />
                  <ReferenceLine
                    y={52.38}
                    stroke="hsl(var(--destructive))"
                    strokeDasharray="4 4"
                    label={{ value: 'Breakeven', fontSize: 11, fill: 'hsl(var(--destructive))' }}
                  />
                  <Bar dataKey="Market WP" fill="hsl(var(--muted-foreground))" radius={[3, 3, 0, 0]} opacity={0.5} />
                  <Bar dataKey="Actual Win %" fill="hsl(var(--primary))" radius={[3, 3, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        )}
      </div>

      {roi && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base font-medium">
              Flat-Bet ROI by Week Group
              <span className="text-muted-foreground font-normal ml-2 text-sm">
                threshold ≥{pct(roi.edge_thresh)} · {roi.odds} odds
              </span>
            </CardTitle>
          </CardHeader>
          <CardContent className="p-0 pb-2">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Week Group</TableHead>
                  <TableHead className="text-right">Bets</TableHead>
                  <TableHead className="text-right">Win Rate</TableHead>
                  <TableHead className="text-right">ROI</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {roi.overall.map((r) => (
                  <TableRow key={r.week_group}>
                    <TableCell>{r.week_group}</TableCell>
                    <TableCell className="text-right tabular-nums">{r.n}</TableCell>
                    <TableCell className="text-right tabular-nums">
                      {pct(r.win_rate)}
                    </TableCell>
                    <TableCell className="text-right tabular-nums">
                      <span className={r.roi >= 0 ? 'text-green-600' : 'text-red-500'}>
                        {r.roi >= 0 ? '+' : ''}
                        {pct(r.roi)}
                      </span>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
