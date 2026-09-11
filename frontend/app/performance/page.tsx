'use client'

import { useEffect, useState } from 'react'
import { api, MetricsResponse, SeasonAccuracyEntry } from '@/lib/api'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts'

function StatCard({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <Card>
      <CardContent className="pt-5 pb-4">
        <p className="text-xs text-muted-foreground uppercase tracking-wide">{label}</p>
        <p className="text-3xl font-semibold mt-1 tabular-nums">{value}</p>
        {sub && <p className="text-xs text-muted-foreground mt-1">{sub}</p>}
      </CardContent>
    </Card>
  )
}

export default function PerformancePage() {
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null)
  const [accuracy, setAccuracy] = useState<SeasonAccuracyEntry[] | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    Promise.all([api.metrics(), api.seasonAccuracy()])
      .then(([m, a]) => {
        setMetrics(m)
        setAccuracy(a.seasons)
      })
      .catch((e: Error) => setError(e.message))
  }, [])

  const xgb = metrics?.xgboost
  const cal = xgb?.test_2024_2025_calibrated

  const calibChartData = cal?.calibration.map((c) => ({
    name: `${(c.predicted * 100).toFixed(0)}%`,
    Model: parseFloat((c.actual * 100).toFixed(1)),
    Perfect: parseFloat((c.predicted * 100).toFixed(1)),
  }))

  if (error) {
    return (
      <div className="max-w-6xl mx-auto px-4 py-8">
        <p className="text-sm text-destructive">Error: {error}</p>
      </div>
    )
  }

  return (
    <div className="max-w-6xl mx-auto px-4 py-8 space-y-8">
      <h1 className="text-2xl font-semibold">Model Performance</h1>

      {cal && xgb && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
          <StatCard
            label="Test AUC"
            value={cal.roc_auc.toFixed(4)}
            sub="2024–2025 seasons"
          />
          <StatCard
            label="Test Brier"
            value={cal.brier.toFixed(4)}
            sub="Lower is better"
          />
          <StatCard
            label="Val AUC"
            value={xgb.val_2023.roc_auc.toFixed(4)}
            sub="2023 season"
          />
          <StatCard
            label="Val Brier"
            value={xgb.val_2023.brier.toFixed(4)}
            sub="2023 season"
          />
        </div>
      )}

      {!metrics && (
        <p className="text-sm text-muted-foreground">Loading metrics…</p>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {accuracy && (
          <Card>
            <CardHeader>
              <CardTitle className="text-base font-medium">
                Season-by-Season Accuracy
              </CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={280}>
                <LineChart
                  data={accuracy}
                  margin={{ top: 4, right: 16, bottom: 4, left: 0 }}
                >
                  <CartesianGrid
                    strokeDasharray="3 3"
                    stroke="hsl(var(--border))"
                  />
                  <XAxis
                    dataKey="season"
                    tick={{ fontSize: 12 }}
                    tickLine={false}
                  />
                  <YAxis
                    domain={[0.5, 0.75]}
                    tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
                    tick={{ fontSize: 12 }}
                    tickLine={false}
                    axisLine={false}
                    width={44}
                  />
                  <Tooltip
                    formatter={(v: unknown) => typeof v === 'number' ? `${(v * 100).toFixed(1)}%` : ''}
                    labelFormatter={(l) => `Season ${l}`}
                    contentStyle={{
                      fontSize: 12,
                      borderColor: 'hsl(var(--border))',
                    }}
                  />
                  <Line
                    type="monotone"
                    dataKey="accuracy"
                    dot={{ r: 3 }}
                    strokeWidth={2}
                    name="Accuracy"
                    activeDot={{ r: 5 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        )}

        {calibChartData && (
          <Card>
            <CardHeader>
              <CardTitle className="text-base font-medium">
                Calibration — Reliability Diagram
              </CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={280}>
                <LineChart
                  data={calibChartData}
                  margin={{ top: 4, right: 16, bottom: 4, left: 0 }}
                >
                  <CartesianGrid
                    strokeDasharray="3 3"
                    stroke="hsl(var(--border))"
                  />
                  <XAxis
                    dataKey="name"
                    tick={{ fontSize: 11 }}
                    tickLine={false}
                    label={{
                      value: 'Predicted WP',
                      position: 'insideBottom',
                      offset: -2,
                      fontSize: 11,
                      fill: 'hsl(var(--muted-foreground))',
                    }}
                    height={36}
                  />
                  <YAxis
                    tickFormatter={(v) => `${v}%`}
                    tick={{ fontSize: 11 }}
                    tickLine={false}
                    axisLine={false}
                    width={44}
                    label={{
                      value: 'Actual Win %',
                      angle: -90,
                      position: 'insideLeft',
                      offset: 8,
                      fontSize: 11,
                      fill: 'hsl(var(--muted-foreground))',
                    }}
                  />
                  <Tooltip
                    formatter={(v: unknown) => typeof v === 'number' ? `${v}%` : ''}
                    contentStyle={{
                      fontSize: 12,
                      borderColor: 'hsl(var(--border))',
                    }}
                  />
                  <Legend wrapperStyle={{ fontSize: 12 }} />
                  <Line
                    type="monotone"
                    dataKey="Perfect"
                    dot={false}
                    strokeDasharray="5 4"
                    strokeWidth={1.5}
                    stroke="hsl(var(--muted-foreground))"
                  />
                  <Line
                    type="monotone"
                    dataKey="Model"
                    dot={{ r: 4 }}
                    strokeWidth={2}
                    activeDot={{ r: 6 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  )
}
