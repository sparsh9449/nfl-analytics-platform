import { NextResponse } from 'next/server'

export async function GET() {
  const res = await fetch(
    'https://site.web.api.espn.com/apis/v2/sports/football/nfl/standings',
    { next: { revalidate: 3600 } }
  )
  if (!res.ok) return NextResponse.json({ standings: [] })

  const raw = await res.json()
  const rows: StandingRow[] = []

  for (const conf of raw.children ?? []) {
    const confAbbr: string = conf.abbreviation ?? ''
    for (const entry of conf.standings?.entries ?? []) {
      const t = entry.team ?? {}
      const stat = (name: string) =>
        entry.stats?.find((s: { name: string; value?: number; displayValue?: string }) => s.name === name)

      rows.push({
        abbr: t.abbreviation ?? '',
        name: t.displayName ?? '',
        shortName: t.shortDisplayName ?? t.name ?? '',
        logo: t.logos?.[0]?.href ?? '',
        color: '1e2d3f',
        conference: confAbbr,
        wins: stat('wins')?.value ?? 0,
        losses: stat('losses')?.value ?? 0,
        ties: stat('ties')?.value ?? 0,
        pct: stat('winPercent')?.displayValue ?? '.000',
        pf: stat('pointsFor')?.value ?? 0,
        pa: stat('pointsAgainst')?.value ?? 0,
        streak: stat('streak')?.displayValue ?? '',
        divisionRecord: stat('divisionRecord')?.displayValue ?? '',
      })
    }
  }

  rows.sort((a, b) => parseFloat(b.pct) - parseFloat(a.pct))

  return NextResponse.json({ standings: rows })
}

interface StandingRow {
  abbr: string
  name: string
  shortName: string
  logo: string
  color: string
  conference: string
  wins: number
  losses: number
  ties: number
  pct: string
  pf: number
  pa: number
  streak: string
  divisionRecord: string
}
