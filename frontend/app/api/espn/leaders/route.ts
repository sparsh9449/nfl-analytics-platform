import { NextResponse } from 'next/server'

const WANT = ['passingYards', 'rushingYards', 'receivingYards', 'passingTouchdowns', 'rushingTouchdowns']

export async function GET() {
  const res = await fetch(
    'https://site.api.espn.com/apis/site/v3/sports/football/nfl/leaders',
    { next: { revalidate: 3600 } }
  )
  if (!res.ok) return NextResponse.json({ categories: [] })

  const raw = await res.json()

  const categories = (raw.leaders?.categories ?? [])
    .filter((c: { name: string }) => WANT.includes(c.name))
    .map((c: { name: string; displayName: string; leaders: Leader[] }) => ({
      key: c.name,
      label: c.displayName,
      leaders: (c.leaders ?? []).slice(0, 5).map((l: Leader) => ({
        name: l.athlete?.displayName ?? l.athlete?.shortName ?? '',
        team: l.team?.abbreviation ?? '',
        headshot: l.athlete?.headshot?.href ?? null,
        value: l.displayValue ?? '',
      })),
    }))

  return NextResponse.json({ categories })
}

interface Leader {
  displayValue: string
  athlete?: {
    displayName?: string
    shortName?: string
    headshot?: { href: string }
  }
  team?: {
    abbreviation?: string
  }
}
