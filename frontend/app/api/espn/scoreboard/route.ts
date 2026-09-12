import { NextResponse } from 'next/server'

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url)
  const week = searchParams.get('week')

  const url = new URL('https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard')
  url.searchParams.set('seasontype', '2')
  if (week) url.searchParams.set('week', week)

  const res = await fetch(url.toString(), { next: { revalidate: 60 } })
  if (!res.ok) return NextResponse.json({ events: [], week: { number: 1 } }, { status: 200 })

  const raw = await res.json()

  const events = (raw.events ?? []).map((e: ESPN.Event) => {
    const comp = e.competitions?.[0]
    const home = comp?.competitors?.find((c: ESPN.Competitor) => c.homeAway === 'home')
    const away = comp?.competitors?.find((c: ESPN.Competitor) => c.homeAway === 'away')
    const statusType = comp?.status?.type ?? e.status?.type
    const odds = comp?.odds?.[0]
    const tv = comp?.broadcasts?.[0]?.names?.[0] ?? comp?.geoBroadcasts?.[0]?.media?.shortName

    return {
      id: e.id,
      name: e.shortName ?? e.name,
      date: e.date,
      status: statusType?.state ?? 'pre',
      statusText: statusType?.shortDetail ?? statusType?.detail ?? 'Scheduled',
      homeTeam: teamFrom(home),
      awayTeam: teamFrom(away),
      spread: odds?.details ?? null,
      overUnder: odds?.overUnder ?? null,
      tv: tv ?? null,
    }
  })

  return NextResponse.json({
    week: raw.week?.number ?? 1,
    season: raw.season?.year ?? new Date().getFullYear(),
    events,
  })
}

function teamFrom(c: ESPN.Competitor | undefined) {
  if (!c) return null
  return {
    id: c.team?.id ?? '',
    abbr: c.team?.abbreviation ?? '',
    name: c.team?.displayName ?? '',
    shortName: c.team?.shortDisplayName ?? '',
    color: c.team?.color ?? '111111',
    logo: c.team?.logo ?? '',
    score: c.score ?? null,
    winner: c.winner ?? false,
  }
}

declare namespace ESPN {
  interface Event {
    id: string
    name: string
    shortName?: string
    date: string
    status?: { type: StatusType }
    competitions?: Competition[]
  }
  interface Competition {
    competitors?: Competitor[]
    status?: { type: StatusType }
    odds?: { details: string; overUnder: number }[]
    broadcasts?: { names?: string[] }[]
    geoBroadcasts?: { media?: { shortName: string } }[]
  }
  interface Competitor {
    homeAway: 'home' | 'away'
    score?: string
    winner?: boolean
    team?: {
      id: string
      abbreviation: string
      displayName: string
      shortDisplayName: string
      color: string
      logo: string
    }
  }
  interface StatusType {
    state: string
    detail: string
    shortDetail: string
  }
}
