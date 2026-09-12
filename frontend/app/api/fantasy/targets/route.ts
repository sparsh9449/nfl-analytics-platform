import { NextResponse } from 'next/server'

const POSITIONS = ['QB', 'RB', 'WR', 'TE'] as const

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url)
  const season = searchParams.get('season') ?? String(new Date().getFullYear())
  const week = searchParams.get('week') ?? '1'

  const playersRes = await fetch('https://api.sleeper.app/v1/players/nfl', {
    next: { revalidate: 86400 },
  })

  if (!playersRes.ok) {
    return NextResponse.json({ byPosition: {}, season, week })
  }

  const players = (await playersRes.json()) as Record<string, SleeperPlayer>

  // Try requested season/week, then fall back to 2025 week 18 if empty
  async function fetchProjections(s: string, w: string): Promise<Record<string, SleeperProj> | null> {
    const res = await fetch(`https://api.sleeper.app/v1/projections/nfl/regular/${s}/${w}`, {
      next: { revalidate: 3600 },
    })
    if (!res.ok) return null
    const data = await res.json()
    if (!data || typeof data !== 'object' || Object.keys(data).length === 0) return null
    return data as Record<string, SleeperProj>
  }

  let projections = await fetchProjections(season, week)
  let effectiveSeason = season
  let effectiveWeek = week

  if (!projections) {
    projections = await fetchProjections('2025', '18')
    effectiveSeason = '2025'
    effectiveWeek = '18'
  }

  if (!projections) {
    return NextResponse.json({ byPosition: {}, season, week })
  }

  const targets: FantasyTarget[] = []

  for (const [pid, proj] of Object.entries(projections)) {
    const player = players[pid]
    if (!player) continue
    if (!(POSITIONS as readonly string[]).includes(player.position)) continue
    if (!player.team || player.active === false) continue
    const pts = proj.pts_ppr ?? 0
    if (pts < 1) continue

    targets.push({
      player_id: pid,
      name: player.full_name ?? `${player.first_name ?? ''} ${player.last_name ?? ''}`.trim(),
      position: player.position,
      team: player.team,
      pts_ppr: Math.round(pts * 10) / 10,
      pass_yd: proj.pass_yd,
      pass_td: proj.pass_td,
      rush_att: proj.rush_att,
      rush_yd: proj.rush_yd,
      rush_td: proj.rush_td,
      rec: proj.rec,
      rec_yd: proj.rec_yd,
      rec_td: proj.rec_td,
      injury_status: player.injury_status ?? null,
    })
  }

  targets.sort((a, b) => b.pts_ppr - a.pts_ppr)

  const byPosition: Record<string, FantasyTarget[]> = {}
  for (const pos of POSITIONS) {
    byPosition[pos] = targets.filter((t) => t.position === pos).slice(0, 15)
  }

  return NextResponse.json({ byPosition, season: effectiveSeason, week: effectiveWeek })
}

interface SleeperPlayer {
  player_id?: string
  first_name?: string
  last_name?: string
  full_name?: string
  position: string
  team?: string
  active?: boolean
  injury_status?: string
}

interface SleeperProj {
  pts_ppr?: number
  pass_yd?: number
  pass_td?: number
  rush_att?: number
  rush_yd?: number
  rush_td?: number
  rec?: number
  rec_yd?: number
  rec_td?: number
}

interface FantasyTarget {
  player_id: string
  name: string
  position: string
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
