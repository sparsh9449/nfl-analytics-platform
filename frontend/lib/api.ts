const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000'

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, { cache: 'no-store' })
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
  return res.json()
}

export interface Season {
  season: number
  min_week: number
  max_week: number
}

export interface GamePick {
  game_id: string
  home_team: string
  away_team: string
  spread_line: number
  total_line: number
  market_wp: number
  model_wp: number
  edge: number
  bet_team: string
  is_late_season: boolean
  week: number
}

export interface PreGamePick {
  game_id: string
  name: string
  date: string
  home_team: string
  away_team: string
  spread_line: number | null
  total_line: number | null
  market_wp: number
  model_wp: number
  home_edge: number  // signed (home perspective); positive = model likes home more than market
  edge: number       // always positive: pick team's advantage over market
  pick_team: string
  pick_side: 'home' | 'away'
  confidence: 'HIGH' | 'MEDIUM' | 'LOW'
  home_stats: Record<string, number | null>
  away_stats: Record<string, number | null>
}

export interface UpcomingPicksResponse {
  n_games: number
  model: string
  warmup_season: number
  picks: PreGamePick[]
}

export interface GamesResponse {
  season: number
  week: number
  n_games: number
  games: GamePick[]
}

export interface HistoryGame {
  game_id: string
  home_team: string
  away_team: string
  spread_line: number
  home_score: number
  away_score: number
  home_won: number
  market_wp: number
  model_wp: number
  edge: number
  model_correct: number
}

export interface HistoryResponse {
  season: number
  week: number
  n_games: number
  games: HistoryGame[]
}

export interface CalibrationPoint {
  predicted: number
  actual: number
  diff: number
}

export interface MetricsResponse {
  logistic_regression: {
    val_2023: { roc_auc: number; brier: number }
  }
  xgboost: {
    val_2023: { roc_auc: number; brier: number }
    test_2024_2025_calibrated: {
      roc_auc: number
      brier: number
      calibration: CalibrationPoint[]
    }
  }
}

export interface EdgeBucket {
  bucket: string
  count: number
  market_wp_mean: number
  model_wp_mean: number
  actual_win_rate: number
  edge_mean: number
}

export interface EdgeSummaryResponse {
  seasons: number[]
  n_games: number
  model: string
  overall: {
    mean_edge: number
    std_edge: number
    pct_pos_edge_5: number
    pct_neg_edge_5: number
    pos_edge_actual_win_rate: number
    neg_edge_actual_win_rate: number
  }
  buckets: EdgeBucket[]
}

export interface WeekBreakdownEntry {
  week_group: string
  bucket: string
  n: number
  market_wp: number
  model_wp: number
  actual_wr: number
  edge_mean: number
  pct_null_roll: number
}

export interface WeekBreakdownResponse {
  seasons: number[]
  n_games: number
  full_breakdown: WeekBreakdownEntry[]
  pos_edge_gt10_by_week_group: Array<{
    week_group: string
    n: number
    market_wp: number
    model_wp: number
    actual_wr: number
    edge_mean: number
    pct_null_roll: number
  }>
}

export interface RoiEntry {
  season?: number
  week_group: string
  n: number
  wins: number
  win_rate: number
  roi: number
}

export interface RoiResponse {
  seasons: number[]
  edge_thresh: number
  odds: number
  breakeven_wr: number
  recommended_thresh: number
  overall: RoiEntry[]
  by_season: RoiEntry[]
}

export interface SeasonAccuracyEntry {
  season: number
  n_games: number
  accuracy: number
  brier: number
}

export const api = {
  seasons: () => get<{ seasons: Season[] }>('/seasons'),
  games: (season: number, week: number, threshold?: number) => {
    const qs = threshold != null ? `?threshold=${threshold}` : ''
    return get<GamesResponse>(`/games/${season}/${week}${qs}`)
  },
  history: (season: number, week: number) =>
    get<HistoryResponse>(`/history/${season}/${week}`),
  metrics: () => get<MetricsResponse>('/model/metrics'),
  edgeSummary: () => get<EdgeSummaryResponse>('/model/edge-summary'),
  weekBreakdown: () => get<WeekBreakdownResponse>('/model/week-breakdown'),
  roi: () => get<RoiResponse>('/model/roi'),
  seasonAccuracy: () => get<{ seasons: SeasonAccuracyEntry[] }>('/model/season-accuracy'),
  upcomingPicks: (week?: number) => {
    const qs = week != null ? `?week=${week}` : ''
    return get<UpcomingPicksResponse>(`/picks/upcoming${qs}`)
  },
}
