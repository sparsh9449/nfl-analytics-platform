import { NextResponse } from 'next/server'

export async function GET() {
  const res = await fetch(
    'https://site.api.espn.com/apis/site/v2/sports/football/nfl/news?limit=8',
    { next: { revalidate: 300 } }
  )
  if (!res.ok) return NextResponse.json({ articles: [] })

  const raw = await res.json()
  const articles = (raw.articles ?? []).slice(0, 8).map((a: Record<string, unknown>) => ({
    headline: a.headline ?? '',
    description: a.description ?? '',
    published: a.published ?? '',
    url: (a.links as Record<string, Record<string, string>>)?.web?.href ?? '#',
    image: ((a.images as Record<string, string>[])?.[0]?.url) ?? null,
    byline: a.byline ?? null,
  }))

  return NextResponse.json({ articles })
}
