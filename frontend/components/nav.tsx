'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { cn } from '@/lib/utils'

const links = [
  { href: '/picks', label: 'Weekly Picks' },
  { href: '/performance', label: 'Model Performance' },
  { href: '/edge', label: 'Edge Analysis' },
  { href: '/history', label: 'History' },
]

export default function Nav() {
  const pathname = usePathname()
  return (
    <header className="border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 sticky top-0 z-10">
      <div className="max-w-6xl mx-auto px-4 h-14 flex items-center gap-8">
        <span className="font-semibold text-sm tracking-tight shrink-0">NFL Analytics</span>
        <nav className="flex items-center gap-6 overflow-x-auto">
          {links.map(({ href, label }) => (
            <Link
              key={href}
              href={href}
              className={cn(
                'text-sm whitespace-nowrap transition-colors',
                pathname === href
                  ? 'text-foreground font-medium'
                  : 'text-muted-foreground hover:text-foreground'
              )}
            >
              {label}
            </Link>
          ))}
        </nav>
      </div>
    </header>
  )
}
