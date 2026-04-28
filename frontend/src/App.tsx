import { useEffect, useState } from 'react'
import { api, type MatchStats } from './api'
import { HomeView }    from './components/HomeView'
import { AnalyzeView } from './components/AnalyzeView'
import { StatsView }   from './components/StatsView'

type View =
  | { name: 'home' }
  | { name: 'analyze' }
  | { name: 'stats'; stats: MatchStats; playerNames: string[] }
  | { name: 'stats-loading'; matchId: string }

export default function App() {
  const [view, setView] = useState<View>({ name: 'home' })

  // Navigate to stats for a match that already exists (from HomeView)
  useEffect(() => {
    if (view.name !== 'stats-loading') return
    const { matchId } = view
    api.getStats(matchId)
      .then(stats => setView({ name: 'stats', stats, playerNames: [] }))
      .catch(() => setView({ name: 'home' }))
  }, [view])

  if (view.name === 'home') {
    return (
      <HomeView
        onNewAnalysis={() => setView({ name: 'analyze' })}
        onViewStats={id => setView({ name: 'stats-loading', matchId: id })}
      />
    )
  }

  if (view.name === 'analyze') {
    return (
      <AnalyzeView
        onDone={(stats, playerNames) => setView({ name: 'stats', stats, playerNames })}
        onBack={() => setView({ name: 'home' })}
      />
    )
  }

  if (view.name === 'stats-loading') {
    return (
      <div className="layout" style={{ textAlign: 'center', paddingTop: '4rem' }}>
        <p style={{ color: 'var(--muted)' }}>Caricamento statistiche…</p>
      </div>
    )
  }

  // stats view
  return (
    <StatsView
      stats={view.stats}
      playerNames={view.playerNames}
      onNewAnalysis={() => setView({ name: 'home' })}
    />
  )
}
