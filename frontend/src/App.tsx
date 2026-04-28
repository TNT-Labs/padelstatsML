import { useEffect, useState } from 'react'
import { api, type Match, type MatchStats } from './api'
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

  // Load stats + player names for an already-completed match from HomeView
  useEffect(() => {
    if (view.name !== 'stats-loading') return
    const { matchId } = view

    // Fetch stats and match (for player names) in parallel
    Promise.all([api.getStats(matchId), api.getMatch(matchId)])
      .then(([stats, match]: [MatchStats, Match]) =>
        setView({
          name: 'stats',
          stats,
          playerNames: match.player_names ?? [],
        })
      )
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
      <div className="layout" style={{ textAlign: 'center', paddingTop: '5rem' }}>
        <p style={{ color: 'var(--muted)', fontSize: '1.1rem' }}>Caricamento statistiche…</p>
      </div>
    )
  }

  return (
    <StatsView
      stats={view.stats}
      playerNames={view.playerNames}
      onNewAnalysis={() => setView({ name: 'home' })}
    />
  )
}
