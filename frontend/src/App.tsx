import { useCallback, useEffect, useState } from 'react'
import { api, type Match, type MatchStats } from './api'
import { CalibrationView } from './components/CalibrationView'
import { HomeView } from './components/HomeView'
import { PlayerIdentificationView } from './components/PlayerIdentificationView'
import { ProcessingView } from './components/ProcessingView'
import { StatsView } from './components/StatsView'
import { UploadView } from './components/UploadView'

/**
 * Screen flow:
 *
 *   home ─┬─ upload ──> calibrate ──> processing ──> identify ──> stats
 *         └─ open an existing match, resuming at the right step for its state
 *
 * The calibration step sits between upload and analysis by design: without a
 * confirmed homography the backend refuses to queue the job, because every
 * number it would produce is in metres.
 */
type View =
  | { name: 'home' }
  | { name: 'upload' }
  | { name: 'calibrate'; match: Match }
  | { name: 'processing'; matchId: string; autoStart: boolean }
  | { name: 'loading'; matchId: string; then: 'identify' | 'stats' }
  | { name: 'identify'; stats: MatchStats }
  | { name: 'stats'; stats: MatchStats }

export default function App() {
  const [view, setView] = useState<View>({ name: 'home' })
  const [error, setError] = useState<string | null>(null)

  const home = useCallback(() => {
    setError(null)
    setView({ name: 'home' })
  }, [])

  /** Resume an existing match at whatever step it is actually waiting on. */
  const open = useCallback(async (match: Match) => {
    setError(null)
    switch (match.status) {
      case 'uploading':
        setError(
          'Il caricamento di questa partita non è stato completato. Eliminala e ricaricala.',
        )
        return
      case 'needs_calibration':
      case 'failed':
        setView({ name: 'calibrate', match })
        return
      case 'ready':
        setView({ name: 'processing', matchId: match.id, autoStart: true })
        return
      case 'queued':
      case 'analyzing':
        setView({ name: 'processing', matchId: match.id, autoStart: false })
        return
      case 'completed':
        setView({
          name: 'loading',
          matchId: match.id,
          then: match.player_names?.some(Boolean) ? 'stats' : 'identify',
        })
        return
    }
  }, [])

  // Fetch stats for the 'loading' step, then move on to identify or stats.
  useEffect(() => {
    if (view.name !== 'loading') return
    let cancelled = false
    const { matchId, then } = view

    api
      .getStats(matchId)
      .then(stats => {
        if (cancelled) return
        setView(then === 'identify' ? { name: 'identify', stats } : { name: 'stats', stats })
      })
      .catch((e: Error) => {
        if (cancelled) return
        setError(e.message)
        setView({ name: 'home' })
      })

    return () => {
      cancelled = true
    }
  }, [view])

  const afterCalibration = useCallback(async (matchId: string) => {
    const match = await api.getMatch(matchId).catch(() => null)
    setView({ name: 'processing', matchId, autoStart: match?.status === 'ready' })
  }, [])

  switch (view.name) {
    case 'upload':
      return (
        <UploadView
          onBack={home}
          onUploaded={async matchId => {
            const match = await api.getMatch(matchId)
            setView({ name: 'calibrate', match })
          }}
        />
      )

    case 'calibrate':
      return (
        <CalibrationView
          match={view.match}
          onBack={home}
          onCalibrated={() => afterCalibration(view.match.id)}
        />
      )

    case 'processing':
      return (
        <ProcessingView
          matchId={view.matchId}
          autoStart={view.autoStart}
          onBack={home}
          onCompleted={() =>
            setView({ name: 'loading', matchId: view.matchId, then: 'identify' })
          }
          onFailed={match => {
            setError(match.error_message ?? 'Analisi fallita.')
            setView({ name: 'home' })
          }}
        />
      )

    case 'loading':
      return (
        <div className="layout" style={{ textAlign: 'center', paddingTop: '5rem' }}>
          <p className="muted-note">Caricamento statistiche…</p>
        </div>
      )

    case 'identify':
      return (
        <PlayerIdentificationView
          stats={view.stats}
          onSkip={() => setView({ name: 'stats', stats: view.stats })}
          onConfirm={() =>
            setView({ name: 'loading', matchId: view.stats.match_id, then: 'stats' })
          }
        />
      )

    case 'stats':
      return (
        <StatsView
          stats={view.stats}
          onBack={home}
          onRename={() => setView({ name: 'identify', stats: view.stats })}
        />
      )

    default:
      return (
        <>
          {error && (
            <div className="layout" style={{ paddingBottom: 0 }}>
              <div className="callout callout-error">{error}</div>
            </div>
          )}
          <HomeView onNew={() => setView({ name: 'upload' })} onOpen={open} />
        </>
      )
  }
}
