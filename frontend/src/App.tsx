import { useCallback, useEffect, useState } from 'react'
import { api, setSessionLostHandler, type Match, type MatchStats, type Me } from './api'
import { AdminView } from './components/AdminView'
import { CalibrationView } from './components/CalibrationView'
import { HomeView } from './components/HomeView'
import { LoginView } from './components/LoginView'
import { PasswordView } from './components/PasswordView'
import { PlayerIdentificationView } from './components/PlayerIdentificationView'
import { ProcessingView } from './components/ProcessingView'
import { StatsView } from './components/StatsView'
import { UploadView } from './components/UploadView'
import { UserBar } from './components/UserBar'

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
  | { name: 'upload'; resume?: Match }
  | { name: 'calibrate'; match: Match }
  | { name: 'processing'; matchId: string; autoStart: boolean }
  | { name: 'loading'; matchId: string; then: 'identify' | 'stats' }
  | { name: 'identify'; stats: MatchStats }
  | { name: 'stats'; stats: MatchStats }

/**
 * Access: nothing but the sign-in form is shown until the server confirms a
 * session, and a temporary password must be replaced before anything else.
 * The rules are enforced by the server; this only decides what to show.
 */
export default function App() {
  const [me, setMe] = useState<Me | null | undefined>(undefined)
  const [page, setPage] = useState<'app' | 'password' | 'admin'>('app')

  const check = useCallback(() => {
    api
      .me()
      .then(setMe)
      .catch(() => setMe(null))
  }, [])

  useEffect(() => {
    check()
    setSessionLostHandler(check)
    return () => setSessionLostHandler(null)
  }, [check])

  const logout = useCallback(async () => {
    await api.logout().catch(() => undefined)
    setPage('app')
    setMe(null)
  }, [])

  if (me === undefined) {
    return (
      <div className="layout" style={{ textAlign: 'center', paddingTop: '5rem' }}>
        <p className="muted-note">Caricamento…</p>
      </div>
    )
  }
  if (me === null) {
    return (
      <LoginView
        onSignedIn={user => {
          setPage('app')
          setMe(user)
        }}
      />
    )
  }
  if (me.must_change_password) {
    return <PasswordView me={me} forced onDone={setMe} onLogout={logout} />
  }

  return (
    <>
      <UserBar
        me={me}
        onPassword={() => setPage('password')}
        onAdmin={() => setPage('admin')}
        onLogout={logout}
      />
      {page === 'password' && (
        <PasswordView me={me} onDone={user => { setMe(user); setPage('app') }} onCancel={() => setPage('app')} />
      )}
      {page === 'admin' && me.role === 'admin' && <AdminView me={me} onBack={() => setPage('app')} />}
      {/* Kept mounted while another page is open: an upload in progress or
          the screen the user was on must survive a look at the settings. */}
      <div hidden={page !== 'app'}>
        <Workspace me={me} />
      </div>
    </>
  )
}

function Workspace({ me }: { me: Me }) {
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
        setView({ name: 'upload', resume: match })
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
          resume={view.resume}
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
          onReanalyse={() =>
            setView({ name: 'processing', matchId: view.stats.match_id, autoStart: true })
          }
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
          <HomeView me={me} onNew={() => setView({ name: 'upload' })} onOpen={open} />
        </>
      )
  }
}
