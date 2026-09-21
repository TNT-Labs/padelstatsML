import { useEffect, useRef, useState, type FC } from 'react'
import { api, type Match } from '../api'
import { useMatch } from '../hooks/useMatch'

interface Props {
  matchId: string
  /** Start the analysis on mount. False when reopening a job already running. */
  autoStart: boolean
  onCompleted: () => void
  onFailed: (match: Match) => void
  onBack: () => void
}

export const ProcessingView: FC<Props> = ({ matchId, autoStart, onCompleted, onFailed, onBack }) => {
  const { match, error, refresh } = useMatch(matchId)
  const [startError, setStartError] = useState<string | null>(null)
  const startRequested = useRef(false)

  useEffect(() => {
    if (!autoStart || startRequested.current) return
    startRequested.current = true
    api
      .startAnalysis(matchId)
      .then(() => refresh())
      .catch((e: Error) => setStartError(e.message))
  }, [autoStart, matchId, refresh])

  useEffect(() => {
    if (match?.status === 'completed') onCompleted()
    if (match?.status === 'failed') onFailed(match)
  }, [match, onCompleted, onFailed])

  const progress = match?.progress ?? 0

  return (
    <div className="layout" style={{ maxWidth: 620 }}>
      <div className="header">
        <h1>Analisi in corso</h1>
        <button className="btn btn-ghost btn-sm" onClick={onBack}>
          ← Partite
        </button>
      </div>

      <div className="card" style={{ textAlign: 'center', padding: '2.5rem 1.5rem' }}>
        <h2 style={{ marginBottom: '.25rem' }}>{match?.title ?? '…'}</h2>
        <p className="muted-note" style={{ marginBottom: '1.25rem' }}>
          {match?.progress_message ?? 'In attesa del worker…'}
        </p>

        <div className="progress-track">
          <div className="progress-fill" style={{ width: `${progress}%` }} />
        </div>
        <p className="muted-note" style={{ marginTop: '.5rem' }}>{progress}%</p>

        <div className="callout callout-info" style={{ marginTop: '1.5rem', textAlign: 'left' }}>
          Su Raspberry Pi 5 l&apos;analisi procede a circa un terzo del tempo reale:
          una partita di un&apos;ora richiede all&apos;incirca un&apos;ora di elaborazione.
          Puoi chiudere questa pagina — il lavoro continua sul Pi.
        </div>

        {(startError || error) && <p className="error-msg">{startError ?? error}</p>}
      </div>
    </div>
  )
}
