import { useEffect, useState, type FC } from 'react'
import { api, type Match, type MatchStatus } from '../api'

const STATUS_LABEL: Record<MatchStatus, string> = {
  uploading: 'Caricamento incompleto',
  needs_calibration: 'Da calibrare',
  ready: 'Pronta',
  queued: 'In coda',
  analyzing: 'Analisi',
  completed: 'Completata',
  failed: 'Errore',
}

/** What tapping the row should do, per state. */
const ACTION_LABEL: Partial<Record<MatchStatus, string>> = {
  needs_calibration: 'Calibra',
  ready: 'Avvia analisi',
  queued: 'Vedi stato',
  analyzing: 'Vedi stato',
  completed: 'Statistiche',
  failed: 'Riprova',
}

interface Props {
  onNew: () => void
  onOpen: (match: Match) => void
}

export const HomeView: FC<Props> = ({ onNew, onOpen }) => {
  const [matches, setMatches] = useState<Match[]>([])
  const [loading, setLoading] = useState(true)
  const [editMode, setEditMode] = useState(false)
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [busy, setBusy] = useState(false)
  const [confirmAll, setConfirmAll] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const load = () =>
    api
      .listMatches()
      .then(setMatches)
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false))

  useEffect(() => {
    void load()
    // Keep the list fresh while an analysis runs in the background.
    const timer = window.setInterval(() => void load(), 15_000)
    return () => window.clearInterval(timer)
  }, [])

  const exitEdit = () => {
    setEditMode(false)
    setSelected(new Set())
    setConfirmAll(false)
  }

  const toggle = (id: string) =>
    setSelected(prev => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })

  const remove = async (ids: string[]) => {
    if (!ids.length || busy) return
    setBusy(true)
    setError(null)
    // Sequential, not parallel: a Pi serving four concurrent deletes while
    // the worker holds the SQLite write lock is how timeouts happen.
    const removed: string[] = []
    try {
      for (const id of ids) {
        try {
          await api.deleteMatch(id)
          removed.push(id)
        } catch (e) {
          setError((e as Error).message)
          break
        }
      }
    } finally {
      setMatches(prev => prev.filter(m => !removed.includes(m.id)))
      setSelected(new Set())
      setConfirmAll(false)
      setBusy(false)
    }
  }

  const allSelected = matches.length > 0 && selected.size === matches.length

  return (
    <div className="layout">
      <div className="header">
        <h1>🎾 Padel Stats</h1>
        <div style={{ display: 'flex', gap: '.5rem' }}>
          {editMode ? (
            <button className="btn btn-ghost btn-sm" onClick={exitEdit} disabled={busy}>
              Fine
            </button>
          ) : (
            <>
              {matches.length > 0 && (
                <button className="btn btn-ghost btn-sm" onClick={() => setEditMode(true)}>
                  Gestisci
                </button>
              )}
              <button className="btn btn-primary" onClick={onNew}>
                + Nuova partita
              </button>
            </>
          )}
        </div>
      </div>

      {error && <p className="error-msg">{error}</p>}

      <div className="card">
        <div className="card-head">
          <h2 style={{ margin: 0 }}>Partite</h2>
          {editMode && matches.length > 0 && (
            <div className="action-row">
              <button
                className="btn btn-ghost btn-sm"
                disabled={busy}
                onClick={() =>
                  setSelected(allSelected ? new Set() : new Set(matches.map(m => m.id)))
                }
              >
                {allSelected ? 'Deseleziona tutto' : 'Seleziona tutto'}
              </button>
              <button
                className="btn btn-danger btn-sm"
                disabled={selected.size === 0 || busy}
                onClick={() => remove([...selected])}
              >
                Elimina ({selected.size})
              </button>
              <button
                className={`btn btn-sm ${confirmAll ? 'btn-danger' : 'btn-danger-ghost'}`}
                disabled={busy}
                onBlur={() => setConfirmAll(false)}
                onClick={() =>
                  confirmAll ? remove(matches.map(m => m.id)) : setConfirmAll(true)
                }
              >
                {confirmAll ? 'Conferma: elimina tutto' : 'Elimina tutto'}
              </button>
            </div>
          )}
        </div>

        {loading && <p className="muted-note">Caricamento…</p>}

        {!loading && matches.length === 0 && (
          <div style={{ textAlign: 'center', padding: '2rem 0' }}>
            <p style={{ fontSize: '2rem', marginBottom: '.5rem' }}>📹</p>
            <p className="muted-note">Nessuna partita. Carica il primo video.</p>
            <button className="btn btn-primary" style={{ marginTop: '1rem' }} onClick={onNew}>
              Inizia
            </button>
          </div>
        )}

        {matches.map(match => (
          <div key={match.id} className="match-row">
            {editMode && (
              <input
                type="checkbox"
                checked={selected.has(match.id)}
                onChange={() => toggle(match.id)}
                disabled={busy}
                className="row-check"
              />
            )}

            <div style={{ minWidth: 0, flex: 1 }}>
              <div className="match-title">{match.title}</div>
              <div className="muted-note">
                {new Date(match.created_at).toLocaleDateString('it-IT', {
                  day: 'numeric',
                  month: 'short',
                  hour: '2-digit',
                  minute: '2-digit',
                })}
                {match.duration_seconds
                  ? ` · ${Math.round(match.duration_seconds / 60)} min di video`
                  : ''}
              </div>
              {match.status === 'analyzing' && match.progress_message && (
                <div className="muted-note">{match.progress_message}</div>
              )}
              {match.status === 'failed' && match.error_message && (
                <div className="error-msg" style={{ marginTop: '.2rem' }}>
                  {match.error_message}
                </div>
              )}
            </div>

            <div className="match-actions">
              <span className={`badge badge-${match.status}`}>{STATUS_LABEL[match.status]}</span>
              {match.status === 'analyzing' && (
                <span className="muted-note">{match.progress}%</span>
              )}
              {!editMode && ACTION_LABEL[match.status] && (
                <button className="btn btn-ghost btn-sm" onClick={() => onOpen(match)}>
                  {ACTION_LABEL[match.status]}
                </button>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
