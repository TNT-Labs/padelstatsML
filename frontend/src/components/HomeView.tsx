import { useEffect, useState, type FC } from 'react'
import { api, type Match } from '../api'

const STATUS_LABEL: Record<string, string> = {
  uploading:  'Caricamento',
  queued:     'In coda',
  processing: 'Elaborazione',
  completed:  'Completato',
  failed:     'Errore',
}

interface Props {
  onNewAnalysis: () => void
  onViewStats: (matchId: string) => void
}

export const HomeView: FC<Props> = ({ onNewAnalysis, onViewStats }) => {
  const [matches, setMatches]         = useState<Match[]>([])
  const [loading, setLoading]         = useState(true)
  const [editMode, setEditMode]       = useState(false)
  const [selected, setSelected]       = useState<Set<string>>(new Set())
  const [deleting, setDeleting]       = useState(false)
  const [confirmAll, setConfirmAll]   = useState(false)

  useEffect(() => {
    api.listMatches()
      .then(setMatches)
      .catch(() => {})
      .finally(() => setLoading(false))
  }, [])

  function enterEditMode() {
    setEditMode(true)
    setSelected(new Set())
    setConfirmAll(false)
  }

  function exitEditMode() {
    setEditMode(false)
    setSelected(new Set())
    setConfirmAll(false)
  }

  function toggleSelect(id: string) {
    setSelected(prev => {
      const next = new Set(prev)
      next.has(id) ? next.delete(id) : next.add(id)
      return next
    })
  }

  function toggleAll() {
    setSelected(prev =>
      prev.size === matches.length
        ? new Set()
        : new Set(matches.map(m => m.id))
    )
  }

  async function deleteSelected() {
    if (selected.size === 0 || deleting) return
    setDeleting(true)
    try {
      await Promise.all([...selected].map(id => api.deleteMatch(id)))
      setMatches(prev => prev.filter(m => !selected.has(m.id)))
      setSelected(new Set())
    } finally {
      setDeleting(false)
    }
  }

  async function deleteAll() {
    if (!confirmAll) { setConfirmAll(true); return }
    setDeleting(true)
    setConfirmAll(false)
    try {
      await Promise.all(matches.map(m => api.deleteMatch(m.id)))
      setMatches([])
      exitEditMode()
    } finally {
      setDeleting(false)
    }
  }

  const allSelected = matches.length > 0 && selected.size === matches.length

  return (
    <div className="layout">
      <div className="header">
        <h1>🎾 Padel Stats</h1>
        <div style={{ display: 'flex', gap: '.5rem' }}>
          {editMode ? (
            <button className="btn btn-ghost btn-sm" onClick={exitEditMode} disabled={deleting}>
              Annulla
            </button>
          ) : (
            <>
              {matches.length > 0 && (
                <button className="btn btn-ghost btn-sm" onClick={enterEditMode}>
                  Gestisci
                </button>
              )}
              <button className="btn btn-primary" onClick={onNewAnalysis}>+ Nuova analisi</button>
            </>
          )}
        </div>
      </div>

      <div className="card">
        {/* Card header */}
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem', flexWrap: 'wrap', gap: '.5rem' }}>
          <h2 style={{ margin: 0 }}>Partite recenti</h2>

          {editMode && matches.length > 0 && (
            <div style={{ display: 'flex', alignItems: 'center', gap: '.5rem', flexWrap: 'wrap' }}>
              <button
                className="btn btn-ghost btn-sm"
                onClick={toggleAll}
                disabled={deleting}
              >
                {allSelected ? 'Deseleziona tutto' : 'Seleziona tutto'}
              </button>
              <button
                className="btn btn-danger btn-sm"
                onClick={deleteSelected}
                disabled={selected.size === 0 || deleting}
              >
                {deleting && selected.size > 0
                  ? 'Eliminazione…'
                  : `Elimina selezionati${selected.size > 0 ? ` (${selected.size})` : ''}`}
              </button>
              <button
                className={`btn btn-sm ${confirmAll ? 'btn-danger' : 'btn-danger-ghost'}`}
                onClick={deleteAll}
                disabled={deleting}
                onBlur={() => setConfirmAll(false)}
              >
                {confirmAll ? 'Conferma eliminazione' : 'Elimina tutto'}
              </button>
            </div>
          )}
        </div>

        {loading && <p style={{ color: 'var(--muted)' }}>Caricamento…</p>}

        {!loading && matches.length === 0 && (
          <div style={{ textAlign: 'center', padding: '2rem 0' }}>
            <p style={{ fontSize: '2rem', marginBottom: '.5rem' }}>📹</p>
            <p style={{ color: 'var(--muted)' }}>Nessuna partita ancora. Carica il primo video!</p>
            <button
              className="btn btn-primary"
              style={{ marginTop: '1rem' }}
              onClick={onNewAnalysis}
            >
              Inizia ora
            </button>
          </div>
        )}

        {matches.map(m => (
          <div key={m.id} className="match-row">
            {editMode && (
              <input
                type="checkbox"
                checked={selected.has(m.id)}
                onChange={() => toggleSelect(m.id)}
                disabled={deleting}
                style={{ width: '1.1rem', height: '1.1rem', accentColor: 'var(--danger)', flexShrink: 0, cursor: 'pointer' }}
              />
            )}

            <div style={{ minWidth: 0, flex: 1 }}>
              <div style={{ fontWeight: 600, marginBottom: '.2rem', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                {m.title}
              </div>
              <div style={{ fontSize: '.8rem', color: 'var(--muted)' }}>
                {new Date(m.created_at).toLocaleDateString('it-IT', {
                  day: 'numeric', month: 'short', year: 'numeric',
                  hour: '2-digit', minute: '2-digit',
                })}
              </div>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: '.75rem', flexShrink: 0 }}>
              <span className={`badge badge-${m.status}`}>{STATUS_LABEL[m.status]}</span>

              {m.status === 'processing' && (
                <div style={{ fontSize: '.8rem', color: 'var(--muted)' }}>{m.progress}%</div>
              )}

              {m.status === 'completed' && !editMode && (
                <button
                  className="btn btn-ghost btn-sm"
                  onClick={() => onViewStats(m.id)}
                >
                  Visualizza
                </button>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
