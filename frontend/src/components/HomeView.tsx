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
  const [matches, setMatches] = useState<Match[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    api.listMatches()
      .then(setMatches)
      .catch(() => {})
      .finally(() => setLoading(false))
  }, [])

  return (
    <div className="layout">
      <div className="header">
        <h1>🎾 Padel Stats</h1>
        <button className="btn btn-primary" onClick={onNewAnalysis}>+ Nuova analisi</button>
      </div>

      <div className="card">
        <h2 style={{ marginBottom: '1rem' }}>Partite recenti</h2>

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
            <div style={{ minWidth: 0 }}>
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

              {m.status === 'completed' && (
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
