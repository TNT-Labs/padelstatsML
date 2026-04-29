import { useState, type FC } from 'react'
import type { MatchStats } from '../api'

interface Props {
  stats: MatchStats
  onConfirm: (names: string[]) => void
}

export const PlayerIdentificationView: FC<Props> = ({ stats, onConfirm }) => {
  const playerIds = Object.keys(stats.per_player).sort()
  const [names, setNames] = useState<string[]>(playerIds.map(() => ''))

  const setName = (i: number, val: string) =>
    setNames(ns => ns.map((n, j) => (j === i ? val : n)))

  return (
    <div className="layout">
      <div className="header">
        <h1>Chi è chi?</h1>
      </div>

      <p style={{ color: 'var(--muted)', marginBottom: '1.5rem', textAlign: 'center' }}>
        Associa un nome a ogni giocatore riconosciuto nel video. Puoi lasciare i campi vuoti.
      </p>

      <div className="grid-2" style={{ marginBottom: '2rem' }}>
        {playerIds.map((pid, i) => {
          const cropUrl = stats.per_player[pid]?.crop_url
          return (
            <div key={pid} className="card" style={{ textAlign: 'center' }}>
              {cropUrl ? (
                <img
                  src={cropUrl}
                  alt={`Giocatore ${i + 1}`}
                  style={{
                    width: 120,
                    height: 180,
                    objectFit: 'cover',
                    borderRadius: 8,
                    marginBottom: '0.75rem',
                    display: 'block',
                    margin: '0 auto 0.75rem',
                  }}
                />
              ) : (
                <div
                  style={{
                    width: 120,
                    height: 180,
                    borderRadius: 8,
                    background: 'var(--surface)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    margin: '0 auto 0.75rem',
                    fontSize: '2.5rem',
                    color: 'var(--muted)',
                  }}
                >
                  👤
                </div>
              )}
              <div style={{ fontSize: '.8rem', color: 'var(--muted)', marginBottom: '.5rem' }}>
                Giocatore {i + 1}
              </div>
              <input
                type="text"
                value={names[i]}
                placeholder={`Giocatore ${i + 1}`}
                onChange={e => setName(i, e.target.value)}
                style={{ width: '100%', textAlign: 'center' }}
              />
            </div>
          )
        })}
      </div>

      <div style={{ textAlign: 'center' }}>
        <button
          className="btn btn-primary"
          style={{ minWidth: 200 }}
          onClick={() => onConfirm(names)}
        >
          Vedi statistiche →
        </button>
      </div>
    </div>
  )
}
