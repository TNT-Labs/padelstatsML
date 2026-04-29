import type { FC } from 'react'
import type { MatchStats } from '../api'
import { CourtHeatmap } from './CourtHeatmap'
import { ShotChart } from './ShotChart'

const PLAYER_COLORS = ['#ef4444', '#3b82f6', '#f59e0b', '#8b5cf6']

interface Props {
  stats: MatchStats
  playerNames?: (string | null)[]
  onNewAnalysis: () => void
}

export const StatsView: FC<Props> = ({ stats, playerNames, onNewAnalysis }) => {
  const players = Object.entries(stats.per_player).sort(([a], [b]) => Number(a) - Number(b))

  return (
    <div className="layout">
      <div className="header">
        <h1>🏆 Risultati partita</h1>
        <button className="btn btn-ghost btn-sm" onClick={onNewAnalysis}>+ Nuova analisi</button>
      </div>

      {/* Summary row */}
      <div className="grid-4" style={{ marginBottom: '1.5rem' }}>
        <div className="card card-sm stat-card">
          <div className="val">{stats.rallies_count}</div>
          <div className="lbl">Rally</div>
        </div>
        <div className="card card-sm stat-card">
          <div className="val">{stats.total_shots}</div>
          <div className="lbl">Colpi totali</div>
        </div>
        <div className="card card-sm stat-card">
          <div className="val">
            {players.reduce((s, [, p]) => s + p.winners, 0)}
          </div>
          <div className="lbl">Vincenti totali</div>
        </div>
        <div className="card card-sm stat-card">
          <div className="val">
            {Math.round(players.reduce((s, [, p]) => s + p.distance_m, 0) / Math.max(players.length, 1))}m
          </div>
          <div className="lbl">Distanza media</div>
        </div>
      </div>

      {/* Heatmap + Shot chart side by side */}
      <div className="grid-2" style={{ marginBottom: '1.5rem' }}>
        <div className="card">
          <h2 style={{ marginBottom: '1rem' }}>Posizionamento in campo</h2>
          <CourtHeatmap
            heatmaps={stats.heatmaps}
            playerNames={playerNames}
            width={240}
          />
        </div>
        <div className="card">
          <h2 style={{ marginBottom: '1rem' }}>Distribuzione colpi</h2>
          <ShotChart perPlayer={stats.per_player} playerNames={playerNames} />
        </div>
      </div>

      {/* Per-player cards */}
      <h2 style={{ marginBottom: '.75rem' }}>Statistiche per giocatore</h2>
      <div className="grid-2" style={{ marginBottom: '1.5rem' }}>
        {players.map(([pid, p]) => {
          const idx = Number(pid)
          const name = playerNames?.[idx] ?? `Giocatore ${idx + 1}`
          const totalShots = Object.values(p.shots).reduce((s, v) => s + v, 0)
          return (
            <div key={pid} className="card">
              <div className="player-header">
                <span className="player-dot" style={{ background: PLAYER_COLORS[idx % 4] }} />
                <h3>{name}</h3>
              </div>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '.9rem' }}>
                <tbody>
                  {[
                    ['Distanza percorsa', `${p.distance_m.toFixed(0)} m`],
                    ['Colpi totali', totalShots],
                    ['Vincenti', p.winners],
                    ['Errori', p.errors],
                    ['Smash', p.shots.smash],
                    ['Volée', p.shots.volley],
                    ['Bandeja', p.shots.bandeja],
                    ['Altri', p.shots.other],
                  ].map(([label, val]) => (
                    <tr key={String(label)} style={{ borderBottom: '1px solid var(--border)' }}>
                      <td style={{ padding: '.4rem 0', color: 'var(--muted)' }}>{label}</td>
                      <td style={{ padding: '.4rem 0', fontWeight: 600, textAlign: 'right' }}>{val}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )
        })}
      </div>

      <div style={{ textAlign: 'center' }}>
        <button className="btn btn-primary" onClick={onNewAnalysis}>Analizza un'altra partita</button>
      </div>
    </div>
  )
}
