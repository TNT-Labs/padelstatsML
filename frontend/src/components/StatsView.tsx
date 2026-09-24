import { useCallback, useRef, type FC } from 'react'
import type { MatchStats } from '../api'
import { CourtHeatmap } from './CourtHeatmap'
import { DataQualityPanel } from './DataQualityPanel'
import { MatchVideo } from './MatchVideo'
import { ZoneChart } from './ZoneChart'
import { PlayerThumb } from './PlayerThumb'
import { kmh, playerPlace } from '../lib/format'
import { clock } from '../lib/videoOverlay'
import { PLAYER_COLORS } from '../lib/players'

const RALLY_LEAD_S = 2

interface Props {
  stats: MatchStats
  onBack: () => void
  onRename: () => void
  onReanalyse: () => void
}

function formatDuration(seconds: number): string {
  const total = Math.round(seconds)
  const minutes = Math.floor(total / 60)
  const rest = total % 60
  return minutes > 0 ? `${minutes}m ${rest}s` : `${rest}s`
}

export const StatsView: FC<Props> = ({ stats, onBack, onRename, onReanalyse }) => {
  const players = Object.entries(stats.per_player).sort(([a], [b]) => Number(a) - Number(b))
  const names = stats.player_names ?? []
  const summary = stats.summary
  const videoRef = useRef<HTMLVideoElement>(null)

  const seek = useCallback((seconds: number) => {
    const video = videoRef.current
    if (!video) return
    video.currentTime = seconds
    video.scrollIntoView({ behavior: 'smooth', block: 'center' })
    void video.play().catch(() => undefined)
  }, [])

  return (
    <div className="layout">
      <div className="header">
        <h1>{stats.title}</h1>
        <div style={{ display: 'flex', gap: '.5rem' }}>
          <button className="btn btn-ghost btn-sm" onClick={onRename}>
            Rinomina giocatori
          </button>
          <button
            className="btn btn-ghost btn-sm"
            onClick={() => {
              if (
                window.confirm(
                  'Rianalizzare la partita? Le statistiche attuali verranno sostituite e i ' +
                    'nomi dei giocatori andranno reinseriti. Richiede circa quanto la durata del video.',
                )
              ) {
                onReanalyse()
              }
            }}
          >
            Rianalizza
          </button>
          <button className="btn btn-ghost btn-sm" onClick={onBack}>
            ← Partite
          </button>
        </div>
      </div>

      <div className="grid-4" style={{ marginBottom: '1.25rem' }}>
        <div className="card card-sm stat-card">
          <div className="val">{summary.rallies_count}</div>
          <div className="lbl">Scambi</div>
        </div>
        <div className="card card-sm stat-card">
          <div className="val">{formatDuration(summary.avg_rally_s)}</div>
          <div className="lbl">Scambio medio</div>
        </div>
        <div className="card card-sm stat-card">
          <div className="val">{formatDuration(summary.longest_rally_s)}</div>
          <div className="lbl">Scambio più lungo</div>
        </div>
        <div className="card card-sm stat-card">
          <div className="val">{Math.round(summary.active_ratio * 100)}%</div>
          <div className="lbl">Tempo di gioco effettivo</div>
        </div>
      </div>

      <div style={{ marginBottom: '1.25rem' }}>
        <DataQualityPanel quality={stats.data_quality} matchId={stats.match_id} />
      </div>

      <MatchVideo matchId={stats.match_id} names={names} videoRef={videoRef} onSeek={seek} />

      <div className="grid-2" style={{ marginBottom: '1.5rem' }}>
        <div className="card">
          <h2 style={{ marginBottom: '1rem' }}>Occupazione del campo</h2>
          <div style={{ display: 'flex', justifyContent: 'center' }}>
            <CourtHeatmap heatmaps={stats.heatmaps} playerNames={names} width={240} />
          </div>
        </div>
        <div className="card">
          <h2 style={{ marginBottom: '1rem' }}>Posizione in campo</h2>
          <ZoneChart perPlayer={stats.per_player} playerNames={names} />
          <p className="muted-note" style={{ marginTop: '.75rem' }}>
            Quota di tempo entro 3 m dalla rete, fra 3 e 6 m, e oltre 6 m.
          </p>
        </div>
      </div>

      <h2 style={{ marginBottom: '.75rem' }}>Giocatori</h2>
      <div className="grid-2" style={{ marginBottom: '1.5rem' }}>
        {players.map(([pid, player]) => {
          const index = Number(pid)
          const name = names[index] || `Giocatore ${index + 1}`
          const partial = player.tracked_ratio < 0.5

          return (
            <div key={pid} className="card">
              <div className="player-header">
                <PlayerThumb url={player.crop_url} alt={name} className="player-thumb" />
                <div style={{ minWidth: 0 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '.4rem' }}>
                    <span className="player-dot" style={{ background: PLAYER_COLORS[index % 4] }} />
                    <h3 style={{ margin: 0 }}>{name}</h3>
                  </div>
                  <div className="muted-note">{playerPlace(player.team, player.role)}</div>
                </div>
              </div>

              {partial && (
                <div className="callout callout-warn" style={{ marginBottom: '.75rem' }}>
                  Seguito solo per il {Math.round(player.tracked_ratio * 100)}% della partita:
                  i valori qui sotto sono sottostimati.
                </div>
              )}

              <table className="stat-table">
                <tbody>
                  <tr>
                    <td>Distanza percorsa</td>
                    <td>{player.distance_m.toFixed(0)} m</td>
                  </tr>
                  <tr>
                    <td>Di cui negli scambi</td>
                    <td>{player.distance_rally_m.toFixed(0)} m</td>
                  </tr>
                  <tr>
                    <td>Velocità media negli scambi</td>
                    <td>{kmh(player.avg_speed_ms)}</td>
                  </tr>
                  <tr>
                    <td>Velocità di punta</td>
                    <td>{kmh(player.peak_speed_ms)}</td>
                  </tr>
                  <tr>
                    <td>Area di campo coperta</td>
                    <td>{player.coverage_m2.toFixed(0)} m²</td>
                  </tr>
                  <tr>
                    <td>Copertura del tracciamento</td>
                    <td>{Math.round(player.tracked_ratio * 100)}%</td>
                  </tr>
                </tbody>
              </table>

              <div style={{ display: 'flex', justifyContent: 'center', marginTop: '.75rem' }}>
                <CourtHeatmap heatmaps={stats.heatmaps} only={pid} width={140} />
              </div>
            </div>
          )
        })}
      </div>

      {stats.rallies.length > 0 && (
        <div className="card" style={{ marginBottom: '1.5rem' }}>
          <h2 style={{ marginBottom: '.75rem' }}>Scambi</h2>
          <div className="rally-strip">
            {stats.rallies.map(rally => (
              <button
                key={rally.index}
                type="button"
                className="rally-chip"
                // A couple of seconds early, to see the serve.
                onClick={() => seek(Math.max(0, rally.start_s - RALLY_LEAD_S))}
                title={`Guarda lo scambio, dal minuto ${clock(rally.start_s)}`}
              >
                <span className="rally-num">#{rally.index + 1}</span>
                <span className="rally-dur">{formatDuration(rally.duration_s)}</span>
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
