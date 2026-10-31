import { useCallback, useEffect, useMemo, useRef, useState, type FC } from 'react'
import type { MatchStats } from '../api'
import { CourtHeatmap } from './CourtHeatmap'
import { DataQualityPanel } from './DataQualityPanel'
import { InfoHeading } from './InfoHeading'
import { MatchVideo } from './MatchVideo'
import { ZoneChart } from './ZoneChart'
import { PlayerThumb } from './PlayerThumb'
import { StrokesCard } from './StrokesCard'
import { analysisTimeText, handLabel, kmh, playerPlace } from '../lib/format'
import { clock } from '../lib/videoOverlay'
import { PLAYER_COLORS } from '../lib/players'

const RALLY_LEAD_S = 2

interface Props {
  stats: MatchStats
  /** Read-only account: no renaming, no re-analysis. */
  readOnly?: boolean
  onBack: () => void
  onRename: () => void
  onVerify: () => void
  onAnnotateBall: () => void
  /** `recalibrate`: place the court corners again before analysing. */
  onReanalyse: (recalibrate: boolean) => void | Promise<void>
  /** Recompute from the stored detections; rejects with the reason. */
  onRefresh: () => Promise<void>
}

function formatDuration(seconds: number): string {
  const total = Math.round(seconds)
  const minutes = Math.floor(total / 60)
  const rest = total % 60
  return minutes > 0 ? `${minutes}m ${rest}s` : `${rest}s`
}

export const StatsView: FC<Props> = ({
  stats, readOnly = false, onBack, onRename, onVerify, onAnnotateBall, onReanalyse, onRefresh,
}) => {
  const players = Object.entries(stats.per_player).sort(([a], [b]) => Number(a) - Number(b))
  const names = stats.player_names ?? []
  const summary = stats.summary
  // Analyses older than the serve check carry no flag: no legend for them.
  const serveChecked = stats.rallies.some(r => r.serve != null)
  // By position in `players`: the high strokes, classified by the server
  // with the racket hand; null without the posture.
  const playerStrokes = players.map(([, player]) => player.strokes ?? null)
  const anyStrokes = playerStrokes.some(s => s !== null)
  const served = stats.rallies.filter(r => r.serve).length
  // Hits come from the ball: only analyses that followed it have them.
  const withHits = stats.rallies.filter(r => r.hits != null)
  const hitsPerRally = withHits.length
    ? withHits.reduce((sum, r) => sum + (r.hits?.length ?? 0), 0) / withHits.length
    : null
  const allHits = useMemo(
    () => stats.rallies.flatMap(r => r.hits ?? []).sort((a, b) => a[0] - b[0]),
    [stats.rallies],
  )
  const videoRef = useRef<HTMLVideoElement>(null)
  const [asking, setAsking] = useState(false)
  // Opening the calibration first reads the match from the Pi: seconds when
  // it is busy analysing. The dialog stays, saying so, until the page opens.
  const [opening, setOpening] = useState(false)
  const [refreshing, setRefreshing] = useState(false)
  const [refreshError, setRefreshError] = useState<string | null>(null)

  const refresh = async () => {
    setRefreshing(true)
    setRefreshError(null)
    try {
      await onRefresh()
    } catch (e) {
      setRefreshError((e as Error).message)
    } finally {
      setRefreshing(false)
    }
  }

  useEffect(() => {
    if (!asking) return
    const escape = (e: KeyboardEvent) => e.key === 'Escape' && !opening && setAsking(false)
    document.addEventListener('keydown', escape)
    return () => document.removeEventListener('keydown', escape)
  }, [asking, opening])

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
        <div style={{ display: 'flex', gap: '.5rem', flexWrap: 'wrap' }}>
          {!readOnly && (
            <>
              <button
                className="btn btn-ghost btn-sm"
                onClick={refresh}
                disabled={refreshing}
                title="Ricalcola statistiche e scambi dalle rilevazioni già fatte, senza rianalizzare il video"
              >
                {refreshing ? 'Aggiornamento…' : 'Aggiorna statistiche'}
              </button>
              <button className="btn btn-ghost btn-sm" onClick={onRename}>
                Rinomina giocatori
              </button>
              <button className="btn btn-ghost btn-sm" onClick={onVerify}>
                Verifica accuratezza
              </button>
              <button
                className="btn btn-ghost btn-sm"
                onClick={onAnnotateBall}
                title="Segna la palla fotogramma per fotogramma: i dati per insegnare all'app a seguirla"
              >
                Annota palla
              </button>
              <button className="btn btn-ghost btn-sm" onClick={() => setAsking(true)}>
                Rianalizza
              </button>
            </>
          )}
          <button className="btn btn-ghost btn-sm" onClick={onBack}>
            ← Partite
          </button>
        </div>
      </div>

      {refreshError && (
        <div className="callout callout-error" role="alert" style={{ marginBottom: '1rem' }}>
          {refreshError}
        </div>
      )}

      {asking && (
        <div className="modal-backdrop" onClick={() => !opening && setAsking(false)}>
          <div
            className="modal card"
            role="dialog"
            aria-modal="true"
            aria-labelledby="reanalyse-title"
            onClick={e => e.stopPropagation()}
          >
            <h2 id="reanalyse-title">Rianalizzare la partita?</h2>
            <p className="muted-note">
              Le statistiche attuali verranno sostituite e i nomi dei giocatori andranno
              reinseriti. Richiede circa quanto la durata del video.
            </p>
            <p style={{ margin: '.75rem 0 1rem' }}>
              Vuoi prima <strong>ricalibrare il campo</strong>? Fallo se rete e linee proiettate
              non coincidevano con quelle reali: la calibrazione salvata viene riproposta da
              correggere.
            </p>
            <div className="modal-actions">
              <button className="btn btn-ghost btn-sm" autoFocus disabled={opening} onClick={() => setAsking(false)}>
                Annulla
              </button>
              <button
                className="btn btn-ghost btn-sm"
                disabled={opening}
                onClick={() => {
                  setAsking(false)
                  onReanalyse(false)
                }}
              >
                Usa la calibrazione attuale
              </button>
              <button
                className="btn btn-primary btn-sm"
                disabled={opening}
                onClick={async () => {
                  setOpening(true)
                  try {
                    await onReanalyse(true)
                  } finally {
                    setOpening(false)
                    setAsking(false)
                  }
                }}
              >
                {opening ? 'Apertura della calibrazione…' : 'Ricalibra il campo'}
              </button>
            </div>
          </div>
        </div>
      )}

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
        {stats.analysis_time && (
          <p className="muted-note" style={{ margin: '0 0 .5rem' }}>
            {analysisTimeText(stats.analysis_time.seconds, summary.analysed_s, stats.analysis_time.finished_at)}
          </p>
        )}
        <DataQualityPanel quality={stats.data_quality} matchId={stats.match_id} />
      </div>

      <MatchVideo matchId={stats.match_id} names={names} videoRef={videoRef} onSeek={seek} hits={allHits} />

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

      <InfoHeading title="Giocatori">
        <dl className="info-list">
          <dt>Distanza percorsa</dt>
          <dd>
            I metri fatti in campo in tutta la parte analizzata, pause fra i punti comprese. I
            movimenti troppo piccoli per essere distinti dal tremolio dell'immagine non contano.
          </dd>
          <dt>Di cui negli scambi</dt>
          <dd>
            La parte della distanza fatta mentre si giocava un punto; il resto è il movimento fra
            un punto e l'altro (palline, cambi di campo).
          </dd>
          <dt>Velocità media negli scambi</dt>
          <dd>
            Metri fatti negli scambi diviso il tempo degli scambi. È bassa perché il padel
            alterna scatti brevi a momenti quasi fermi.
          </dd>
          <dt>Velocità di punta</dt>
          <dd>
            La velocità superata solo nel 5% del tempo: non il picco assoluto, che può essere un
            errore di misura. Gli scatti più rapidi possono essere un po' più veloci.
          </dd>
          <dt>Area di campo coperta</dt>
          <dd>
            Il campo è diviso in quadrati di 50 × 50 cm: si contano quelli in cui il giocatore è
            rimasto almeno 1 secondo in tutto. Una metà campo misura 100 m².
          </dd>
          <dt>Copertura del tracciamento</dt>
          <dd>
            Per quanta parte del tempo il giocatore è stato visto e seguito. Nel resto era
            coperto, fuori inquadratura o non riconosciuto: le distanze sono sottostimate di
            altrettanto. Sotto il 50% compare un avviso.
          </dd>
          <dt>Smash e bandeja</dt>
          <dd>
            Dallo scheletro del giocatore (solo con l'acceleratore): il braccio della racchetta e il
            braccio libero entrambi alzati, entro 6,5 m dalla rete. Verificati colpo per colpo su una
            partita: giusti 29 su 30, trovati 29 su 34. Serve la mano indicata in «Rinomina
            giocatori». L'elenco è nella sezione Colpi alti.
          </dd>
          <dt>Mappa del giocatore</dt>
          <dd>Dove è stato in campo: più il colore è intenso, più tempo ci ha passato.</dd>
        </dl>
      </InfoHeading>
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
                  <div className="muted-note">
                    {[playerPlace(player.team, player.role), handLabel(stats.player_hands?.[index])]
                      .filter(Boolean)
                      .join(' · ')}
                  </div>
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
                  {player.hits != null && (
                    <tr>
                      <td>Colpi (dalla palla)</td>
                      <td>{player.hits}</td>
                    </tr>
                  )}
                  {playerStrokes[index] && (
                    <tr>
                      <td>Smash e bandeja</td>
                      <td>{playerStrokes[index]!.filter(s => s.kind === 'overhead').length}</td>
                    </tr>
                  )}
                </tbody>
              </table>

              <div style={{ display: 'flex', justifyContent: 'center', marginTop: '.75rem' }}>
                <CourtHeatmap heatmaps={stats.heatmaps} only={pid} width={140} />
              </div>
            </div>
          )
        })}
      </div>

      {anyStrokes && (
        <StrokesCard
          // Fresh statistics (a refresh) bring their own labels.
          key={`high:${stats.match_id}:${JSON.stringify(stats.stroke_labels ?? {})}`}
          title="Colpi alti"
          info={
            <>
              Dallo scheletro dei giocatori, con la mano della racchetta. <strong>Smash e bandeja</strong>:
              entro 6,5 m dalla rete, la racchetta e il braccio libero entrambi alzati, come per puntare
              la palla; verificati colpo per colpo su una partita, giusti 29 su 30 e trovati 29 su 34.
              <strong> Lob</strong>: la racchetta alta più indietro, quasi fermi; giusti 7 su 8, ma su
              pochi casi, per questo non è contato nelle schede. Con «Verifica» indichi cosa è ciascuno
              nel video: la precisione si misura qui sotto e le verifiche servono a migliorarlo.
            </>
          }
          stats={stats}
          strokes={playerStrokes}
          players={players.map(([pid]) => pid)}
          names={names}
          readOnly={readOnly}
          onSeek={seek}
        />
      )}

      {stats.rallies.length > 0 && (
        <div className="card" style={{ marginBottom: '1.5rem' }}>
          <InfoHeading title="Scambi">
            <dl className="info-list">
              <dt>Come si riconosce uno scambio</dt>
              <dd>
                Dal movimento dei giocatori, senza seguire la palla: durante un punto tutti si
                muovono, fra un punto e l'altro no. Un nuovo scambio inizia anche dalla battuta,
                riconosciuta dai giocatori fermi in posizione e dal gioco che riparte.
              </dd>
              <dt>Scambi</dt>
              <dd>Quanti punti giocati sono stati rilevati.</dd>
              <dt>Scambio medio e più lungo</dt>
              <dd>La durata media degli scambi e quella del più lungo.</dd>
              <dt>Tempo di gioco effettivo</dt>
              <dd>
                La somma della durata degli scambi sulla durata analizzata: il resto sono pause,
                cambi di campo, palline da raccogliere.
              </dd>
              {hitsPerRally != null && (
                <>
                  <dt>Colpi</dt>
                  <dd>
                    Dalla palla: un colpo è dove la sua traiettoria cambia di colpo vicino a un
                    giocatore, che è chi colpisce. Nel video, con Palla attivo, ogni colpo compare
                    sopra il giocatore. Sperimentale: da verificare guardando il video.
                  </dd>
                </>
              )}
              <dt>Punto verde</dt>
              <dd>
                Lo scambio è iniziato da una battuta riconosciuta. Uno scambio molto lungo senza
                punto verde può contenere più punti uniti.
              </dd>
              <dt>Clic su uno scambio</dt>
              <dd>Il video va all'inizio dello scambio, un paio di secondi prima della battuta.</dd>
            </dl>
          </InfoHeading>
          {hitsPerRally != null && (
            <p className="muted-note rally-legend">
              In media {hitsPerRally.toLocaleString('it-IT', { maximumFractionDigits: 1 })} colpi per
              scambio, visti dalla palla
            </p>
          )}
          {serveChecked && (
            <p className="muted-note rally-legend">
              <span className="rally-serve" aria-hidden="true" /> {served} su {stats.rallies.length} aperti
              da una battuta riconosciuta
            </p>
          )}
          <div className="rally-strip">
            {stats.rallies.map(rally => (
              <button
                key={rally.index}
                type="button"
                className="rally-chip"
                // A couple of seconds early, to see the serve.
                onClick={() => seek(Math.max(0, rally.start_s - RALLY_LEAD_S))}
                title={`Guarda lo scambio, dal minuto ${clock(rally.start_s)}${
                  rally.serve ? ' · aperto da una battuta' : ''}`}
              >
                <span className="rally-num">
                  #{rally.index + 1}
                  {rally.serve && <span className="rally-serve" aria-label="aperto da una battuta" />}
                </span>
                <span className="rally-dur">
                  {formatDuration(rally.duration_s)}
                  {rally.hits != null && ` · ${rally.hits.length} colpi`}
                </span>
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
