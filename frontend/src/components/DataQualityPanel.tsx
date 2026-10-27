/**
 * Reliability of the numbers on screen, stated up front.
 *
 * The previous version reported shot types, winners and errors with no
 * indication that they were guesses. Here every result carries how it was
 * obtained, what was excluded, and anything the pipeline flagged — so a
 * partially-tracked match reads as a partially-tracked match rather than as
 * a player who barely moved.
 */
import { useState, type FC } from 'react'
import type { DataQuality } from '../api'
import { clock } from '../lib/videoOverlay'

const SOURCE_LABEL: Record<string, string> = {
  manual: 'angoli indicati manualmente',
  preset: 'posizione camera salvata',
  auto: 'rilevamento automatico confermato',
}

interface Props {
  quality: DataQuality
  /** Serve per gli strumenti diagnostici da riga di comando, che lo chiedono
   *  come argomento. Senza mostrarlo qui, l'unico modo di conoscerlo è
   *  interrogare l'API: l'app non ha rotte e l'URL non lo contiene. */
  matchId?: string
}

export const DataQualityPanel: FC<Props> = ({ quality, matchId }) => {
  const [open, setOpen] = useState(quality.warnings.length > 0)
  const hasWarnings = quality.warnings.length > 0

  return (
    <div className={`card quality-card ${hasWarnings ? 'quality-warn' : 'quality-ok'}`}>
      <button className="quality-header" onClick={() => setOpen(o => !o)}>
        <span>
          {hasWarnings ? '⚠️' : '✅'}{' '}
          <strong>
            {hasWarnings
              ? `Affidabilità dei dati — ${quality.warnings.length} nota${quality.warnings.length > 1 ? 'e' : ''}`
              : 'Affidabilità dei dati — nessun problema rilevato'}
          </strong>
        </span>
        <span className="quality-toggle">{open ? '−' : '+'}</span>
      </button>

      {open && (
        <div className="quality-body">
          {hasWarnings && (
            <ul className="quality-warnings">
              {quality.warnings.map((warning, i) => (
                <li key={i}>{warning}</li>
              ))}
            </ul>
          )}

          <dl className="quality-grid">
            <div>
              <dt>Calibrazione</dt>
              <dd>
                {SOURCE_LABEL[quality.calibration_source] ?? quality.calibration_source}
                {quality.net_error_m !== null && ` · scarto rete ${quality.net_error_m.toFixed(2)} m`}
              </dd>
            </div>
            {!!quality.analysis_start_s && (
              <div>
                <dt>Analisi</dt>
                <dd>dal minuto {clock(quality.analysis_start_s)} del video</dd>
              </div>
            )}
            {quality.rally_speed_threshold_ms != null && (
              <div>
                <dt>Scambi</dt>
                <dd>in gioco sopra {quality.rally_speed_threshold_ms.toFixed(2)} m/s di velocità media</dd>
              </div>
            )}
            {quality.far_pass && (
              <div>
                <dt>Metà lontana</dt>
                <dd>
                  {quality.far_pass.active
                    ? `secondo passaggio del rilevatore: +${quality.far_pass.added.toLocaleString('it-IT')} rilevazioni`
                    : 'secondo passaggio non applicato: con questa inquadratura non ingrandisce i giocatori'}
                </dd>
              </div>
            )}
            {quality.tracking &&
              (quality.tracking.feet_below_net > 0 || quality.tracking.weak_detections > 0 || !!quality.tracking.scenery) && (
              <div>
                <dt>Tracciamento</dt>
                <dd>
                  {quality.tracking.feet_below_net > 0 &&
                    `${quality.tracking.feet_below_net.toLocaleString('it-IT')} piedi coperti dalla rete stimati`}
                  {quality.tracking.feet_below_net > 0 && quality.tracking.weak_detections > 0 && ' · '}
                  {quality.tracking.weak_detections > 0 &&
                    `${quality.tracking.weak_detections.toLocaleString('it-IT')} rilevazioni incerte usate per non perdere un giocatore`}
                  {!!quality.tracking.scenery &&
                    (quality.tracking.feet_below_net > 0 || quality.tracking.weak_detections > 0) &&
                    ' · '}
                  {!!quality.tracking.scenery &&
                    `${quality.tracking.scenery.toLocaleString('it-IT')} rilevazioni di oggetti fissi (cartelloni, pali) scartate`}
                </dd>
              </div>
            )}
            {quality.verification && (
              <div>
                <dt>Verifica a mano</dt>
                <dd>
                  {quality.verification.calibration.all.count > 0 &&
                    `calibrazione ${quality.verification.calibration.all.mean_m?.toFixed(2)} m`}
                  {quality.verification.calibration.all.count > 0 && quality.verification.players && ' · '}
                  {quality.verification.players?.smoothed.all.count
                    ? `giocatori ${quality.verification.players.smoothed.all.mean_m?.toFixed(2)} m`
                    : ''}
                  {' '}
                  <span className="muted-note">
                    ({new Date(quality.verification.at).toLocaleDateString('it-IT')})
                  </span>
                </dd>
              </div>
            )}
            <div>
              <dt>Campionamento</dt>
              <dd>
                {quality.sample_hz.toFixed(1)} Hz · {quality.frames_sampled.toLocaleString('it-IT')} frame
                {quality.variable_frame_rate && ' · video a frequenza variabile, tempi letti dal video'}
              </dd>
            </div>
            <div>
              <dt>Giocatori</dt>
              <dd>
                {quality.players_found} su 4
                {quality.side_changes > 0 && ` · ${quality.side_changes} cambi campo`}
              </dd>
            </div>
            <div>
              <dt>Modello</dt>
              <dd>{quality.detector_model || '—'}</dd>
            </div>
            {quality.identity_cue && (
              <div>
                <dt>Identità</dt>
                <dd>
                  {quality.identity_cue.cue === 'reid'
                    ? `re-ID · stessa persona ${quality.identity_cue.same?.toFixed(2)} · ` +
                      `persone diverse ${quality.identity_cue.different?.toFixed(2)}`
                    : `colore della divisa${quality.identity_cue.reason ? ` · ${quality.identity_cue.reason}` : ''}`}
                  {quality.identity_cue.method === 'ruoli' && ' · coppia e lato in campo'}
                </dd>
              </div>
            )}
            {matchId && (
              <div>
                <dt>ID partita</dt>
                <dd>
                  <code className="match-id" title="Usalo con gli script diagnostici">
                    {matchId}
                  </code>
                </dd>
              </div>
            )}
          </dl>

          {quality.excluded_metrics.length > 0 && (
            <p className="quality-excluded">
              <strong>Non misurato su questo hardware:</strong>{' '}
              {quality.excluded_metrics.join(', ')}. Richiede il tracciamento della palla,
              fuori portata per un Raspberry Pi 5 senza acceleratore. Preferiamo non
              mostrare questi dati piuttosto che stimarli.
            </p>
          )}
        </div>
      )}
    </div>
  )
}
