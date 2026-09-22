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
            <div>
              <dt>Campionamento</dt>
              <dd>
                {quality.sample_hz.toFixed(1)} Hz · {quality.frames_sampled.toLocaleString('it-IT')} frame
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
