/**
 * Court calibration: drag four corners onto the keyframe.
 *
 * This screen is the reason the rest of the analysis can be trusted. Every
 * metric the system reports is in metres, and metres come from this
 * homography. Automatic court detection on a padel court — glass walls, metal
 * mesh, few painted lines — is unreliable enough that the previous version
 * silently fell back to invented corners, so here a human always confirms.
 *
 * Two things make that fast rather than tedious:
 *   1. The court model (net, service lines, outline) is projected live from
 *      the current corners, so a bad corner is visible immediately.
 *   2. A confirmed calibration can be saved as a camera preset and reused in
 *      one click for every future match filmed from the same position.
 */
import { useCallback, useEffect, useMemo, useRef, useState, type FC, type PointerEvent } from 'react'
import { api, ApiError, type CameraPreset, type Match, type Point } from '../api'
import { CORNER_LABELS, checkQuad, netErrorMetres, projectCourtLines, type Pt } from '../lib/court'

interface Props {
  match: Match
  onCalibrated: () => void
  onBack: () => void
}

type Handle = { kind: 'corner' | 'net'; index: number }

const HANDLE_COLORS = ['#f97316', '#22c55e', '#3b82f6', '#a855f7']

export const CalibrationView: FC<Props> = ({ match, onCalibrated, onBack }) => {
  const frameW = match.width ?? 1920
  const frameH = match.height ?? 1080

  const [corners, setCorners] = useState<Pt[] | null>(null)
  const [netPoints, setNetPoints] = useState<Pt[] | null>(null)
  const [presets, setPresets] = useState<CameraPreset[]>([])
  const [presetName, setPresetName] = useState('')
  const [dragging, setDragging] = useState<Handle | null>(null)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [hint, setHint] = useState<string | null>(null)

  const svgRef = useRef<SVGSVGElement>(null)

  useEffect(() => {
    let cancelled = false
    Promise.all([api.getSuggestion(match.id), api.listPresets().catch(() => [])])
      .then(([suggestion, presetList]) => {
        if (cancelled) return
        setCorners(suggestion.corners_px.map(p => [p[0], p[1]] as Pt))
        setPresets(presetList)
        setHint(
          suggestion.method === 'saved'
            ? 'Calibrazione già salvata per questa partita: modificala se serve.'
            : suggestion.method === 'lines'
              ? 'Proposta automatica dalle linee rilevate — controllala e correggila.'
              : 'Nessuna linea riconosciuta: trascina i 4 angoli sul campo.',
        )
      })
      .catch((e: Error) => !cancelled && setError(e.message))
    return () => {
      cancelled = true
    }
  }, [match.id])

  const check = useMemo(
    () => (corners ? checkQuad(corners, frameW, frameH) : { valid: false }),
    [corners, frameW, frameH],
  )
  const courtLines = useMemo(
    () => (corners && check.valid ? projectCourtLines(corners) : []),
    [corners, check.valid],
  )
  const netError = useMemo(
    () => (corners && netPoints && check.valid ? netErrorMetres(corners, netPoints) : null),
    [corners, netPoints, check.valid],
  )

  /** Convert a pointer event to image-native pixel coordinates. */
  const toImagePoint = useCallback(
    (event: PointerEvent): Pt | null => {
      const svg = svgRef.current
      if (!svg) return null
      const rect = svg.getBoundingClientRect()
      if (rect.width === 0 || rect.height === 0) return null
      const x = ((event.clientX - rect.left) / rect.width) * frameW
      const y = ((event.clientY - rect.top) / rect.height) * frameH
      return [
        Math.max(0, Math.min(frameW, x)),
        Math.max(0, Math.min(frameH, y)),
      ]
    },
    [frameW, frameH],
  )

  const onPointerDown = (handle: Handle) => (event: PointerEvent<SVGGElement>) => {
    event.preventDefault()
    event.currentTarget.setPointerCapture(event.pointerId)
    setDragging(handle)
  }

  const onPointerMove = (event: PointerEvent<SVGGElement>) => {
    if (!dragging) return
    const point = toImagePoint(event)
    if (!point) return
    if (dragging.kind === 'corner') {
      setCorners(prev => prev && prev.map((p, i) => (i === dragging.index ? point : p)))
    } else {
      setNetPoints(prev => prev && prev.map((p, i) => (i === dragging.index ? point : p)))
    }
  }

  const endDrag = (event: PointerEvent<SVGGElement>) => {
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId)
    }
    setDragging(null)
  }

  const applyPreset = async (preset: CameraPreset) => {
    setError(null)
    setSaving(true)
    try {
      const result = await api.submitCalibration(match.id, { preset_id: preset.id })
      setCorners(result.corners_px.map(p => [p[0], p[1]] as Pt))
      setHint(result.message ?? `Preset "${preset.name}" applicato.`)
      onCalibrated()
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e))
    } finally {
      setSaving(false)
    }
  }

  const save = async () => {
    if (!corners || !check.valid) return
    setError(null)
    setSaving(true)
    try {
      await api.submitCalibration(match.id, {
        corners_px: corners.map(p => [p[0], p[1]] as Point),
        net_px: netPoints ? netPoints.map(p => [p[0], p[1]] as Point) : null,
        save_as_preset: presetName.trim() || undefined,
      })
      onCalibrated()
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e))
      setSaving(false)
    }
  }

  const toggleNet = () => {
    setNetPoints(prev =>
      prev
        ? null
        : [
            [frameW * 0.2, frameH * 0.55],
            [frameW * 0.8, frameH * 0.55],
          ],
    )
  }

  if (!corners) {
    return (
      <div className="layout">
        <div className="header">
          <h1>Calibrazione campo</h1>
          <button className="btn btn-ghost btn-sm" onClick={onBack}>← Indietro</button>
        </div>
        <div className="card">
          <p style={{ color: 'var(--muted)' }}>{error ?? 'Caricamento del frame…'}</p>
        </div>
      </div>
    )
  }

  const handleRadius = Math.max(frameW, frameH) * 0.012

  return (
    <div className="layout">
      <div className="header">
        <h1>Calibrazione campo</h1>
        <button className="btn btn-ghost btn-sm" onClick={onBack}>← Indietro</button>
      </div>

      <div className="callout callout-info" style={{ marginBottom: '1rem' }}>
        <strong>Trascina i 4 angoli sugli spigoli del campo</strong>, partendo dal fondo a
        sinistra e procedendo in senso orario. Le linee bianche sovrapposte mostrano dove
        il sistema calcola rete e linee di servizio: se coincidono con quelle reali, la
        calibrazione è corretta.
      </div>

      {presets.length > 0 && (
        <div className="card card-sm" style={{ marginBottom: '1rem' }}>
          <div className="preset-row">
            <span style={{ fontWeight: 600 }}>Posizione camera salvata:</span>
            {presets.map(preset => (
              <button
                key={preset.id}
                className="btn btn-ghost btn-sm"
                disabled={saving}
                onClick={() => applyPreset(preset)}
              >
                {preset.name}
              </button>
            ))}
          </div>
        </div>
      )}

      <div className="card" style={{ marginBottom: '1rem', padding: '.75rem' }}>
        <svg
          ref={svgRef}
          viewBox={`0 0 ${frameW} ${frameH}`}
          className="calib-canvas"
          onPointerMove={onPointerMove}
          onPointerUp={endDrag}
          onPointerCancel={endDrag}
        >
          <image href={api.keyframeUrl(match.id)} x={0} y={0} width={frameW} height={frameH} />

          <polygon
            points={corners.map(p => p.join(',')).join(' ')}
            fill={check.valid ? 'rgba(34,197,94,0.12)' : 'rgba(239,68,68,0.15)'}
            stroke={check.valid ? '#22c55e' : '#ef4444'}
            strokeWidth={Math.max(frameW, frameH) * 0.0025}
          />

          {courtLines.map((line, i) => (
            <polyline
              key={i}
              points={line.map(p => p.join(',')).join(' ')}
              fill="none"
              stroke="#ffffff"
              strokeOpacity={i === 1 ? 0.95 : 0.55}
              strokeWidth={Math.max(frameW, frameH) * (i === 1 ? 0.003 : 0.0018)}
              strokeDasharray={i === 0 ? undefined : '10 8'}
            />
          ))}

          {netPoints && (
            <polyline
              points={netPoints.map(p => p.join(',')).join(' ')}
              fill="none"
              stroke="#facc15"
              strokeWidth={Math.max(frameW, frameH) * 0.003}
            />
          )}

          {corners.map((point, index) => (
            <g
              key={`c${index}`}
              onPointerDown={onPointerDown({ kind: 'corner', index })}
              onPointerMove={onPointerMove}
              onPointerUp={endDrag}
              style={{ cursor: 'grab', touchAction: 'none' }}
            >
              <circle
                cx={point[0]}
                cy={point[1]}
                r={handleRadius}
                fill={HANDLE_COLORS[index]}
                stroke="#fff"
                strokeWidth={handleRadius * 0.22}
              />
              <text
                x={point[0]}
                y={point[1] - handleRadius * 1.6}
                textAnchor="middle"
                fontSize={handleRadius * 1.5}
                fill="#fff"
                stroke="#000"
                strokeWidth={handleRadius * 0.08}
                paintOrder="stroke"
              >
                {index + 1}. {CORNER_LABELS[index]}
              </text>
            </g>
          ))}

          {netPoints?.map((point, index) => (
            <g
              key={`n${index}`}
              onPointerDown={onPointerDown({ kind: 'net', index })}
              onPointerMove={onPointerMove}
              onPointerUp={endDrag}
              style={{ cursor: 'grab', touchAction: 'none' }}
            >
              <rect
                x={point[0] - handleRadius}
                y={point[1] - handleRadius}
                width={handleRadius * 2}
                height={handleRadius * 2}
                fill="#facc15"
                stroke="#fff"
                strokeWidth={handleRadius * 0.22}
              />
            </g>
          ))}
        </svg>
      </div>

      {hint && !error && <p className="muted-note">{hint}</p>}
      {!check.valid && check.reason && <p className="error-msg">{check.reason}</p>}
      {error && <p className="error-msg">{error}</p>}

      {netPoints && (
        <p className={netError !== null && netError > 0.5 ? 'error-msg' : 'muted-note'}>
          {netError === null
            ? 'Rete non verificabile con questi angoli.'
            : `Scarto della rete indicata: ${netError.toFixed(2)} m` +
              (netError > 0.5 ? ' — gli angoli sono probabilmente imprecisi.' : ' — ottimo.')}
        </p>
      )}

      <div className="card" style={{ marginTop: '1rem' }}>
        <div className="form-group">
          <label>Salva questa posizione camera (opzionale)</label>
          <input
            type="text"
            value={presetName}
            placeholder="es. Campo 2 — tripode angolo nord"
            onChange={e => setPresetName(e.target.value)}
          />
          <small className="muted-note">
            Riutilizzabile con un clic per ogni partita ripresa dalla stessa posizione.
          </small>
        </div>

        <div className="action-row">
          <button className="btn btn-ghost btn-sm" onClick={toggleNet}>
            {netPoints ? 'Rimuovi verifica rete' : 'Verifica con la rete'}
          </button>
          <button
            className="btn btn-primary"
            disabled={!check.valid || saving}
            onClick={save}
          >
            {saving ? 'Salvataggio…' : 'Conferma calibrazione'}
          </button>
        </div>
      </div>
    </div>
  )
}
