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
 *
 * A camera close to the court often leaves one or two corners out of the
 * picture. "Estendi oltre l'inquadratura" adds a margin around the keyframe
 * where those corners can be placed: the user lines up the sides of the
 * polygon with the visible lines, and the projected net and service lines
 * with the real ones. The homography extrapolates exactly, so the court is
 * still measured in full.
 */
import { useCallback, useEffect, useMemo, useRef, useState, type FC, type PointerEvent } from 'react'
import { api, ApiError, type CameraPreset, type Match, type Point } from '../api'
import {
  CORNER_LABELS,
  MAX_OFF_FRAME,
  checkQuad,
  isOffFrame,
  netErrorMetres,
  projectCourtLines,
  type Pt,
} from '../lib/court'

interface Props {
  match: Match
  onCalibrated: () => void
  onBack: () => void
}

type Handle = { kind: 'corner' | 'net'; index: number }

const HANDLE_COLORS = ['#f97316', '#22c55e', '#3b82f6', '#a855f7']

/**
 * Margin around the keyframe in extended mode, as a fraction of the frame on
 * each side. It starts small, so the keyframe stays large, and grows by
 * MARGIN_STEP whenever a point is dropped on its edge.
 */
const DEFAULT_MARGIN = 0.25
const MARGIN_STEP = 0.2

/** How far the farthest point lies outside the frame, as a fraction of it. */
function offFrameExtent(points: Pt[], frameW: number, frameH: number): number {
  let extent = 0
  for (const [x, y] of points) {
    extent = Math.max(extent, -x / frameW, (x - frameW) / frameW, -y / frameH, (y - frameH) / frameH)
  }
  return extent
}

/** Smallest margin that shows every point with room to spare. */
function marginFor(points: Pt[], frameW: number, frameH: number): number {
  const extent = offFrameExtent(points, frameW, frameH)
  return Math.min(MAX_OFF_FRAME, Math.max(DEFAULT_MARGIN, extent + 0.1))
}

// Minimum on-screen sizes, in CSS pixels: on a phone the keyframe is a few
// hundred pixels wide and handles sized on the image alone become too small
// to grab.
const MIN_HANDLE_PX = 9
const MIN_GRAB_PX = 22
// Below this width the full corner names would overlap: numbers only.
const LABELS_MIN_WIDTH_PX = 600

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
  // Fraction of the frame added on every side; 0 = the keyframe only.
  const [margin, setMargin] = useState(0)
  // Rendered width of the canvas, for the minimum on-screen handle size.
  const [renderWidth, setRenderWidth] = useState(0)

  const svgRef = useRef<SVGSVGElement>(null)
  const loaded = corners !== null

  useEffect(() => {
    const svg = svgRef.current
    if (!svg) return
    const observer = new ResizeObserver(() => setRenderWidth(svg.getBoundingClientRect().width))
    observer.observe(svg)
    return () => observer.disconnect()
  }, [loaded])

  useEffect(() => {
    let cancelled = false
    Promise.all([api.getSuggestion(match.id), api.listPresets().catch(() => [])])
      .then(([suggestion, presetList]) => {
        if (cancelled) return
        const initial = suggestion.corners_px.map(p => [p[0], p[1]] as Pt)
        setCorners(initial)
        // A saved calibration may already have corners out of the picture.
        if (initial.some(p => isOffFrame(p, frameW, frameH))) {
          setMargin(marginFor(initial, frameW, frameH))
        }
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
  }, [match.id, frameW, frameH])

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

  // The visible area, in image pixels: the keyframe plus the margin.
  const mx = margin * frameW
  const my = margin * frameH
  const view = { x: -mx, y: -my, w: frameW + 2 * mx, h: frameH + 2 * my }

  /** Convert a pointer event to image-native pixel coordinates. */
  const toImagePoint = useCallback(
    (event: PointerEvent): Pt | null => {
      const svg = svgRef.current
      if (!svg) return null
      const rect = svg.getBoundingClientRect()
      if (rect.width === 0 || rect.height === 0) return null
      const x = -mx + ((event.clientX - rect.left) / rect.width) * (frameW + 2 * mx)
      const y = -my + ((event.clientY - rect.top) / rect.height) * (frameH + 2 * my)
      return [
        Math.max(-mx, Math.min(frameW + mx, x)),
        Math.max(-my, Math.min(frameH + my, y)),
      ]
    },
    [frameW, frameH, mx, my],
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
    // Dropped on the edge of the extended area: the user wants to go further.
    // Read from the event: the last move may not have rendered yet.
    const point = dragging && margin > 0 && margin < MAX_OFF_FRAME ? toImagePoint(event) : null
    if (point && offFrameExtent([point], frameW, frameH) >= margin - 0.01) {
      setMargin(Math.min(MAX_OFF_FRAME, margin + MARGIN_STEP))
    }
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

  /** Switching back to the keyframe only brings off-frame corners to its edge. */
  const toggleExtended = () => {
    if (margin > 0) {
      const clamp = (p: Pt): Pt => [
        Math.max(0, Math.min(frameW, p[0])),
        Math.max(0, Math.min(frameH, p[1])),
      ]
      setCorners(prev => prev && prev.map(clamp))
      setNetPoints(prev => prev && prev.map(clamp))
      setMargin(0)
    } else {
      setMargin(DEFAULT_MARGIN)
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

  // Sized on the visible area, so handles keep their on-screen size when the
  // margin shrinks the keyframe.
  const scale = Math.max(view.w, view.h)
  const unitsPerPx = renderWidth > 0 ? view.w / renderWidth : 0
  const handleRadius = Math.max(scale * 0.012, MIN_HANDLE_PX * unitsPerPx)
  const grabRadius = Math.max(handleRadius * 1.8, MIN_GRAB_PX * unitsPerPx)
  const fullLabels = renderWidth === 0 || renderWidth >= LABELS_MIN_WIDTH_PX
  const offFrame = corners.map(p => isOffFrame(p, frameW, frameH))
  const anyOffFrame = offFrame.some(Boolean)

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
        {margin === 0 && (
          <>
            {' '}
            Se un angolo non è inquadrato, usa <strong>Estendi oltre l'inquadratura</strong>.
          </>
        )}
      </div>

      {margin > 0 && (
        <div className="callout callout-warn" style={{ marginBottom: '1rem' }}>
          <strong>Angoli fuori inquadratura:</strong> porta l'angolo nella zona scura dove si
          incontrerebbero le linee del campo. Allinea i lati del perimetro alle linee laterali e
          di fondo visibili, poi controlla che rete e linee di servizio proiettate coincidano con
          quelle reali. Serve più spazio? Lascia l'angolo sul bordo della zona scura: si
          allarga. Con angoli fuori campo la <strong>verifica con la rete</strong> è consigliata.
        </div>
      )}

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
          viewBox={`${view.x} ${view.y} ${view.w} ${view.h}`}
          className="calib-canvas"
          onPointerMove={onPointerMove}
          onPointerUp={endDrag}
          onPointerCancel={endDrag}
        >
          {margin > 0 && (
            <>
              <defs>
                <pattern
                  id="calib-off-frame"
                  width={scale * 0.02}
                  height={scale * 0.02}
                  patternUnits="userSpaceOnUse"
                  patternTransform="rotate(45)"
                >
                  <line
                    x1={0}
                    y1={0}
                    x2={0}
                    y2={scale * 0.02}
                    stroke="#334155"
                    strokeWidth={scale * 0.004}
                  />
                </pattern>
              </defs>
              <rect x={view.x} y={view.y} width={view.w} height={view.h} fill="#0f172a" />
              <rect x={view.x} y={view.y} width={view.w} height={view.h} fill="url(#calib-off-frame)" />
            </>
          )}
          <image href={api.keyframeUrl(match.id)} x={0} y={0} width={frameW} height={frameH} />
          {margin > 0 && (
            <rect
              x={0}
              y={0}
              width={frameW}
              height={frameH}
              fill="none"
              stroke="#e2e8f0"
              strokeOpacity={0.8}
              strokeWidth={scale * 0.0018}
              strokeDasharray={`${scale * 0.008} ${scale * 0.006}`}
              pointerEvents="none"
            />
          )}

          <polygon
            points={corners.map(p => p.join(',')).join(' ')}
            fill={check.valid ? 'rgba(34,197,94,0.12)' : 'rgba(239,68,68,0.15)'}
            stroke={check.valid ? '#22c55e' : '#ef4444'}
            strokeWidth={scale * 0.0025}
          />

          {courtLines.map((line, i) => (
            <polyline
              key={i}
              points={line.map(p => p.join(',')).join(' ')}
              fill="none"
              stroke="#ffffff"
              strokeOpacity={i === 1 ? 0.95 : 0.55}
              strokeWidth={scale * (i === 1 ? 0.003 : 0.0018)}
              strokeDasharray={i === 0 ? undefined : '10 8'}
            />
          ))}

          {netPoints && (
            <polyline
              points={netPoints.map(p => p.join(',')).join(' ')}
              fill="none"
              stroke="#facc15"
              strokeWidth={scale * 0.003}
            />
          )}

          {corners.map((point, index) => {
            // Label below the handle when there is no room above it.
            const labelBelow = point[1] - handleRadius * 3.2 < view.y
            // Anchored inwards when it would overflow a side (≈0.55 em per character).
            const label = `${index + 1}. ${CORNER_LABELS[index]}${offFrame[index] ? ' (fuori)' : ''}`
            const halfWidth = label.length * handleRadius * 1.5 * 0.28
            const anchor =
              point[0] - halfWidth < view.x
                ? 'start'
                : point[0] + halfWidth > view.x + view.w
                  ? 'end'
                  : 'middle'
            const labelX =
              anchor === 'start'
                ? Math.max(view.x + handleRadius * 0.3, point[0] - handleRadius)
                : anchor === 'end'
                  ? Math.min(view.x + view.w - handleRadius * 0.3, point[0] + handleRadius)
                  : point[0]
            return (
              <g
                key={`c${index}`}
                onPointerDown={onPointerDown({ kind: 'corner', index })}
                onPointerMove={onPointerMove}
                onPointerUp={endDrag}
                style={{ cursor: 'grab', touchAction: 'none' }}
              >
                {/* Invisible, larger target: easy to grab with a finger. */}
                <circle cx={point[0]} cy={point[1]} r={grabRadius} fill="transparent" />
                {offFrame[index] && (
                  <circle
                    cx={point[0]}
                    cy={point[1]}
                    r={handleRadius * 1.7}
                    fill="none"
                    stroke={HANDLE_COLORS[index]}
                    strokeWidth={handleRadius * 0.18}
                    strokeDasharray={`${handleRadius * 0.5} ${handleRadius * 0.35}`}
                  />
                )}
                <circle
                  cx={point[0]}
                  cy={point[1]}
                  r={handleRadius}
                  fill={HANDLE_COLORS[index]}
                  stroke="#fff"
                  strokeWidth={handleRadius * 0.22}
                />
                {fullLabels ? (
                  <text
                    x={labelX}
                    y={labelBelow ? point[1] + handleRadius * 2.9 : point[1] - handleRadius * 1.6}
                    textAnchor={anchor}
                    fontSize={handleRadius * 1.5}
                    fill="#fff"
                    stroke="#000"
                    strokeWidth={handleRadius * 0.08}
                    paintOrder="stroke"
                  >
                    {label}
                  </text>
                ) : (
                  <text
                    x={point[0]}
                    y={point[1]}
                    textAnchor="middle"
                    dominantBaseline="central"
                    fontSize={handleRadius * 1.25}
                    fontWeight={700}
                    fill="#fff"
                    pointerEvents="none"
                  >
                    {index + 1}
                  </text>
                )}
              </g>
            )
          })}

          {netPoints?.map((point, index) => (
            <g
              key={`n${index}`}
              onPointerDown={onPointerDown({ kind: 'net', index })}
              onPointerMove={onPointerMove}
              onPointerUp={endDrag}
              style={{ cursor: 'grab', touchAction: 'none' }}
            >
              <circle cx={point[0]} cy={point[1]} r={grabRadius} fill="transparent" />
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
        {!fullLabels && (
          <div className="calib-legend">
            {CORNER_LABELS.map((label, index) => (
              <span key={label}>
                <span className="player-dot" style={{ background: HANDLE_COLORS[index] }} />
                {index + 1}. {label}
                {offFrame[index] ? ' (fuori)' : ''}
              </span>
            ))}
          </div>
        )}
      </div>

      {hint && !error && <p className="muted-note">{hint}</p>}
      {!check.valid && check.reason && <p className="error-msg">{check.reason}</p>}
      {error && <p className="error-msg">{error}</p>}

      {anyOffFrame && !netPoints && (
        <p className="muted-note">
          {offFrame.filter(Boolean).length === 1 ? 'Un angolo è' : 'Alcuni angoli sono'} fuori
          dall'inquadratura: usa «Verifica con la rete» per confermare la precisione.
        </p>
      )}

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
          <button className="btn btn-ghost btn-sm" onClick={toggleExtended}>
            {margin > 0 ? 'Solo inquadratura' : "Estendi oltre l'inquadratura"}
          </button>
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
