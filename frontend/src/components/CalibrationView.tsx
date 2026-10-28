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
import { parseClock } from '../lib/format'
import { clock } from '../lib/videoOverlay'
import {
  MAX_K1,
  MAX_OFF_FRAME,
  MIN_HANDLES_AREA,
  REFERENCE_LABELS,
  checkQuad,
  cornersFromReference,
  isOffFrame,
  netErrorMetres,
  projectCourtLines,
  referenceFromCorners,
  type Pt,
  type Reference,
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

/** Where the handles start when switching to a reference from nothing usable. */
function defaultHandles(reference: Reference, frameW: number, frameH: number): Pt[] {
  const at = (fx: number, fy: number): Pt => [fx * frameW, fy * frameH]
  return reference === 'net_service'
    ? [at(0.25, 0.5), at(0.75, 0.5), at(0.9, 0.78), at(0.1, 0.78)]
    : [at(0.3, 0.3), at(0.7, 0.3), at(0.95, 0.92), at(0.05, 0.92)]
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
// The analysis must keep at least this much video after its start (the
// server refuses less).
const MIN_ANALYSED_S = 10

export const CalibrationView: FC<Props> = ({ match, onCalibrated, onBack }) => {
  const frameW = match.width ?? 1920
  const frameH = match.height ?? 1080

  // The four handles, placed on the points of `reference`; the court's
  // corners are computed from them.
  const [handles, setHandles] = useState<Pt[] | null>(null)
  const [reference, setReference] = useState<Reference>('corners')
  const [netPoints, setNetPoints] = useState<Pt[] | null>(null)
  const [presets, setPresets] = useState<CameraPreset[]>([])
  const [presetName, setPresetName] = useState('')
  // The preset whose deletion waits for a confirmation.
  const [deleting, setDeleting] = useState<CameraPreset | null>(null)
  const [dragging, setDragging] = useState<Handle | null>(null)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [hint, setHint] = useState<string | null>(null)
  // Fraction of the frame added on every side; 0 = the keyframe only.
  const [margin, setMargin] = useState(0)
  // Lens correction for a wide-angle lens (phone 0.6x, action cameras): the
  // projected lines bend with it, to follow the real ones.
  const [k1, setK1] = useState(0)
  // Rendered width of the canvas, for the minimum on-screen handle size.
  const [renderWidth, setRenderWidth] = useState(0)
  // Where the analysis starts, in seconds of video: the warm-up before the
  // match, players walking on, are left out.
  const [start, setStart] = useState(match.analysis_start_s ?? 0)
  const [startText, setStartText] = useState(clock(match.analysis_start_s ?? 0))
  const [videoBroken, setVideoBroken] = useState(false)
  const videoRef = useRef<HTMLVideoElement>(null)

  const svgRef = useRef<SVGSVGElement>(null)
  const loaded = handles !== null

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
        const corners = suggestion.corners_px.map(p => [p[0], p[1]] as Pt)
        const lens = { k1: suggestion.k1 ?? 0, frameW, frameH }
        // Corners too far out to be placed by hand: the net and the service
        // line are, whatever the calibration was saved with.
        const tooFar = corners.some(p => offFrameExtent([p], frameW, frameH) > MAX_OFF_FRAME)
        const kind: Reference = tooFar ? 'net_service' : suggestion.reference ?? 'corners'
        const initial = referenceFromCorners(corners, kind, lens) ?? corners
        setK1(lens.k1)
        setReference(initial === corners ? 'corners' : kind)
        setHandles(initial)
        // A saved calibration may already have handles out of the picture.
        if (initial.some(p => isOffFrame(p, frameW, frameH))) {
          setMargin(marginFor(initial, frameW, frameH))
        }
        setPresets(presetList)
        setHint(
          suggestion.note ??
            (suggestion.method === 'saved'
              ? 'Calibrazione già salvata per questa partita: modificala se serve.'
              : 'Rete e linee non riconosciute in automatico: trascina i 4 punti sul campo.'),
        )
      })
      .catch((e: Error) => !cancelled && setError(e.message))
    return () => {
      cancelled = true
    }
  }, [match.id, frameW, frameH])

  const corners = useMemo(
    () => (handles ? cornersFromReference(handles, reference, { k1, frameW, frameH }) : null),
    [handles, reference, k1, frameW, frameH],
  )
  const check = useMemo(
    () =>
      handles
        ? corners
          ? checkQuad(handles, frameW, frameH, k1, MIN_HANDLES_AREA[reference])
          : { valid: false, reason: 'I punti non definiscono una prospettiva valida.' }
        : { valid: false },
    [handles, corners, reference, frameW, frameH, k1],
  )
  const courtLines = useMemo(
    () => (corners && check.valid ? projectCourtLines(corners, { k1, frameW, frameH }) : []),
    [corners, check.valid, k1, frameW, frameH],
  )
  const netError = useMemo(
    () => (corners && netPoints && check.valid ? netErrorMetres(corners, netPoints, { k1, frameW, frameH }) : null),
    [corners, netPoints, check.valid, k1, frameW, frameH],
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
      setHandles(prev => prev && prev.map((p, i) => (i === dragging.index ? point : p)))
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
      if (startProblem) throw new ApiError(422, startProblem)
      const result = await api.submitCalibration(match.id, {
        preset_id: preset.id,
        analysis_start_s: start,
      })
      const presetCorners = result.corners_px.map(p => [p[0], p[1]] as Pt)
      setHandles(referenceFromCorners(presetCorners, reference, { k1, frameW, frameH }) ?? presetCorners)
      setHint(result.message ?? `Preset "${preset.name}" applicato.`)
      onCalibrated()
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e))
    } finally {
      setSaving(false)
    }
  }

  const deletePreset = async (preset: CameraPreset) => {
    setError(null)
    setSaving(true)
    try {
      await api.deletePreset(preset.id)
      setPresets(prev => prev.filter(p => p.id !== preset.id))
      setHint(`Posizione camera "${preset.name}" eliminata.`)
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e))
    } finally {
      setDeleting(null)
      setSaving(false)
    }
  }

  const save = async () => {
    if (!handles || !check.valid || startProblem) return
    setError(null)
    setSaving(true)
    try {
      await api.submitCalibration(match.id, {
        corners_px: handles.map(p => [p[0], p[1]] as Point),
        reference,
        net_px: netPoints ? netPoints.map(p => [p[0], p[1]] as Point) : null,
        save_as_preset: presetName.trim() || undefined,
        analysis_start_s: start,
        k1,
      })
      onCalibrated()
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e))
      setSaving(false)
    }
  }

  const duration = match.duration_seconds ?? null
  const startProblem =
    duration !== null && start > Math.max(duration - MIN_ANALYSED_S, 0)
      ? `L'inizio dell'analisi deve lasciare almeno ${MIN_ANALYSED_S} secondi di video (durata ${clock(duration)}).`
      : null

  const chooseStart = (seconds: number) => {
    const value = Math.max(0, Math.round(seconds))
    setStart(value)
    setStartText(clock(value))
  }

  const typeStart = (text: string) => {
    setStartText(text)
    const seconds = parseClock(text)
    if (seconds !== null) setStart(seconds)
  }

  /** Switching back to the keyframe only brings off-frame corners to its edge. */
  const toggleExtended = () => {
    if (margin > 0) {
      const clamp = (p: Pt): Pt => [
        Math.max(0, Math.min(frameW, p[0])),
        Math.max(0, Math.min(frameH, p[1])),
      ]
      setHandles(prev => prev && prev.map(clamp))
      setNetPoints(prev => prev && prev.map(clamp))
      setMargin(0)
    } else {
      setMargin(DEFAULT_MARGIN)
    }
  }

  /** The same calibration on the other handles when it can be placed by hand
   *  on them; otherwise the handles start over. */
  const switchReference = (next: Reference) => {
    if (next === reference || !handles) return
    const moved = corners && check.valid ? referenceFromCorners(corners, next, { k1, frameW, frameH }) : null
    const usable = moved && moved.every(p => offFrameExtent([p], frameW, frameH) <= MAX_OFF_FRAME)
    const nextHandles = usable ? moved : defaultHandles(next, frameW, frameH)
    setReference(next)
    setHandles(nextHandles)
    if (nextHandles.some(p => isOffFrame(p, frameW, frameH))) {
      setMargin(m => Math.max(m, marginFor(nextHandles, frameW, frameH)))
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

  if (!handles) {
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
  const offFrame = handles.map(p => isOffFrame(p, frameW, frameH))
  const labels = REFERENCE_LABELS[reference]
  const anyOffFrame = offFrame.some(Boolean)

  return (
    <div className="layout">
      <div className="header">
        <h1>Calibrazione campo</h1>
        <button className="btn btn-ghost btn-sm" onClick={onBack}>← Indietro</button>
      </div>

      <div className="card card-sm" style={{ marginBottom: '1rem' }}>
        <div className="preset-row">
          <span style={{ fontWeight: 600 }}>Punti di riferimento:</span>
          <button
            className={`btn btn-sm ${reference === 'corners' ? 'btn-primary' : 'btn-ghost'}`}
            aria-pressed={reference === 'corners'}
            onClick={() => switchReference('corners')}
          >
            Angoli del campo
          </button>
          <button
            className={`btn btn-sm ${reference === 'net_service' ? 'btn-primary' : 'btn-ghost'}`}
            aria-pressed={reference === 'net_service'}
            onClick={() => switchReference('net_service')}
          >
            Rete e linea di servizio
          </button>
        </div>
      </div>

      <div className="callout callout-info" style={{ marginBottom: '1rem' }}>
        {reference === 'corners' ? (
          <>
            <strong>Trascina i 4 angoli sugli spigoli del campo</strong>, partendo dal fondo a
            sinistra e procedendo in senso orario.
          </>
        ) : (
          <>
            <strong>Trascina i 4 punti</strong>: la base del palo sinistro e del palo destro della
            rete, poi l'estremità destra e sinistra della linea di servizio vicina, dove tocca la
            parete. Adatto alla camera bassa o vicina al vetro, quando gli angoli vicini del campo
            sono lontani dall'inquadratura.
          </>
        )}{' '}
        Le linee bianche sovrapposte mostrano dove il sistema calcola rete e linee di servizio: se
        coincidono con quelle reali, la calibrazione è corretta.
        {margin === 0 && (
          <>
            {' '}
            Se un punto non è inquadrato, usa <strong>Estendi oltre l'inquadratura</strong>.
          </>
        )}
      </div>

      {margin > 0 && (
        <div className="callout callout-warn" style={{ marginBottom: '1rem' }}>
          <strong>Punti fuori inquadratura:</strong> porta il punto nella zona scura dove si
          incontrerebbero le linee e le pareti visibili, poi controlla che rete e linee di servizio
          proiettate coincidano con quelle reali. Serve più spazio? Lascia il punto sul bordo della
          zona scura: si allarga. Con punti fuori campo la <strong>verifica con la rete</strong> è
          consigliata.
        </div>
      )}

      {presets.length > 0 && (
        <div className="card card-sm" style={{ marginBottom: '1rem' }}>
          <div className="preset-row">
            <span style={{ fontWeight: 600 }}>Posizione camera salvata:</span>
            {presets.map(preset =>
              deleting?.id === preset.id ? (
                <span key={preset.id} className="preset-confirm" role="group" aria-label={`Elimina ${preset.name}`}>
                  Eliminare «{preset.name}»?
                  <span className="preset-confirm-actions">
                    <button className="btn btn-danger btn-sm" disabled={saving} onClick={() => deletePreset(preset)}>
                      Elimina
                    </button>
                    <button className="btn btn-ghost btn-sm" disabled={saving} autoFocus onClick={() => setDeleting(null)}>
                      Annulla
                    </button>
                  </span>
                </span>
              ) : (
                <span key={preset.id} className="preset-chip">
                  <button className="btn btn-ghost btn-sm" disabled={saving} onClick={() => applyPreset(preset)}>
                    {preset.name}
                  </button>
                  <button
                    className="btn btn-ghost btn-sm preset-remove"
                    disabled={saving}
                    onClick={() => setDeleting(preset)}
                    aria-label={`Elimina la posizione camera ${preset.name}`}
                    title="Elimina"
                  >
                    ×
                  </button>
                </span>
              ),
            )}
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
            points={(courtLines[0] ?? handles).map(p => p.join(',')).join(' ')}
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

          {handles.map((point, index) => {
            // Label below the handle when there is no room above it.
            const labelBelow = point[1] - handleRadius * 3.2 < view.y
            // Anchored inwards when it would overflow a side (≈0.55 em per character).
            const label = `${index + 1}. ${labels[index]}${offFrame[index] ? ' (fuori)' : ''}`
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
            {labels.map((label, index) => (
              <span key={label}>
                <span className="player-dot" style={{ background: HANDLE_COLORS[index] }} />
                {index + 1}. {label}
                {offFrame[index] ? ' (fuori)' : ''}
              </span>
            ))}
          </div>
        )}
      </div>

      <div className="card card-sm lens-row">
        <label htmlFor="lens-k1">
          <strong>Obiettivo grandangolare</strong>
          <span className="muted-note">
            {' '}
            (es. 0.6x del telefono): se le linee reali del campo sono curve, aumenta finché le linee bianche le
            seguono.
          </span>
        </label>
        <div className="lens-control">
          <input
            id="lens-k1"
            type="range"
            min={0}
            max={MAX_K1}
            step={0.01}
            value={k1}
            onChange={e => setK1(Number(e.target.value))}
          />
          <span className="lens-value">{k1 > 0 ? k1.toFixed(2) : 'nessuna'}</span>
          {k1 > 0 && (
            <button className="btn btn-ghost btn-sm" type="button" onClick={() => setK1(0)}>
              Azzera
            </button>
          )}
        </div>
      </div>

      {hint && !error && <p className="muted-note">{hint}</p>}
      {!check.valid && check.reason && <p className="error-msg">{check.reason}</p>}
      {error && <p className="error-msg">{error}</p>}

      {anyOffFrame && !netPoints && (
        <p className="muted-note">
          {offFrame.filter(Boolean).length === 1 ? 'Un punto è' : 'Alcuni punti sono'} fuori
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
        <h2 style={{ marginBottom: '.35rem' }}>Inizio dell'analisi</h2>
        <p className="muted-note" style={{ marginBottom: '.75rem' }}>
          Escludi i minuti iniziali in cui i giocatori non sono ancora in campo: porta il video al
          momento in cui inizia il gioco e premi <strong>Inizia da qui</strong>, oppure scrivi il
          minuto.
        </p>
        {!videoBroken && (
          <video
            ref={videoRef}
            className="start-video"
            src={api.videoUrl(match.id)}
            controls
            playsInline
            muted
            preload="metadata"
            onError={() => setVideoBroken(true)}
          />
        )}
        {videoBroken && (
          <p className="muted-note">
            Il browser non riesce a riprodurre questo video: scrivi il minuto di inizio.
          </p>
        )}
        <div className="start-row">
          {!videoBroken && (
            <button
              className="btn btn-ghost btn-sm"
              type="button"
              onClick={() => videoRef.current && chooseStart(videoRef.current.currentTime)}
            >
              Inizia da qui
            </button>
          )}
          <label className="start-field">
            Dal minuto
            <input
              type="text"
              inputMode="numeric"
              value={startText}
              aria-invalid={parseClock(startText) === null || startProblem !== null}
              onChange={e => typeStart(e.target.value)}
              onBlur={() => parseClock(startText) !== null && setStartText(clock(start))}
              placeholder="m:ss"
            />
          </label>
          {start > 0 && (
            <button className="btn btn-ghost btn-sm" type="button" onClick={() => chooseStart(0)}>
              Dall'inizio del video
            </button>
          )}
        </div>
        {parseClock(startText) === null ? (
          <p className="error-msg">Scrivi il minuto come m:ss, ad esempio 3:20.</p>
        ) : (
          startProblem && <p className="error-msg">{startProblem}</p>
        )}
        {!startProblem && start > 0 && (
          <p className="muted-note">
            L'analisi parte dal minuto {clock(start)}: la parte precedente non viene analizzata.
          </p>
        )}
      </div>

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
            disabled={!check.valid || saving || startProblem !== null || parseClock(startText) === null}
            onClick={save}
          >
            {saving ? 'Salvataggio…' : 'Conferma calibrazione'}
          </button>
        </div>
      </div>
    </div>
  )
}
