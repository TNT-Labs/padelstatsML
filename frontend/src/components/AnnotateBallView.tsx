import { useCallback, useEffect, useMemo, useRef, useState, type FC } from 'react'
import { api, type BallClips, type BallLabel } from '../api'
import { clock } from '../lib/videoOverlay'

interface Props {
  matchId: string
  title: string
  onBack: () => void
}

// The lens: this many times larger, this many pixels on each side.
const ZOOM = 4
const LENS_PX = 220

/**
 * Marking the ball, frame by frame, in short clips of the match's rallies:
 * the data a ball detector is trained on (backend app/ml/ball_data.py).
 * One click on the ball marks the frame and moves to the next; N for a ball
 * out of sight. The lens shows the ball, a few pixels in the far half, four
 * times larger around the pointer.
 */
export const AnnotateBallView: FC<Props> = ({ matchId, title, onBack }) => {
  const [data, setData] = useState<BallClips | null>(null)
  const [labels, setLabels] = useState<Record<string, BallLabel>>({})
  const [clipIndex, setClipIndex] = useState(0)
  const [offset, setOffset] = useState(0)
  const [error, setError] = useState<string | null>(null)
  const [loaded, setLoaded] = useState(false)
  const [pointer, setPointer] = useState<[number, number] | null>(null)
  const imgRef = useRef<HTMLImageElement>(null)
  const overlayRef = useRef<HTMLCanvasElement>(null)
  const lensRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    Promise.all([api.ballClips(matchId), api.ballLabels(matchId)])
      .then(([clips, stored]) => {
        setData(clips)
        setLabels(stored.labels)
        // Resume at the first clip not marked to the end.
        const first = clips.clips.findIndex(c => c.labelled < c.frames)
        setClipIndex(first >= 0 ? first : 0)
      })
      .catch(e => setError((e as Error).message))
  }, [matchId])

  const clip = data?.clips[clipIndex]
  const frame = clip ? clip.start_frame + offset : 0
  const label = clip ? labels[String(frame)] : undefined
  const previous = clip && offset > 0 ? labels[String(frame - 1)] : undefined
  const src = clip ? api.ballFrameUrl(matchId, clip.id, offset) : ''

  const marked = useMemo(() => Object.keys(labels).length, [labels])
  const visible = useMemo(() => Object.values(labels).filter(v => v !== 'none').length, [labels])
  const markedInClip = clip
    ? Array.from({ length: clip.frames }, (_, i) => labels[String(clip.start_frame + i)]).filter(Boolean).length
    : 0

  // The next frame, fetched ahead: the clip is cut on the Pi at the first.
  useEffect(() => {
    if (!clip || offset + 1 >= clip.frames) return
    const next = new Image()
    next.src = api.ballFrameUrl(matchId, clip.id, offset + 1)
  }, [clip, offset, matchId])

  useEffect(() => setLoaded(false), [src])

  const save = useCallback(
    async (value: BallLabel | null) => {
      if (!clip) return
      const key = String(frame)
      const before = labels
      const next = { ...labels }
      if (value === null) delete next[key]
      else next[key] = value
      setLabels(next)
      setError(null)
      try {
        await api.labelBall(matchId, { [key]: value })
      } catch (e) {
        setLabels(before)
        setError((e as Error).message)
      }
    },
    [clip, frame, labels, matchId],
  )

  const step = useCallback(
    (delta: number) => {
      if (!clip) return
      setOffset(o => Math.min(clip.frames - 1, Math.max(0, o + delta)))
    },
    [clip],
  )

  const goClip = (index: number) => {
    if (!data || index < 0 || index >= data.clips.length) return
    setClipIndex(index)
    setOffset(0)
  }

  const markAndNext = async (value: BallLabel) => {
    await save(value)
    step(1)
  }

  useEffect(() => {
    const keys = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement)?.tagName === 'INPUT') return
      if (e.key === 'ArrowRight' || e.key === 'd') step(1)
      else if (e.key === 'ArrowLeft' || e.key === 'a') step(-1)
      else if (e.key === 'n' || e.key === ' ') {
        e.preventDefault()
        void markAndNext('none')
      } else if (e.key === 'Delete' || e.key === 'Backspace') void save(null)
      else return
    }
    window.addEventListener('keydown', keys)
    return () => window.removeEventListener('keydown', keys)
  })

  /** Video pixels of a pointer event on the picture. */
  const toVideo = (e: React.MouseEvent): [number, number] | null => {
    const img = imgRef.current
    if (!img || !img.naturalWidth) return null
    const rect = img.getBoundingClientRect()
    return [
      ((e.clientX - rect.left) / rect.width) * img.naturalWidth,
      ((e.clientY - rect.top) / rect.height) * img.naturalHeight,
    ]
  }

  // Markers on the picture, and the lens.
  useEffect(() => {
    const img = imgRef.current
    const overlay = overlayRef.current
    if (!img || !overlay || !loaded) return
    const rect = img.getBoundingClientRect()
    overlay.width = rect.width
    overlay.height = rect.height
    const ctx = overlay.getContext('2d')
    if (!ctx) return
    const sx = rect.width / img.naturalWidth
    ctx.clearRect(0, 0, overlay.width, overlay.height)
    const ring = (p: [number, number], color: string, r: number) => {
      ctx.strokeStyle = color
      ctx.lineWidth = 2
      ctx.beginPath()
      ctx.arc(p[0] * sx, p[1] * sx, r, 0, 2 * Math.PI)
      ctx.stroke()
    }
    if (Array.isArray(previous)) ring(previous, 'rgba(255,255,255,.55)', 7)
    if (Array.isArray(label)) ring(label, '#ff3b30', 8)

    const lens = lensRef.current
    const lctx = lens?.getContext('2d')
    if (!lens || !lctx) return
    lctx.fillStyle = '#111'
    lctx.fillRect(0, 0, LENS_PX, LENS_PX)
    const centre = pointer ?? (Array.isArray(label) ? label : null)
    if (!centre) return
    const half = LENS_PX / ZOOM / 2
    lctx.imageSmoothingEnabled = false
    lctx.drawImage(img, centre[0] - half, centre[1] - half, 2 * half, 2 * half, 0, 0, LENS_PX, LENS_PX)
    const inLens = (p: [number, number]): [number, number] => [
      (p[0] - centre[0] + half) * ZOOM,
      (p[1] - centre[1] + half) * ZOOM,
    ]
    lctx.lineWidth = 2
    if (Array.isArray(label)) {
      const [x, y] = inLens(label)
      lctx.strokeStyle = '#ff3b30'
      lctx.beginPath()
      lctx.arc(x, y, 14, 0, 2 * Math.PI)
      lctx.stroke()
    }
    lctx.strokeStyle = 'rgba(255,255,255,.7)'
    lctx.lineWidth = 1
    lctx.beginPath()
    lctx.moveTo(LENS_PX / 2, LENS_PX / 2 - 12)
    lctx.lineTo(LENS_PX / 2, LENS_PX / 2 + 12)
    lctx.moveTo(LENS_PX / 2 - 12, LENS_PX / 2)
    lctx.lineTo(LENS_PX / 2 + 12, LENS_PX / 2)
    lctx.stroke()
  }, [label, previous, pointer, loaded])

  if (error && !data) {
    return (
      <div className="layout">
        <div className="header">
          <h1>Annota la palla</h1>
          <button className="btn btn-ghost btn-sm" onClick={onBack}>← Statistiche</button>
        </div>
        <div className="callout callout-error">{error}</div>
      </div>
    )
  }
  if (!data || !clip) {
    return (
      <div className="layout" style={{ textAlign: 'center', paddingTop: '5rem' }}>
        <p className="muted-note">Preparazione degli spezzoni…</p>
      </div>
    )
  }

  return (
    <div className="layout">
      <div className="header">
        <h1>Annota la palla</h1>
        <div style={{ display: 'flex', gap: '.5rem', flexWrap: 'wrap' }}>
          {marked > 0 && (
            <a className="btn btn-ghost btn-sm" href={api.ballExportUrl(matchId)} download>
              Scarica i dati annotati
            </a>
          )}
          <button className="btn btn-ghost btn-sm" onClick={onBack}>← Statistiche</button>
        </div>
      </div>

      <p className="muted-note" style={{ marginBottom: '.75rem' }}>
        {title} · fotogrammi annotati <strong>{marked}</strong> (palla visibile in {visible}). Clicca sul
        centro della palla, anche se è una scia; <strong>N</strong> o spazio se non si vede (dietro un
        giocatore, fuori inquadratura). <strong>A / D</strong> o le frecce per tornare indietro e
        avanzare, <strong>Canc</strong> per togliere un segno. Il cerchio bianco è la palla del
        fotogramma prima.
      </p>

      <div className="ball-bar">
        <button className="btn btn-ghost btn-sm" onClick={() => goClip(clipIndex - 1)} disabled={clipIndex === 0}>
          ◀ Spezzone
        </button>
        <span>
          Spezzone {clipIndex + 1}/{data.clips.length} · min {clock(clip.start_s)} · fotogramma {offset + 1}/
          {clip.frames} · annotati {markedInClip}/{clip.frames}
        </span>
        <button
          className="btn btn-ghost btn-sm"
          onClick={() => goClip(clipIndex + 1)}
          disabled={clipIndex + 1 >= data.clips.length}
        >
          Spezzone ▶
        </button>
      </div>

      <div className="ball-stage">
        <div className="ball-frame">
          <img
            ref={imgRef}
            src={src}
            alt={`Fotogramma ${frame}`}
            onLoad={() => setLoaded(true)}
            draggable={false}
          />
          <canvas
            ref={overlayRef}
            className="ball-overlay"
            onMouseMove={e => setPointer(toVideo(e))}
            onMouseLeave={() => setPointer(null)}
            onClick={e => {
              const p = toVideo(e)
              if (p) void markAndNext([Math.round(p[0] * 10) / 10, Math.round(p[1] * 10) / 10])
            }}
          />
          {!loaded && <div className="ball-loading">Caricamento del fotogramma…</div>}
        </div>
        <div className="ball-side">
          <canvas ref={lensRef} width={LENS_PX} height={LENS_PX} className="ball-lens" aria-hidden="true" />
          <div className="ball-state">
            {label === undefined ? 'da annotare' : label === 'none' ? 'palla non visibile' : 'palla segnata'}
          </div>
          <div className="ball-buttons">
            <button className="btn btn-ghost btn-sm" onClick={() => step(-1)} disabled={offset === 0}>
              ◀ Indietro
            </button>
            <button className="btn btn-ghost btn-sm" onClick={() => step(1)} disabled={offset + 1 >= clip.frames}>
              Avanti ▶
            </button>
            <button className="btn btn-ghost btn-sm" onClick={() => void markAndNext('none')}>
              Non visibile (N)
            </button>
            <button className="btn btn-ghost btn-sm" onClick={() => void save(null)} disabled={label === undefined}>
              Cancella
            </button>
          </div>
          {offset + 1 >= clip.frames && markedInClip === clip.frames && clipIndex + 1 < data.clips.length && (
            <button className="btn btn-primary btn-sm" onClick={() => goClip(clipIndex + 1)}>
              Spezzone successivo ▶
            </button>
          )}
        </div>
      </div>
      {error && (
        <div className="callout callout-error" role="alert" style={{ marginTop: '.75rem' }}>
          {error}
        </div>
      )}
    </div>
  )
}
