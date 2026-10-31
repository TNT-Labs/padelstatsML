import { useCallback, useEffect, useRef, useState, type FC, type RefObject } from 'react'
import { api, type BallTrack, type PlayerTracks } from '../api'
import { PLAYER_COLORS } from '../lib/players'
import { MIN_VISIBLE, SKELETON, ballAt, boxAt, clock, fitRect, placeTags, pointsAt, type Tag } from '../lib/videoOverlay'

interface Props {
  matchId: string
  names: string[]
  /** Owned by the page, so the rally list can seek the same video. */
  videoRef: RefObject<HTMLVideoElement>
  onSeek: (seconds: number) => void
}

// Name tags: smaller on a phone, where the far players are a few pixels
// tall and a desktop-sized tag would cover them.
const NARROW_PX = 600
const TAG = { wide: { font: 12, height: 16 }, narrow: { font: 10, height: 13 } }
// A changeover's time is when the pairs are found on their new ends: start
// a little earlier, to see them walk there.
const CHANGEOVER_LEAD_S = 10

/**
 * The match video, with each player's box and name drawn over it.
 *
 * Numbers can look right while two players are swapped: watching who the
 * boxes follow is how to check. The boxes are drawn on a transparent canvas
 * laid over the native player, so seeking, full-speed playback and the
 * browser's own controls all keep working.
 */
export const MatchVideo: FC<Props> = ({ matchId, names, videoRef, onSeek }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [tracks, setTracks] = useState<PlayerTracks | null>(null)
  const [tracksNote, setTracksNote] = useState<string | null>(null)
  const [loadingTracks, setLoadingTracks] = useState(true)
  const [showBoxes, setShowBoxes] = useState(true)
  const [showSkeleton, setShowSkeleton] = useState(true)
  // The ball, when the analysis followed it (ball network on the accelerator).
  const [ball, setBall] = useState<BallTrack | null>(null)
  const [showBall, setShowBall] = useState(true)
  const hasSkeleton = !!tracks?.players.some(p => p.kp)
  const [videoError, setVideoError] = useState<string | null>(null)
  const src = api.videoUrl(matchId)

  useEffect(() => {
    let cancelled = false
    setTracks(null)
    setTracksNote(null)
    setLoadingTracks(true)
    api
      .getPlayerTracks(matchId)
      .then(result => !cancelled && setTracks(result))
      .catch(err => !cancelled && setTracksNote(err instanceof Error ? err.message : String(err)))
      .finally(() => !cancelled && setLoadingTracks(false))
    setBall(null)
    api
      .ballTrack(matchId)
      .then(result => !cancelled && setBall(result.flights.length ? result : null))
      .catch(() => undefined)          // 404: no ball in this analysis
    return () => {
      cancelled = true
    }
  }, [matchId])

  const draw = useCallback(() => {
    const video = videoRef.current
    const canvas = canvasRef.current
    if (!video || !canvas) return
    const dpr = window.devicePixelRatio || 1
    const width = video.clientWidth
    const height = video.clientHeight
    if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) {
      canvas.width = Math.round(width * dpr)
      canvas.height = Math.round(height * dpr)
    }
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    ctx.setTransform(1, 0, 0, 1, 0, 0)
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    const showingBall = showBall && !!ball
    if ((!showBoxes || !tracks) && !showingBall) return

    const frameW = tracks?.frame.width || ball?.frame[0] || video.videoWidth
    const frameH = tracks?.frame.height || ball?.frame[1] || video.videoHeight
    if (!frameW || !frameH) return
    const { scale, offsetX, offsetY } = fitRect(frameW, frameH, width, height)
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)

    if (showingBall) {
      const found = ballAt(ball.flights, video.currentTime)
      if (found) {
        const narrow = width < NARROW_PX
        ctx.strokeStyle = 'rgba(255, 214, 10, .85)'
        ctx.lineWidth = narrow ? 1.5 : 2
        ctx.beginPath()
        found.trail.forEach(([x, y], k) => {
          const px = offsetX + x * scale
          const py = offsetY + y * scale
          if (k === 0) ctx.moveTo(px, py)
          else ctx.lineTo(px, py)
        })
        ctx.stroke()
        ctx.strokeStyle = '#ff3b30'
        ctx.lineWidth = 2
        ctx.beginPath()
        ctx.arc(offsetX + found.at[0] * scale, offsetY + found.at[1] * scale, narrow ? 5 : 8, 0, 2 * Math.PI)
        ctx.stroke()
      }
    }
    if (!showBoxes || !tracks) return
    // Samples are a few per second: bridge one missed sample, not more.
    const maxGap = 2.5 / (tracks.sample_hz || 5)

    const tag = width < NARROW_PX ? TAG.narrow : TAG.wide
    ctx.lineWidth = width < NARROW_PX ? 1.5 : 2
    ctx.font = `600 ${tag.font}px system-ui, sans-serif`
    ctx.textBaseline = 'bottom'
    // Names after every box and skeleton, placed so that none hides another.
    const tags: (Tag & { label: string; color: string })[] = []
    for (const player of tracks.players) {
      const box = boxAt(player.t, player.box, video.currentTime, maxGap)
      if (!box) continue
      const color = PLAYER_COLORS[player.id % PLAYER_COLORS.length]
      const x = offsetX + box[0] * scale
      const y = offsetY + box[1] * scale
      ctx.strokeStyle = color
      ctx.strokeRect(x, y, (box[2] - box[0]) * scale, (box[3] - box[1]) * scale)

      const points = showSkeleton ? pointsAt(player.t, player.kp, video.currentTime, maxGap) : null
      if (points) {
        const px = (k: number) => offsetX + points[3 * k] * scale
        const py = (k: number) => offsetY + points[3 * k + 1] * scale
        const seen = (k: number) => points[3 * k + 2] >= MIN_VISIBLE
        ctx.beginPath()
        for (const [a, b] of SKELETON) {
          if (!seen(a) || !seen(b)) continue
          ctx.moveTo(px(a), py(a))
          ctx.lineTo(px(b), py(b))
        }
        ctx.stroke()
        ctx.fillStyle = color
        for (let k = 0; k < 17; k++) {
          if (!seen(k)) continue
          ctx.beginPath()
          ctx.arc(px(k), py(k), width < NARROW_PX ? 1.5 : 2.5, 0, 2 * Math.PI)
          ctx.fill()
        }
      }

      const label = names[player.id] || `G${player.id + 1}`
      tags.push({
        label, color, x: x - 1, bottom: Math.max(y, tag.height),
        w: ctx.measureText(label).width + 6, h: tag.height,
      })
    }
    placeTags(tags).forEach((bottom, i) => {
      const { label, color, x, w, h } = tags[i]
      ctx.fillStyle = color
      ctx.fillRect(x, bottom - h, w, h)
      ctx.fillStyle = '#fff'
      ctx.fillText(label, x + 3, bottom - 2)
    })
  }, [tracks, showBoxes, showSkeleton, ball, showBall, names, videoRef])

  useEffect(() => {
    const video = videoRef.current
    if (!video) return
    let frame = 0
    const loop = () => {
      draw()
      frame = requestAnimationFrame(loop)
    }
    const start = () => {
      cancelAnimationFrame(frame)
      frame = requestAnimationFrame(loop)
    }
    const stop = () => {
      cancelAnimationFrame(frame)
      draw()
    }
    const playing = ['play', 'playing']
    const still = ['pause', 'ended', 'seeked', 'loadedmetadata']
    playing.forEach(e => video.addEventListener(e, start))
    still.forEach(e => video.addEventListener(e, stop))
    const resize = new ResizeObserver(() => draw())
    resize.observe(video)
    if (video.paused) draw()
    else start()
    return () => {
      cancelAnimationFrame(frame)
      playing.forEach(e => video.removeEventListener(e, start))
      still.forEach(e => video.removeEventListener(e, stop))
      resize.disconnect()
    }
  }, [draw, videoRef])

  const onVideoError = async () => {
    // A missing file and an unplayable one raise the same media error;
    // one byte from the server tells them apart.
    try {
      const probe = await fetch(src, { headers: { Range: 'bytes=0-0' } })
      if (!probe.ok) {
        setVideoError('Il file del video non è più sul Pi: restano solo le statistiche.')
        return
      }
    } catch {
      setVideoError('Il Pi non risponde: controlla la connessione.')
      return
    }
    setVideoError(
      'Il browser non riesce a riprodurre questo video. Succede di solito con i video ' +
        'H.265/HEVC degli iPhone: prova con Safari, oppure converti il file in H.264.',
    )
  }

  return (
    <div className="card" style={{ marginBottom: '1.25rem' }}>
      <div className="card-head">
        <h2 style={{ margin: 0 }}>Video della partita</h2>
        <div className="toggles">
          <label className="toggle">
            <input
              type="checkbox"
              checked={showBoxes}
              onChange={e => setShowBoxes(e.target.checked)}
              disabled={!tracks}
            />
            Mostra i giocatori
          </label>
          {hasSkeleton && (
            <label className="toggle">
              <input
                type="checkbox"
                checked={showSkeleton}
                onChange={e => setShowSkeleton(e.target.checked)}
                disabled={!showBoxes}
              />
              Scheletro
            </label>
          )}
          {ball && (
            <label className="toggle">
              <input type="checkbox" checked={showBall} onChange={e => setShowBall(e.target.checked)} />
              Palla
            </label>
          )}
        </div>
      </div>

      <div className="video-wrap">
        <video
          ref={videoRef}
          src={src}
          controls
          preload="metadata"
          playsInline
          onError={onVideoError}
        />
        <canvas ref={canvasRef} className="video-overlay" aria-hidden="true" />
      </div>

      {videoError && (
        <div className="callout callout-error" style={{ marginTop: '.75rem' }}>
          {videoError}
        </div>
      )}
      {loadingTracks && (
        <p className="muted-note" style={{ marginTop: '.6rem' }}>
          Preparazione dei riquadri dei giocatori… per le partite analizzate prima di questa
          versione, la prima volta può richiedere mezzo minuto.
        </p>
      )}
      {tracksNote && (
        <p className="muted-note" style={{ marginTop: '.6rem' }}>
          {tracksNote}
        </p>
      )}
      {tracks && !tracks.matches_stats && (
        <div className="callout callout-warn" style={{ marginTop: '.75rem' }}>
          I riquadri sono ricalcolati con la versione attuale del programma, le statistiche con
          quella dell'analisi: numeri e nomi possono non corrispondere. Rianalizza la partita per
          allinearli.
        </div>
      )}
      {tracks && tracks.changeovers.length > 0 && (
        <div className="seek-row">
          <span className="muted-note">Cambi di campo trovati:</span>
          {tracks.changeovers.map(t => (
            <button key={t} type="button" className="btn btn-ghost btn-sm" onClick={() => onSeek(Math.max(0, t - CHANGEOVER_LEAD_S))}>
              {clock(t)}
            </button>
          ))}
        </div>
      )}
    </div>
  )
}
