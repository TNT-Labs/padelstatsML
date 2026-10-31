import { useEffect, useState, type FC, type RefObject } from 'react'
import { PLAYER_COLORS } from '../lib/players'

interface Props {
  title: string
  /** Player ids in order: key 1 is the first, 2 the second… */
  players: number[]
  names: string[]
  videoRef: RefObject<HTMLVideoElement>
  hits: [number, number][]
  onChange: (hits: [number, number][]) => void
  error: string | null
}

// One frame of a 30 fps video, for stepping.
const FRAME_S = 1 / 30
const SPEEDS = [0.25, 0.5, 1]
// Hits listed around the current time.
const AROUND_S = 8

function stamp(t: number): string {
  const m = Math.floor(t / 60)
  return `${m}:${(t - 60 * m).toFixed(2).padStart(5, '0')}`
}

/**
 * Marking the hits while watching the match: the truth the hits found from
 * the ball are measured against. Pause on the frame where the ball leaves
 * the racket and press the player's number; at 0.25x it can be done while
 * the video plays.
 */
export const HitMarker: FC<Props> = ({ title, players, names, videoRef, hits, onChange, error }) => {
  const [now, setNow] = useState(0)
  const [speed, setSpeed] = useState(1)
  const [copied, setCopied] = useState<string | null>(null)
  const name = (pid: number) => names[pid] || `Giocatore ${pid + 1}`

  useEffect(() => {
    const video = videoRef.current
    if (!video) return
    const tick = () => setNow(video.currentTime)
    video.addEventListener('timeupdate', tick)
    video.addEventListener('seeked', tick)
    return () => {
      video.removeEventListener('timeupdate', tick)
      video.removeEventListener('seeked', tick)
    }
  }, [videoRef])

  useEffect(() => {
    const video = videoRef.current
    if (video) video.playbackRate = speed
  }, [speed, videoRef])

  useEffect(() => {
    const keys = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null
      if (target && ['INPUT', 'SELECT', 'TEXTAREA'].includes(target.tagName)) return
      if (e.ctrlKey || e.metaKey || e.altKey) return
      const video = videoRef.current
      if (!video) return
      const index = ['1', '2', '3', '4'].indexOf(e.key)
      if (index >= 0 && index < players.length) {
        e.preventDefault()
        const t = Math.round(video.currentTime * 100) / 100
        const next = [...hits, [t, players[index]] as [number, number]].sort((a, b) => a[0] - b[0])
        onChange(next)
      } else if (e.key === ',' || e.key === '.') {
        e.preventDefault()
        video.pause()
        video.currentTime = Math.max(0, video.currentTime + (e.key === '.' ? FRAME_S : -FRAME_S))
      } else if (e.key === ' ') {
        e.preventDefault()
        if (video.paused) void video.play()
        else video.pause()
      }
    }
    window.addEventListener('keydown', keys)
    return () => window.removeEventListener('keydown', keys)
  }, [hits, onChange, players, videoRef])

  const near = hits.filter(([t]) => Math.abs(t - now) <= AROUND_S)

  const copy = async () => {
    const text = [
      `${title} · colpi segnati ${hits.length}`,
      ...hits.map(([t, pid]) => `${stamp(t)}\t${name(pid)}\t${pid}`),
    ].join('\n')
    try {
      await navigator.clipboard.writeText(text)
      setCopied('Copiati: incollali nella conversazione.')
    } catch {
      setCopied(text)
    }
  }

  return (
    <div className="hit-marker">
      <p className="muted-note" style={{ marginTop: 0 }}>
        Metti in pausa sul fotogramma in cui la palla lascia la racchetta e premi il numero del
        giocatore. <strong>,</strong> e <strong>.</strong> tornano indietro e avanzano di un fotogramma,
        la <strong>barra spaziatrice</strong> avvia e ferma. Segna tutti i colpi di qualche scambio
        intero, battuta compresa.
      </p>
      <div className="hit-keys">
        {players.map((pid, i) => (
          <span key={pid} className="hit-key">
            <kbd>{i + 1}</kbd>
            <span className="player-dot" style={{ background: PLAYER_COLORS[pid % 4] }} />
            {name(pid)}
          </span>
        ))}
        <span className="hit-speed">
          Velocità
          {SPEEDS.map(v => (
            <button
              key={v}
              type="button"
              className={`btn btn-sm ${speed === v ? 'btn-primary' : 'btn-ghost'}`}
              onClick={() => setSpeed(v)}
            >
              {v}×
            </button>
          ))}
        </span>
      </div>
      <div className="hit-list">
        <span className="muted-note">
          Segnati {hits.length} · qui vicino ({stamp(now)}):
        </span>
        {near.length === 0 && <span className="muted-note">nessuno</span>}
        {near.map(([t, pid]) => (
          <span key={`${t}:${pid}`} className="hit-chip" style={{ borderColor: PLAYER_COLORS[pid % 4] }}>
            <button
              type="button"
              className="hit-chip-time"
              onClick={() => {
                const video = videoRef.current
                if (video) {
                  video.pause()
                  video.currentTime = t
                }
              }}
            >
              {stamp(t)} {name(pid)}
            </button>
            <button
              type="button"
              className="hit-chip-x"
              aria-label={`Togli il colpo di ${name(pid)} a ${stamp(t)}`}
              onClick={() => onChange(hits.filter(h => !(h[0] === t && h[1] === pid)))}
            >
              ×
            </button>
          </span>
        ))}
      </div>
      {hits.length > 0 && (
        <button type="button" className="btn btn-ghost btn-sm" onClick={copy}>
          Copia i colpi segnati
        </button>
      )}
      {copied && (copied.includes('\n') ? (
        <textarea className="stroke-copy" readOnly value={copied} onFocus={e => e.target.select()} />
      ) : (
        <p className="muted-note">{copied}</p>
      ))}
      {error && (
        <div className="callout callout-error" role="alert" style={{ marginTop: '.5rem' }}>
          {error}
        </div>
      )}
    </div>
  )
}
