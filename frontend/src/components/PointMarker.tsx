import { useEffect, useState, type FC, type RefObject } from 'react'
import { PLAYER_COLORS } from '../lib/players'
import { addPoint, pointAt, pointsText, setWinner, stamp, type MarkedPoint } from '../lib/points'

interface Props {
  title: string
  /** The two pairs, player ids: key 1 is the first, 2 the second. Null when
   *  the pairs are not known (the winner cannot be said). */
  pairs: [[number, number], [number, number]] | null
  names: string[]
  videoRef: RefObject<HTMLVideoElement>
  points: MarkedPoint[]
  onChange: (points: MarkedPoint[]) => void
  error: string | null
}

const FRAME_S = 1 / 30
const SPEEDS = [0.5, 1, 2]
const AROUND_S = 60

/**
 * Marking the points while watching the match: the truth the rallies are
 * measured against (backend/scripts/eval_points.py). B at the serve, F when
 * the point is decided, 1 or 2 for the pair that won it.
 */
export const PointMarker: FC<Props> = ({ title, pairs, names, videoRef, points, onChange, error }) => {
  const [now, setNow] = useState(0)
  const [speed, setSpeed] = useState(1)
  const [start, setStart] = useState<number | null>(null)
  const [problem, setProblem] = useState<string | null>(null)
  const [copied, setCopied] = useState<string | null>(null)
  const name = (pid: number) => names[pid] || `Giocatore ${pid + 1}`
  const pairName = (pair: [number, number]) => `${name(pair[0])} e ${name(pair[1])}`

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
      const key = e.key.toLowerCase()
      const t = video.currentTime
      if (key === 'b') {
        e.preventDefault()
        setStart(t)
        setProblem(null)
      } else if (key === 'f') {
        e.preventDefault()
        if (start === null) {
          setProblem('Prima la battuta: B quando la palla lascia la racchetta di chi batte.')
          return
        }
        const added = addPoint(points, start, t)
        if ('error' in added) {
          setProblem(added.error)
          return
        }
        setStart(null)
        setProblem(null)
        onChange(added.points)
      } else if ((key === '1' || key === '2') && pairs) {
        e.preventDefault()
        const index = pointAt(points, t)
        if (index < 0) {
          setProblem('Nessun punto qui: il vincitore si segna durante il punto o subito dopo la fine.')
          return
        }
        setProblem(null)
        onChange(setWinner(points, index, pairs[key === '1' ? 0 : 1]))
      } else if (key === 'escape') {
        setStart(null)
        setProblem(null)
      } else if (e.key === ',' || e.key === '.') {
        e.preventDefault()
        video.pause()
        video.currentTime = Math.max(0, t + (e.key === '.' ? FRAME_S : -FRAME_S))
      } else if (e.key === ' ') {
        e.preventDefault()
        if (video.paused) void video.play()
        else video.pause()
      }
    }
    window.addEventListener('keydown', keys)
    return () => window.removeEventListener('keydown', keys)
  }, [points, onChange, pairs, start, videoRef])

  const seek = (t: number) => {
    const video = videoRef.current
    if (video) {
      video.pause()
      video.currentTime = t
    }
  }
  const near = points.filter(([s, e]) => e >= now - AROUND_S && s <= now + AROUND_S)
  const unwon = points.filter(p => !p[2]).length

  const copy = async () => {
    const text = pointsText(title, points, name)
    try {
      await navigator.clipboard.writeText(text)
      setCopied('Copiati: incollali nella conversazione, o salvali in un file per eval_points.py.')
    } catch {
      setCopied(text)
    }
  }

  return (
    <div className="hit-marker">
      <p className="muted-note" style={{ marginTop: 0 }}>
        <strong>B</strong> sul fotogramma della battuta (la palla lascia la racchetta; se la prima è fallo,
        dalla seconda), <strong>F</strong> quando il punto è deciso (secondo rimbalzo, palla fuori o in
        rete, ultimo colpo mancato). {pairs ? <><strong>1</strong> o <strong>2</strong> per la coppia che
        l'ha vinto, durante il punto o subito dopo. </> : null}
        <strong>Esc</strong> annulla una battuta segnata. Segna <em>tutti</em> i punti di un tratto della
        partita, uno dopo l'altro: uno scambio fra due punti segnati che non è un punto conta come errore
        dell'analisi.
      </p>
      <div className="hit-keys">
        {pairs && pairs.map((pair, i) => (
          <span key={pair.join()} className="hit-key">
            <kbd>{i + 1}</kbd>
            {pair.map(pid => (
              <span key={pid} className="player-dot" style={{ background: PLAYER_COLORS[pid % 4] }} />
            ))}
            {pairName(pair)}
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
      {start !== null && (
        <p className="muted-note" role="status">
          Battuta a {stamp(start)}: premi <strong>F</strong> alla fine del punto.
        </p>
      )}
      {problem && <p className="muted-note" role="alert">{problem}</p>}
      <div className="hit-list">
        <span className="muted-note">
          Segnati {points.length}{unwon && pairs ? ` (${unwon} senza vincitore)` : ''} · qui vicino ({stamp(now)}):
        </span>
        {near.length === 0 && <span className="muted-note">nessuno</span>}
        {near.map(([s, e, w]) => (
          <span key={s} className="hit-chip">
            <button type="button" className="hit-chip-time" onClick={() => seek(s)}>
              {stamp(s)}–{stamp(e)}
              {w ? ` · ${pairName(w)}` : ''}
            </button>
            <button
              type="button"
              className="hit-chip-x"
              aria-label={`Togli il punto da ${stamp(s)} a ${stamp(e)}`}
              onClick={() => onChange(points.filter(p => p[0] !== s))}
            >
              ×
            </button>
          </span>
        ))}
      </div>
      {points.length > 0 && (
        <button type="button" className="btn btn-ghost btn-sm" onClick={copy}>
          Copia i punti segnati
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
