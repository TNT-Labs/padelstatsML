import { useEffect, useState, type FC, type RefObject } from 'react'
import { PLAYER_COLORS } from '../lib/players'
import { HIT_KINDS, hitsText, kindLabel, kindOfKey, nextUnkinded, setKind, type MarkedHit } from '../lib/hitTypes'

interface Props {
  title: string
  /** Player ids in order: key 1 is the first, 2 the second… */
  players: number[]
  names: string[]
  videoRef: RefObject<HTMLVideoElement>
  hits: MarkedHit[]
  onChange: (hits: MarkedHit[]) => void
  error: string | null
}

// One frame of a 30 fps video, for stepping.
const FRAME_S = 1 / 30
const SPEEDS = [0.25, 0.5, 1]
// Hits listed around the current time.
const AROUND_S = 8
// N shows the next hit from this long before it.
const LEAD_S = 1.5
// A kind key with no hit chosen goes to the hit this near the current time.
const NEAR_S = 1.5

function stamp(t: number): string {
  const m = Math.floor(t / 60)
  return `${m}:${(t - 60 * m).toFixed(2).padStart(5, '0')}`
}

/**
 * Marking the hits while watching the match: the truth the hits found from
 * the ball are measured against, and the kinds of stroke are learnt from.
 * Pause on the frame where the ball leaves the racket and press the
 * player's number, then the kind's letter; N goes to the next hit marked
 * without a kind, to give the kind to hits marked before.
 */
export const HitMarker: FC<Props> = ({ title, players, names, videoRef, hits, onChange, error }) => {
  const [now, setNow] = useState(0)
  const [speed, setSpeed] = useState(1)
  const [chosen, setChosen] = useState<number | null>(null)
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
      const kind = kindOfKey(e.key)
      if (index >= 0 && index < players.length) {
        e.preventDefault()
        const t = Math.round(video.currentTime * 100) / 100
        const next = [...hits.filter(h => h[0] !== t), [t, players[index]] as MarkedHit].sort((a, b) => a[0] - b[0])
        onChange(next)
        setChosen(t)
      } else if (kind) {
        e.preventDefault()
        // The hit chosen, or else the one nearest the current time.
        const near = hits.reduce<MarkedHit | null>((best, h) =>
          Math.abs(h[0] - video.currentTime) <= NEAR_S &&
          (!best || Math.abs(h[0] - video.currentTime) < Math.abs(best[0] - video.currentTime)) ? h : best, null)
        const t = chosen !== null && hits.some(h => h[0] === chosen) ? chosen : near?.[0]
        if (t === undefined) return
        onChange(setKind(hits, t, kind))
        setChosen(t)
      } else if (e.key === 'n' || e.key === 'N') {
        e.preventDefault()
        const next = nextUnkinded(hits, chosen ?? video.currentTime)
        if (!next) return
        setChosen(next[0])
        video.currentTime = Math.max(0, next[0] - LEAD_S)
        void video.play()
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
  }, [hits, onChange, players, videoRef, chosen])

  const near = hits.filter(([t]) => Math.abs(t - now) <= AROUND_S)
  const kinded = hits.filter(h => h.length === 3).length
  const chosenHit = hits.find(h => h[0] === chosen)

  const copy = async () => {
    const text = hitsText(title, hits, name, stamp)
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
        Metti in pausa sul fotogramma in cui la palla lascia la racchetta, premi il numero del
        giocatore e poi la lettera del tipo di colpo. <strong>N</strong> va al prossimo colpo segnato
        ancora senza tipo (un secondo e mezzo prima, e fa partire il video): la lettera gli dà il tipo.
        <strong> ,</strong> e <strong>.</strong> tornano indietro e avanzano di un fotogramma, la{' '}
        <strong>barra spaziatrice</strong> avvia e ferma. Segna tutti i colpi di qualche scambio intero,
        battuta compresa. Un lob o un'uscita di parete si segnano così anche se di dritto o di rovescio.
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
      <div className="hit-keys">
        {HIT_KINDS.map(k => (
          <span key={k.kind} className="hit-key">
            <kbd>{k.key.toUpperCase()}</kbd>
            {k.label}
          </span>
        ))}
      </div>
      {chosenHit && (
        <p className="muted-note" role="status">
          Colpo scelto: {stamp(chosenHit[0])} · {name(chosenHit[1])}
          {chosenHit.length === 3 ? ` · ${kindLabel(chosenHit[2])}` : ' · senza tipo: premi la lettera'}
        </p>
      )}
      <div className="hit-list">
        <span className="muted-note">
          Segnati {hits.length}{hits.length > 0 && `, con il tipo ${kinded}`} · qui vicino ({stamp(now)}):
        </span>
        {near.length === 0 && <span className="muted-note">nessuno</span>}
        {near.map(h => (
          <span
            key={`${h[0]}:${h[1]}`}
            className={`hit-chip${h[0] === chosen ? ' hit-chip-chosen' : ''}`}
            style={{ borderColor: PLAYER_COLORS[h[1] % 4] }}
          >
            <button
              type="button"
              className="hit-chip-time"
              onClick={() => {
                setChosen(h[0])
                const video = videoRef.current
                if (video) {
                  video.pause()
                  video.currentTime = h[0]
                }
              }}
            >
              {stamp(h[0])} {name(h[1])}
              {h.length === 3 && ` · ${kindLabel(h[2])}`}
            </button>
            <button
              type="button"
              className="hit-chip-x"
              aria-label={`Togli il colpo di ${name(h[1])} a ${stamp(h[0])}`}
              onClick={() => {
                onChange(hits.filter(x => !(x[0] === h[0] && x[1] === h[1])))
                if (chosen === h[0]) setChosen(null)
              }}
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
