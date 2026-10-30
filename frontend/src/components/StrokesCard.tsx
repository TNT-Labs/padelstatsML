import { useState, type FC, type ReactNode } from 'react'
import { api, type MatchStats, type Stroke, type StrokeLabel } from '../api'
import { STROKE_LABELS, strokeCheck, strokeName } from '../lib/format'
import { PLAYER_COLORS } from '../lib/players'
import { clock } from '../lib/videoOverlay'
import { InfoHeading } from './InfoHeading'

// A stroke is seen a moment before it: the swing, the ball arriving.
const STROKE_LEAD_S = 1.5

interface Props {
  title: string
  /** What the strokes are and how far they are to be trusted. */
  info: ReactNode
  stats: MatchStats
  /** Per player, in the order of `players`: the racket hand's strokes, or null. */
  strokes: (Stroke[] | null)[]
  players: string[]
  names: string[]
  readOnly: boolean
  onSeek: (seconds: number) => void
  /** What to say for a player with none, by position: "nessuno" by default. */
  emptyNote?: (index: number) => string
}

/**
 * Strokes found from the posture, to watch and to check: the high ones
 * (smash and bandeja, lob) and the forehands and backhands. Checking them
 * stroke by stroke measures the precision on each match and gives the cases
 * to tune on.
 */
export const StrokesCard: FC<Props> = ({
  title, info, stats, strokes, players, names, readOnly, onSeek, emptyNote,
}) => {
  const [checking, setChecking] = useState(false)
  const [labels, setLabels] = useState<Record<string, StrokeLabel>>(stats.stroke_labels ?? {})
  const [error, setError] = useState<string | null>(null)
  const [copied, setCopied] = useState<string | null>(null)

  const all = players.flatMap((pid, i) => (strokes[i] ?? []).map(s => ({ ...s, key: `${pid}:${s.t}`, i })))
  const check = strokeCheck(all, labels)

  const label = async (key: string, value: StrokeLabel | null) => {
    setError(null)
    const before = labels
    const next = { ...labels }
    if (value) next[key] = value
    else delete next[key]
    setLabels(next)
    try {
      await api.labelStrokes(stats.match_id, { [key]: value })
    } catch (e) {
      setLabels(before)
      setError((e as Error).message)
    }
  }

  const copy = async () => {
    const lines = all
      .filter(s => labels[s.key])
      .map(s => `${names[s.i] || `G${s.i + 1}`} ${clock(s.t)} ${strokeName(s)} -> ${labels[s.key]}`)
    const text = [`${stats.title} · ${title.toLowerCase()} verificati ${check.checked}`, ...lines].join('\n')
    try {
      await navigator.clipboard.writeText(text)
      setCopied('Copiate: incollale nella conversazione.')
    } catch {
      // Without the clipboard (http on the LAN), the text to copy by hand.
      setCopied(text)
    }
  }

  return (
    <div className="card" style={{ marginBottom: '1.5rem' }}>
      <InfoHeading title={title}>
        <p style={{ margin: 0 }}>{info}</p>
      </InfoHeading>

      <div className="stroke-tools">
        {check.checked > 0 && (
          <span className="muted-note">
            Verificati {check.checked}: colpi veri {Math.round((100 * check.real) / check.checked)}% · tipo
            giusto {Math.round((100 * check.rightKind) / check.checked)}%
          </span>
        )}
        {!readOnly && (
          <button
            type="button"
            className={`btn btn-sm ${checking ? 'btn-primary' : 'btn-ghost'}`}
            aria-pressed={checking}
            onClick={() => setChecking(c => !c)}
          >
            Verifica
          </button>
        )}
        {!readOnly && check.checked > 0 && (
          <button type="button" className="btn btn-ghost btn-sm" onClick={copy}>
            Copia le verifiche
          </button>
        )}
      </div>
      {error && (
        <div className="callout callout-error" role="alert" style={{ marginBottom: '.75rem' }}>
          {error}
        </div>
      )}
      {copied && (copied.includes('\n') ? (
        <textarea className="stroke-copy" readOnly value={copied} onFocus={e => e.target.select()} />
      ) : (
        <p className="muted-note">{copied}</p>
      ))}

      {players.map((pid, index) => {
        const list = strokes[index]
        if (!list) return null
        return (
          <div key={pid} className="stroke-row">
            <div className="stroke-player">
              <span className="player-dot" style={{ background: PLAYER_COLORS[index % 4] }} />
              {names[index] || `Giocatore ${index + 1}`}
              <span className="muted-note">· {list.length}</span>
            </div>
            <div className="rally-strip">
              {list.map(stroke => {
                const key = `${pid}:${stroke.t}`
                const seen = labels[key]
                return (
                  <div key={key} className={`stroke-item${seen ? ' stroke-item-checked' : ''}`}>
                    <button
                      type="button"
                      className="rally-chip"
                      onClick={() => onSeek(Math.max(0, stroke.t - STROKE_LEAD_S))}
                      title={`Guarda il colpo, al minuto ${clock(stroke.t)}`}
                    >
                      <span className="rally-num">{clock(stroke.t)}</span>
                      <span className="rally-dur">{strokeName(stroke)}</span>
                    </button>
                    {checking && (
                      <select
                        className="stroke-label"
                        value={seen ?? ''}
                        aria-label={`Cosa è il colpo delle ${clock(stroke.t)}`}
                        onChange={e => label(key, (e.target.value || null) as StrokeLabel | null)}
                      >
                        <option value="">— da vedere</option>
                        {Object.entries(STROKE_LABELS).map(([value, text]) => (
                          <option key={value} value={value}>
                            {text}
                          </option>
                        ))}
                      </select>
                    )}
                    {!checking && seen && <span className="stroke-seen">{STROKE_LABELS[seen]}</span>}
                  </div>
                )
              })}
              {list.length === 0 && <span className="muted-note">{emptyNote ? emptyNote(index) : 'nessuno'}</span>}
            </div>
          </div>
        )
      })}
    </div>
  )
}
