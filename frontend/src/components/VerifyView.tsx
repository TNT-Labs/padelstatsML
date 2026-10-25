/**
 * Accuracy check: how many metres the analysis is off, against points a
 * person marks by hand (see backend app/ml/verification.py).
 *
 *   1. Court landmarks on the keyframe — service lines, the T, the foot of
 *      the net posts: the calibration's error, near and far.
 *   2. The players' feet in a few frames of the video: the error of the
 *      positions the statistics are computed from, players missed, players
 *      placed where nobody is.
 */
import { useCallback, useEffect, useRef, useState, type FC, type PointerEvent } from 'react'
import {
  api,
  ApiError,
  type ErrorSummary,
  type Landmark,
  type Point,
  type RefineProposal,
  type VerificationPlan,
  type VerificationResult,
} from '../api'
import { COURT_LENGTH_M, COURT_WIDTH_M, NET_Y_M, SERVICE_OFFSET_M } from '../lib/court'
import { clock } from '../lib/videoOverlay'

interface Props {
  matchId: string
  title: string
  onBack: (changed: boolean) => void
}

type Step = 'intro' | 'landmarks' | 'players' | 'result'

export const VerifyView: FC<Props> = ({ matchId, title, onBack }) => {
  const [plan, setPlan] = useState<VerificationPlan | null>(null)
  const [step, setStep] = useState<Step>('intro')
  const [marks, setMarks] = useState<Record<string, Point>>({})
  const [landmarkIndex, setLandmarkIndex] = useState(0)
  const [feet, setFeet] = useState<Point[][]>([])
  const [frameIndex, setFrameIndex] = useState(0)
  const [result, setResult] = useState<VerificationResult | null>(null)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [changed, setChanged] = useState(false)
  const [notice, setNotice] = useState<string | null>(null)

  useEffect(() => {
    api
      .getVerification(matchId)
      .then(p => {
        setPlan(p)
        setFeet(p.frames.map(() => []))
        setResult(p.last)
      })
      .catch((e: Error) => setError(e.message))
  }, [matchId])

  const submit = async (withPlayers: boolean) => {
    if (!plan) return
    setSaving(true)
    setError(null)
    try {
      const outcome = await api.submitVerification(matchId, {
        landmarks: marks,
        frames: withPlayers ? plan.frames.map((f, i) => ({ frame_index: f.frame_index, feet: feet[i] })) : [],
      })
      setResult(outcome)
      setChanged(true)
      setStep('result')
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e))
    } finally {
      setSaving(false)
    }
  }

  const header = (
    <div className="header">
      <h1>Verifica accuratezza</h1>
      <button className="btn btn-ghost btn-sm" onClick={() => onBack(changed)} disabled={saving}>
        ← Statistiche
      </button>
    </div>
  )

  if (!plan) {
    return (
      <div className="layout">
        {header}
        <div className="card">
          {error ? <p className="error-msg">{error}</p> : <p className="muted-note">Preparazione della verifica…</p>}
        </div>
      </div>
    )
  }

  const [frameW, frameH] = plan.frame_size

  if (step === 'intro') {
    return (
      <div className="layout">
        {header}
        <div className="card" style={{ marginBottom: '1rem' }}>
          <h2 style={{ marginBottom: '.5rem' }}>{title}</h2>
          <p style={{ marginBottom: '.6rem' }}>
            Quanti metri sbaglia l'analisi? Lo misuri segnando a mano alcuni punti: l'app li confronta con
            quello che ha calcolato.
          </p>
          <ol className="verify-steps">
            <li>
              <strong>Linee del campo</strong> ({plan.landmarks.length} punti): tocchi sul fotogramma della
              calibrazione gli incroci indicati. Misura l'errore della calibrazione, vicino e lontano.
            </li>
            <li>
              <strong>Giocatori</strong> ({plan.frames.length} fotogrammi): tocchi il punto a terra fra i piedi
              di ogni giocatore in campo. Misura l'errore delle posizioni su cui sono calcolate le statistiche.
            </li>
          </ol>
          <p className="muted-note">
            Servono pochi minuti. Da computer è più preciso: tocca il punto con cura, è quello che conta.
          </p>
          <div className="action-row" style={{ marginTop: '.8rem' }}>
            <button className="btn btn-primary" onClick={() => setStep('landmarks')}>
              Inizia
            </button>
          </div>
        </div>
        {result && <ResultCards result={result} />}
        {result && (
          <RefineCard
            matchId={matchId}
            result={result}
            onApplied={(fresh, namesKept) => {
              setResult(fresh)
              setChanged(true)
              setNotice(namesKept ? 'Calibrazione corretta e statistiche ricalcolate.'
                : 'Calibrazione corretta: i giocatori sono cambiati, i nomi vanno reinseriti.')
            }}
          />
        )}
        {notice && <div className="callout callout-info">{notice}</div>}
      </div>
    )
  }

  if (step === 'landmarks') {
    const landmark = plan.landmarks[landmarkIndex]
    const last = landmarkIndex === plan.landmarks.length - 1
    const next = () => (last ? setStep('players') : setLandmarkIndex(i => i + 1))
    return (
      <div className="layout">
        {header}
        <div className="verify-instruction card card-sm">
          <CourtDiagram target={landmark} />
          <div>
            <div className="muted-note">
              Linee del campo · punto {landmarkIndex + 1} di {plan.landmarks.length}
            </div>
            <strong>{landmark.label}</strong>
            <p className="muted-note" style={{ marginTop: '.25rem' }}>
              Tocca l'incrocio sul fotogramma; tocca di nuovo per correggere. Se non si vede, saltalo.
            </p>
          </div>
        </div>
        <MarkCanvas
          frameW={frameW}
          frameH={frameH}
          background={<image href={api.keyframeUrl(matchId)} x={0} y={0} width={frameW} height={frameH} />}
          points={marks[landmark.key] ? [marks[landmark.key]] : []}
          onMark={p => setMarks(m => ({ ...m, [landmark.key]: p }))}
          numbered={false}
        />
        <div className="action-row" style={{ marginTop: '.8rem' }}>
          <button
            className="btn btn-ghost btn-sm"
            onClick={() => (landmarkIndex === 0 ? setStep('intro') : setLandmarkIndex(i => i - 1))}
          >
            Indietro
          </button>
          <button
            className="btn btn-ghost btn-sm"
            onClick={() => {
              setMarks(m => {
                const copy = { ...m }
                delete copy[landmark.key]
                return copy
              })
              next()
            }}
          >
            Non visibile, salta
          </button>
          <button className="btn btn-primary btn-sm" disabled={!marks[landmark.key]} onClick={next}>
            Avanti
          </button>
        </div>
      </div>
    )
  }

  if (step === 'players') {
    return (
      <PlayersStep
        header={header}
        matchId={matchId}
        plan={plan}
        feet={feet}
        setFeet={setFeet}
        frameIndex={frameIndex}
        setFrameIndex={setFrameIndex}
        saving={saving}
        error={error}
        onBackToLandmarks={() => setStep('landmarks')}
        onDone={withPlayers => void submit(withPlayers)}
      />
    )
  }

  return (
    <div className="layout">
      {header}
      {error && <p className="error-msg">{error}</p>}
      {result && <ResultCards result={result} />}
      {result && (
        <RefineCard
          matchId={matchId}
          result={result}
          onApplied={(fresh, namesKept) => {
            setResult(fresh)
            setChanged(true)
            setNotice(
              namesKept
                ? 'Calibrazione corretta e statistiche ricalcolate.'
                : 'Calibrazione corretta e statistiche ricalcolate: i giocatori sono cambiati, i nomi vanno reinseriti.',
            )
          }}
        />
      )}
      {notice && <div className="callout callout-info" style={{ marginBottom: '1rem' }}>{notice}</div>}
      <div className="action-row" style={{ marginTop: '1rem' }}>
        <button className="btn btn-ghost btn-sm" onClick={() => setStep('intro')}>
          Ripeti la verifica
        </button>
        <button className="btn btn-primary btn-sm" onClick={() => onBack(changed)}>
          Torna alle statistiche
        </button>
      </div>
    </div>
  )
}

// ── Players ──────────────────────────────────────────────────────────────────

interface PlayersProps {
  header: JSX.Element
  matchId: string
  plan: VerificationPlan
  feet: Point[][]
  setFeet: (update: (prev: Point[][]) => Point[][]) => void
  frameIndex: number
  setFrameIndex: (update: (i: number) => number) => void
  saving: boolean
  error: string | null
  onBackToLandmarks: () => void
  onDone: (withPlayers: boolean) => void
}

const PlayersStep: FC<PlayersProps> = ({
  header, matchId, plan, feet, setFeet, frameIndex, setFrameIndex, saving, error, onBackToLandmarks, onDone,
}) => {
  const videoRef = useRef<HTMLVideoElement>(null)
  const [ready, setReady] = useState(false)
  const [broken, setBroken] = useState(false)
  const [frameW, frameH] = plan.frame_size
  const frame = plan.frames[frameIndex]
  const last = frameIndex === plan.frames.length - 1

  // Show the frame to mark: the video paused at its time. A hair past it,
  // so the browser lands inside the frame and not on the one before.
  useEffect(() => {
    const video = videoRef.current
    if (!video || !frame) return
    setReady(false)
    const seek = () => {
      video.currentTime = frame.t + 0.01
    }
    const done = () => setReady(true)
    video.addEventListener('seeked', done)
    if (video.readyState >= 1) seek()
    else video.addEventListener('loadedmetadata', seek, { once: true })
    return () => {
      video.removeEventListener('seeked', done)
      video.removeEventListener('loadedmetadata', seek)
    }
  }, [frame])

  if (broken) {
    return (
      <div className="layout">
        {header}
        <div className="card">
          <p style={{ marginBottom: '.6rem' }}>
            Il browser non riesce a riprodurre questo video (succede con i video H.265/HEVC): la parte sui
            giocatori non si può fare da qui. Puoi calcolare comunque l'errore della calibrazione.
          </p>
          {error && <p className="error-msg">{error}</p>}
          <div className="action-row">
            <button className="btn btn-ghost btn-sm" onClick={onBackToLandmarks}>Indietro</button>
            <button className="btn btn-primary btn-sm" disabled={saving} onClick={() => onDone(false)}>
              {saving ? 'Calcolo…' : 'Calcola solo la calibrazione'}
            </button>
          </div>
        </div>
      </div>
    )
  }

  const points = feet[frameIndex] ?? []
  return (
    <div className="layout">
      {header}
      <div className="card card-sm" style={{ marginBottom: '.75rem' }}>
        <div className="muted-note">
          Giocatori · fotogramma {frameIndex + 1} di {plan.frames.length} · minuto {clock(frame.t)}
        </div>
        <strong>Tocca il punto a terra fra i piedi di ogni giocatore in campo.</strong>
        <p className="muted-note" style={{ marginTop: '.25rem' }}>
          Anche se la rete o un altro giocatore li coprono: dove sono davvero. Gli spettatori non contano.
        </p>
      </div>
      <MarkCanvas
        frameW={frameW}
        frameH={frameH}
        video={
          <video
            ref={videoRef}
            className="verify-video"
            src={api.videoUrl(matchId)}
            muted
            playsInline
            preload="auto"
            onError={() => setBroken(true)}
          />
        }
        points={points}
        disabled={!ready}
        onMark={p => setFeet(prev => prev.map((f, i) => (i === frameIndex ? [...f, p].slice(0, 8) : f)))}
        numbered
      />
      {!ready && <p className="muted-note">Caricamento del fotogramma…</p>}
      {error && <p className="error-msg">{error}</p>}
      <div className="action-row" style={{ marginTop: '.8rem' }}>
        <button
          className="btn btn-ghost btn-sm"
          disabled={saving}
          onClick={() => (frameIndex === 0 ? onBackToLandmarks() : setFrameIndex(i => i - 1))}
        >
          Indietro
        </button>
        <button
          className="btn btn-ghost btn-sm"
          disabled={points.length === 0 || saving}
          onClick={() => setFeet(prev => prev.map((f, i) => (i === frameIndex ? f.slice(0, -1) : f)))}
        >
          Annulla l'ultimo
        </button>
        {last ? (
          <button className="btn btn-primary btn-sm" disabled={saving} onClick={() => onDone(true)}>
            {saving ? 'Calcolo…' : 'Calcola gli errori'}
          </button>
        ) : (
          <button className="btn btn-primary btn-sm" onClick={() => setFrameIndex(i => i + 1)}>
            {points.length ? 'Avanti' : 'Nessun giocatore, avanti'}
          </button>
        )}
      </div>
    </div>
  )
}

// ── Marking surface ──────────────────────────────────────────────────────────

interface CanvasProps {
  frameW: number
  frameH: number
  background?: JSX.Element
  video?: JSX.Element
  points: Point[]
  onMark: (p: Point) => void
  numbered: boolean
  disabled?: boolean
}

/** The picture with an SVG on top in its own pixels: the SVG letterboxes
 *  exactly as the picture does, so a tap converts through it. */
const MarkCanvas: FC<CanvasProps> = ({ frameW, frameH, background, video, points, onMark, numbered, disabled }) => {
  const svgRef = useRef<SVGSVGElement>(null)
  const mark = useCallback(
    (event: PointerEvent<SVGSVGElement>) => {
      const svg = svgRef.current
      const matrix = svg?.getScreenCTM()
      if (!svg || !matrix || disabled) return
      const point = svg.createSVGPoint()
      point.x = event.clientX
      point.y = event.clientY
      const local = point.matrixTransform(matrix.inverse())
      if (local.x < 0 || local.y < 0 || local.x > frameW || local.y > frameH) return
      onMark([Math.round(local.x * 10) / 10, Math.round(local.y * 10) / 10])
    },
    [disabled, frameW, frameH, onMark],
  )
  const r = Math.max(frameW, frameH) * 0.008
  return (
    <div className="verify-canvas">
      {video}
      <svg
        ref={svgRef}
        viewBox={`0 0 ${frameW} ${frameH}`}
        className={video ? 'verify-overlay' : 'verify-still'}
        onPointerDown={mark}
      >
        {background}
        {points.map((p, i) => (
          <g key={i} pointerEvents="none">
            <circle cx={p[0]} cy={p[1]} r={r * 2.2} fill="none" stroke="#000" strokeWidth={r * 0.9} />
            <circle cx={p[0]} cy={p[1]} r={r * 2.2} fill="none" stroke="#facc15" strokeWidth={r * 0.5} />
            <circle cx={p[0]} cy={p[1]} r={r * 0.45} fill="#facc15" />
            {numbered && (
              <text x={p[0] + r * 2.8} y={p[1] - r * 2} fontSize={r * 3.2} fill="#fff" stroke="#000"
                    strokeWidth={r * 0.35} paintOrder="stroke" fontWeight={700}>
                {i + 1}
              </text>
            )}
          </g>
        ))}
      </svg>
    </div>
  )
}

// ── Court diagram: where the landmark is ─────────────────────────────────────

const CourtDiagram: FC<{ target: Landmark }> = ({ target }) => {
  // Seen from the camera: the near baseline at the bottom.
  const s = 9
  const x = (m: number) => 6 + m * s
  const y = (m: number) => 6 + (COURT_LENGTH_M - m) * s
  const line = (x1: number, y1: number, x2: number, y2: number, key: string, strong = false) => (
    <line key={key} x1={x(x1)} y1={y(y1)} x2={x(x2)} y2={y(y2)}
          stroke={strong ? '#334155' : '#94a3b8'} strokeWidth={strong ? 2 : 1.2} />
  )
  return (
    <svg viewBox={`0 0 ${COURT_WIDTH_M * s + 12} ${COURT_LENGTH_M * s + 12}`} className="court-diagram" aria-hidden>
      <rect x={x(0)} y={y(COURT_LENGTH_M)} width={COURT_WIDTH_M * s} height={COURT_LENGTH_M * s}
            fill="#dbeafe" stroke="#334155" strokeWidth={1.5} />
      {line(0, NET_Y_M, COURT_WIDTH_M, NET_Y_M, 'net', true)}
      {line(0, NET_Y_M - SERVICE_OFFSET_M, COURT_WIDTH_M, NET_Y_M - SERVICE_OFFSET_M, 's1')}
      {line(0, NET_Y_M + SERVICE_OFFSET_M, COURT_WIDTH_M, NET_Y_M + SERVICE_OFFSET_M, 's2')}
      {line(COURT_WIDTH_M / 2, NET_Y_M - SERVICE_OFFSET_M, COURT_WIDTH_M / 2, NET_Y_M + SERVICE_OFFSET_M, 'c')}
      <circle cx={x(target.x_m)} cy={y(target.y_m)} r={6} fill="#f97316" stroke="#fff" strokeWidth={2} />
      <text x={x(COURT_WIDTH_M / 2)} y={y(0) + 4} fontSize={8} textAnchor="middle" fill="#64748b" dominantBaseline="hanging">
        camera
      </text>
    </svg>
  )
}

// ── Correcting the calibration ───────────────────────────────────────────────

const RefineCard: FC<{
  matchId: string
  result: VerificationResult
  onApplied: (result: VerificationResult, namesKept: boolean) => void
}> = ({ matchId, result, onApplied }) => {
  const [proposal, setProposal] = useState<RefineProposal | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const marks: Record<string, Point> = {}
  for (const item of result.calibration.items) if (item.px) marks[item.key] = item.px
  const count = Object.keys(marks).length

  if (result.refined) {
    return (
      <div className="callout callout-info" style={{ marginBottom: '1rem' }}>
        La calibrazione è stata corretta con questi punti: gli errori qui sopra sono misurati sugli stessi
        punti e quindi ottimistici. Per una misura indipendente ripeti la verifica.
      </div>
    )
  }
  if (count < 4) return null

  const propose = async (method: RefineProposal['method']) => {
    setBusy(true)
    setError(null)
    try {
      setProposal(await api.refineCalibration(matchId, marks, method))
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  const apply = async () => {
    if (!proposal) return
    setBusy(true)
    setError(null)
    try {
      const outcome = await api.applyRefinement(matchId, marks, proposal)
      const plan = await api.getVerification(matchId)
      if (plan.last) onApplied(plan.last, !outcome.names_cleared)
      setProposal(null)
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  const better = proposal && (proposal.after.mean_m ?? Infinity) < (proposal.before.mean_m ?? 0) - 0.05
  const lines = proposal?.method === 'lines'
  return (
    <div className="card" style={{ marginBottom: '1rem' }}>
      <h2 style={{ marginBottom: '.5rem' }}>Correggi la calibrazione</h2>
      <p className="muted-note" style={{ marginBottom: '.6rem' }}>
        La calibrazione usa solo i 4 angoli, spesso fuori inquadratura e messi a stima. Si può ricalcolare
        con i {count} punti che hai segnato, oppure agganciandola alle linee dipinte del fotogramma (i tuoi
        punti servono allora a misurarla): le statistiche si aggiornano in pochi secondi, senza rifare
        l'analisi.
      </p>
      {result.snapped && !proposal && (
        <p className="muted-note" style={{ marginBottom: '.6rem' }}>
          La calibrazione attuale è agganciata alle linee: gli errori qui sopra sono misurati sui tuoi punti,
          che l'aggancio non usa.
        </p>
      )}
      {!proposal && (
        <div className="action-row">
          <button className="btn btn-primary btn-sm" disabled={busy} onClick={() => propose('landmarks')}>
            {busy ? 'Calcolo…' : 'Correggi con i punti segnati'}
          </button>
          <button className="btn btn-ghost btn-sm" disabled={busy} onClick={() => propose('lines')}>
            Aggancia alle linee
          </button>
        </div>
      )}
      {proposal && (
        <>
          <table className="stat-table" style={{ marginBottom: '.6rem' }}>
            <tbody>
              <tr>
                <td>Errore attuale</td>
                <td>{proposal.before.mean_m?.toFixed(2)} m</td>
              </tr>
              <tr>
                <td>{lines ? "Errore con l'aggancio" : 'Errore atteso con la correzione'}</td>
                <td>{proposal.after.mean_m?.toFixed(2)} m</td>
              </tr>
              {lines && proposal.lines && (
                <tr>
                  <td>Campo sulle linee</td>
                  <td>
                    {Math.round(proposal.lines.aligned_before * 100)}% →{' '}
                    {Math.round(proposal.lines.aligned_after * 100)}%
                  </td>
                </tr>
              )}
              <tr>
                <td>Correzione grandangolo</td>
                <td>{proposal.k1 > 0 ? proposal.k1.toFixed(2) : 'non necessaria'}</td>
              </tr>
            </tbody>
          </table>
          <p className="muted-note" style={{ marginBottom: '.6rem' }}>
            {lines
              ? "L'aggancio non usa i punti che hai segnato: l'errore misurato su di essi è indipendente."
              : "L'errore atteso è misurato su ogni punto escluso a turno dal calcolo: non è ottimistico."}
          </p>
          {!better && (
            <p style={{ marginBottom: '.6rem' }}>La correzione non migliora la calibrazione: lasciala com'è.</p>
          )}
          <div className="action-row">
            {better && (
              <button className="btn btn-primary btn-sm" disabled={busy} onClick={apply}>
                {busy ? 'Ricalcolo delle statistiche…' : 'Applica e ricalcola le statistiche'}
              </button>
            )}
            <button className="btn btn-ghost btn-sm" disabled={busy} onClick={() => setProposal(null)}>
              Prova l'altro metodo
            </button>
          </div>
        </>
      )}
      {error && <p className="error-msg">{error}</p>}
    </div>
  )
}

// ── Results ──────────────────────────────────────────────────────────────────

/** In words, for a mean error in metres. */
function verdict(mean: number | undefined): string {
  if (mean === undefined) return ''
  if (mean <= 0.2) return 'ottimo'
  if (mean <= 0.4) return 'buono'
  if (mean <= 0.7) return 'discreto'
  return 'da migliorare'
}

const metres = (s: ErrorSummary | undefined) =>
  s && s.count ? `${s.mean_m?.toFixed(2)} m (mediana ${s.median_m?.toFixed(2)}, 90% entro ${s.p90_m?.toFixed(2)})` : '—'

export const ResultCards: FC<{ result: VerificationResult }> = ({ result }) => {
  const cal = result.calibration
  const players = result.players
  return (
    <>
      <div className="card" style={{ marginBottom: '1rem' }}>
        <h2 style={{ marginBottom: '.5rem' }}>Calibrazione</h2>
        {cal.all.count === 0 ? (
          <p className="muted-note">Nessun punto del campo segnato.</p>
        ) : (
          <>
            <p style={{ marginBottom: '.5rem' }}>
              Errore medio <strong>{cal.all.mean_m?.toFixed(2)} m</strong> — {verdict(cal.all.mean_m)}.
            </p>
            <table className="stat-table">
              <tbody>
                {(['vicina', 'rete', 'lontana'] as const).map(half =>
                  cal.by_half[half]?.count ? (
                    <tr key={half}>
                      <td>{half === 'rete' ? 'Rete' : `Metà ${half}`}</td>
                      <td>
                        {cal.by_half[half].mean_m?.toFixed(2)} m · {verdict(cal.by_half[half].mean_m)}
                      </td>
                    </tr>
                  ) : null,
                )}
              </tbody>
            </table>
            {cal.all.mean_m !== undefined && cal.all.mean_m > 0.4 && (
              <p className="muted-note" style={{ marginTop: '.5rem' }}>
                Riapri la calibrazione e sposta gli angoli finché le linee bianche coincidono con quelle reali,
                soprattutto nella metà con l'errore maggiore.
              </p>
            )}
          </>
        )}
      </div>
      {players && (
        <div className="card" style={{ marginBottom: '1rem' }}>
          <h2 style={{ marginBottom: '.5rem' }}>Giocatori</h2>
          <p style={{ marginBottom: '.5rem' }}>
            {players.marks} giocatori segnati in {players.frames} fotogrammi: {players.matched} trovati
            dall'analisi, {players.missed} mancati, {players.extra} collocati dove non c'era nessuno.
          </p>
          <table className="stat-table">
            <tbody>
              <tr>
                <td>Posizioni usate per le statistiche</td>
                <td>{metres(players.smoothed.all)}</td>
              </tr>
              <tr>
                <td>· metà vicina</td>
                <td>{metres(players.smoothed.by_half.vicina)}</td>
              </tr>
              <tr>
                <td>· metà lontana</td>
                <td>{metres(players.smoothed.by_half.lontana)}</td>
              </tr>
              <tr>
                <td>Rilevazioni grezze</td>
                <td>{metres(players.raw.all)}</td>
              </tr>
              <tr>
                <td>· metà vicina</td>
                <td>{metres(players.raw.by_half.vicina)}</td>
              </tr>
              <tr>
                <td>· metà lontana</td>
                <td>{metres(players.raw.by_half.lontana)}</td>
              </tr>
            </tbody>
          </table>
          <p className="muted-note" style={{ marginTop: '.5rem' }}>
            Qui la calibrazione non conta: i punti segnati e le posizioni calcolate passano per la stessa.
            Si misura il punto dei piedi trovato dall'analisi (rete che li copre, camera bassa, giocatori
            sovrapposti). L'errore totale in metri è circa la somma dei due.
          </p>
        </div>
      )}
      <p className="muted-note">Verifica del {new Date(result.at).toLocaleString('it-IT')}.</p>
    </>
  )
}
