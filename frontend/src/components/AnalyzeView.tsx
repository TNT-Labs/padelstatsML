import { useRef, useState, DragEvent, ChangeEvent, type FC } from 'react'
import { useMatchAnalysis } from '../hooks/useMatchAnalysis'
import type { MatchStats } from '../api'

interface Props {
  onDone: (stats: MatchStats) => void
  onBack: () => void
}

const PHASE_LABEL: Record<string, string> = {
  creating:   'Creazione partita…',
  uploading:  'Caricamento video…',
  processing: 'Analisi ML in corso…',
}

export const AnalyzeView: FC<Props> = ({ onDone, onBack }) => {
  const { state, analyze, reset } = useMatchAnalysis()
  const [title, setTitle] = useState('Partita')
  const [dragOver, setDragOver] = useState(false)
  const [file, setFile] = useState<File | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleFile = (f: File) => {
    if (!f.type.startsWith('video/')) return
    setFile(f)
  }
  const onDrop = (e: DragEvent) => {
    e.preventDefault(); setDragOver(false)
    const f = e.dataTransfer.files[0]
    if (f) handleFile(f)
  }
  const onFileInput = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.[0]) handleFile(e.target.files[0])
  }

  if (state.phase === 'done' && state.stats) {
    onDone(state.stats)
    return null
  }

  return (
    <div className="layout" style={{ maxWidth: 560 }}>
      <div className="header">
        <h1>🎾 Nuova analisi</h1>
        <button className="btn btn-ghost btn-sm" onClick={onBack}>← Indietro</button>
      </div>

      {/* ── Idle / form ─────────────────────────────────── */}
      {state.phase === 'idle' && (
        <>
          <div className="card" style={{ marginBottom: '1rem' }}>
            <div className="form-group" style={{ marginBottom: 0 }}>
              <label>Titolo partita</label>
              <input
                type="text"
                value={title}
                onChange={e => setTitle(e.target.value)}
                placeholder="es. Mercoledì sera"
              />
            </div>
          </div>

          <div
            className={`dropzone ${dragOver ? 'active' : ''}`}
            onClick={() => fileInputRef.current?.click()}
            onDragOver={e => { e.preventDefault(); setDragOver(true) }}
            onDragLeave={() => setDragOver(false)}
            onDrop={onDrop}
            style={{ marginBottom: '1rem' }}
          >
            <div style={{ fontSize: '2.5rem' }}>📹</div>
            {file ? (
              <p style={{ color: 'var(--green)', fontWeight: 600 }}>{file.name}</p>
            ) : (
              <>
                <p style={{ fontWeight: 600 }}>Trascina il video qui o clicca per scegliere</p>
                <p>MP4 / MOV · max 2GB · inquadratura fissa dall&apos;alto</p>
              </>
            )}
            <input
              ref={fileInputRef}
              type="file"
              accept="video/*"
              style={{ display: 'none' }}
              onChange={onFileInput}
            />
          </div>

          <button
            className="btn btn-primary"
            style={{ width: '100%' }}
            disabled={!file || !title.trim()}
            onClick={() => file && analyze(file, title)}
          >
            Avvia analisi
          </button>
        </>
      )}

      {/* ── In progress ─────────────────────────────────── */}
      {(state.phase === 'creating' || state.phase === 'uploading' || state.phase === 'processing') && (
        <div className="card" style={{ textAlign: 'center', padding: '2.5rem 1.5rem' }}>
          <div style={{ fontSize: '2rem', marginBottom: '1rem' }}>⏳</div>
          <h2 style={{ marginBottom: '.75rem' }}>{PHASE_LABEL[state.phase]}</h2>
          <div className="progress-track" style={{ maxWidth: 360, margin: '0 auto' }}>
            <div className="progress-fill" style={{ width: `${state.progress * 100}%` }} />
          </div>
          <p style={{ color: 'var(--muted)', marginTop: '.5rem', fontSize: '.9rem' }}>
            {Math.round(state.progress * 100)}%
            {state.phase === 'processing' && " — l'analisi richiede 5-20 min"}
          </p>
        </div>
      )}

      {/* ── Error ───────────────────────────────────────── */}
      {state.phase === 'error' && (
        <div className="card" style={{ textAlign: 'center' }}>
          <p style={{ fontSize: '1.5rem', marginBottom: '.5rem' }}>⚠️</p>
          <p className="error-msg" style={{ marginBottom: '1rem' }}>{state.error}</p>
          <button className="btn btn-primary" onClick={reset}>Riprova</button>
        </div>
      )}
    </div>
  )
}
