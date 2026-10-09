import { useRef, useState, type ChangeEvent, type DragEvent, type FC } from 'react'
import type { Match } from '../api'
import { useUpload } from '../hooks/useMatch'

interface Props {
  onUploaded: (matchId: string) => void
  onBack: () => void
  /** An upload left unfinished: the same file is chosen again and only the
   *  missing part is sent. */
  resume?: Match
}

export const UploadView: FC<Props> = ({ onUploaded, onBack, resume }) => {
  const { state, upload, reset } = useUpload()
  const [title, setTitle] = useState(resume?.title ?? 'Partita')
  const [file, setFile] = useState<File | null>(null)
  const [dragOver, setDragOver] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  const accept = (candidate: File | undefined) => {
    if (candidate && candidate.type.startsWith('video/')) setFile(candidate)
  }

  const onDrop = (e: DragEvent) => {
    e.preventDefault()
    setDragOver(false)
    accept(e.dataTransfer.files[0])
  }

  const onPick = (e: ChangeEvent<HTMLInputElement>) => accept(e.target.files?.[0])

  const start = async () => {
    if (!file) return
    // After a failure the match already exists: carry on with it rather
    // than leave an unfinished one behind.
    const matchId = await upload(file, title.trim(), state.matchId ?? resume?.id)
    if (matchId) onUploaded(matchId)
  }

  const busy = state.phase === 'creating' || state.phase === 'uploading' || state.phase === 'checking'

  return (
    <div className="layout" style={{ maxWidth: 620 }}>
      <div className="header">
        <h1>{resume ? 'Riprendi caricamento' : 'Nuova partita'}</h1>
        <button className="btn btn-ghost btn-sm" onClick={onBack} disabled={busy}>
          ← Indietro
        </button>
      </div>

      {state.phase === 'error' && (
        <div className="callout callout-error" style={{ marginBottom: '1rem' }}>
          {state.error}
          <button className="btn btn-ghost btn-sm" style={{ marginLeft: '.75rem' }} onClick={reset}>
            Riprova
          </button>
        </div>
      )}

      {!busy && resume && (
        <div className="callout callout-info" style={{ marginBottom: '1rem' }}>
          Il caricamento di <strong>{resume.title}</strong> si è interrotto. Scegli di nuovo lo
          stesso video: verrà inviata solo la parte mancante.
        </div>
      )}

      {!busy && (
        <>
          <div className="card" style={{ marginBottom: '1rem' }} hidden={!!resume}>
            <div className="form-group" style={{ marginBottom: 0 }}>
              <label htmlFor="match-title">Titolo partita</label>
              <input
                id="match-title"
                type="text"
                value={title}
                onChange={e => setTitle(e.target.value)}
                placeholder="es. Mercoledì sera"
              />
            </div>
          </div>

          <div
            className={`dropzone ${dragOver ? 'active' : ''}`}
            onClick={() => inputRef.current?.click()}
            onDragOver={e => {
              e.preventDefault()
              setDragOver(true)
            }}
            onDragLeave={() => setDragOver(false)}
            onDrop={onDrop}
          >
            <div style={{ fontSize: '2.5rem' }}>🎥</div>
            {file ? (
              <p style={{ color: 'var(--green)', fontWeight: 600 }}>
                {file.name} · {(file.size / 1_048_576).toFixed(0)} MB
              </p>
            ) : (
              <>
                <p style={{ fontWeight: 600 }}>Trascina il video o tocca per sceglierlo</p>
                <p>MP4 / MOV · camera fissa che inquadra tutto il campo</p>
              </>
            )}
            <input
              ref={inputRef}
              type="file"
              accept="video/*"
              style={{ display: 'none' }}
              onChange={onPick}
            />
          </div>

          <div className="callout callout-info" style={{ margin: '1rem 0' }}>
            Dopo il caricamento ti verrà chiesto di indicare i quattro angoli del campo.
            Serve una volta sola per ogni posizione della camera.
          </div>

          <button
            className="btn btn-primary"
            style={{ width: '100%' }}
            disabled={!file || !title.trim()}
            onClick={start}
          >
            {resume ? 'Riprendi caricamento' : 'Carica video'}
          </button>
        </>
      )}

      {busy && (
        <div className="card" style={{ textAlign: 'center', padding: '2.5rem 1.5rem' }}>
          <h2 style={{ marginBottom: '.75rem' }}>
            {state.phase === 'creating'
              ? 'Preparazione…'
              : state.phase === 'checking'
                ? 'Verifica del video…'
                : 'Caricamento video…'}
          </h2>
          <div className="progress-track">
            <div className="progress-fill" style={{ width: `${state.progress * 100}%` }} />
          </div>
          <p className="muted-note" style={{ marginTop: '.5rem' }}>
            {state.phase === 'checking'
              ? 'Lettura del video sul server, qualche secondo…'
              : `${Math.round(state.progress * 100)}% — non chiudere questa pagina`}
          </p>
          {state.retrying !== null && (
            <p className="callout callout-warn" style={{ marginTop: '.75rem' }}>
              Connessione instabile: nuovo tentativo ({state.retrying}) tra pochi secondi…
            </p>
          )}
        </div>
      )}
    </div>
  )
}
