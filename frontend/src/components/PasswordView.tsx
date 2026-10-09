import { useState, type FC, type FormEvent } from 'react'
import { api, type Me } from '../api'

interface Props {
  me: Me
  /** First sign-in or after a reset: there is no way around it. */
  forced?: boolean
  onDone: (me: Me) => void
  onCancel?: () => void
  onLogout?: () => void
}

export const PasswordView: FC<Props> = ({ me, forced, onDone, onCancel, onLogout }) => {
  const [current, setCurrent] = useState('')
  const [next, setNext] = useState('')
  const [repeat, setRepeat] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const mismatch = repeat.length > 0 && next !== repeat

  const submit = async (e: FormEvent) => {
    e.preventDefault()
    if (busy || mismatch) return
    setBusy(true)
    setError(null)
    try {
      onDone(await api.changePassword(current, next))
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className={forced ? 'auth-page' : 'layout'} style={forced ? undefined : { maxWidth: 480 }}>
      <form className={`card ${forced ? 'auth-card' : ''}`} onSubmit={submit}>
        <h1 style={{ marginBottom: '.4rem' }}>{forced ? 'Scegli la tua password' : 'Cambia password'}</h1>
        <p className="muted-note" style={{ marginBottom: '1.25rem' }}>
          {forced
            ? `Ciao ${me.username}: la password che hai usato è provvisoria. Scegline una tua per continuare.`
            : 'Le altre sessioni aperte con il tuo account verranno chiuse.'}
        </p>

        {error && (
          <div className="callout callout-error" role="alert" style={{ marginBottom: '1rem' }}>
            {error}
          </div>
        )}

        {/* Lets password managers attach the new password to the right account. */}
        <input type="text" autoComplete="username" value={me.username} readOnly hidden />
        <div className="form-group">
          <label htmlFor="pw-current">{forced ? 'Password provvisoria' : 'Password attuale'}</label>
          <input
            id="pw-current"
            type="password"
            autoComplete="current-password"
            value={current}
            onChange={e => setCurrent(e.target.value)}
            required
            autoFocus
          />
        </div>
        <div className="form-group">
          <label htmlFor="pw-new">Nuova password</label>
          <input
            id="pw-new"
            type="password"
            autoComplete="new-password"
            value={next}
            onChange={e => setNext(e.target.value)}
            required
          />
          <p className="muted-note" style={{ marginTop: '.3rem' }}>
            Almeno 10 caratteri, con lettere e cifre, senza il nome utente.
          </p>
        </div>
        <div className="form-group">
          <label htmlFor="pw-repeat">Ripeti la nuova password</label>
          <input
            id="pw-repeat"
            type="password"
            autoComplete="new-password"
            value={repeat}
            onChange={e => setRepeat(e.target.value)}
            aria-invalid={mismatch}
            required
          />
          {mismatch && <p className="error-msg">Le due password non coincidono.</p>}
        </div>

        <div className="action-row" style={{ justifyContent: 'space-between' }}>
          {forced ? (
            <button type="button" className="btn btn-ghost" onClick={onLogout}>
              Esci
            </button>
          ) : (
            <button type="button" className="btn btn-ghost" onClick={onCancel}>
              Annulla
            </button>
          )}
          <button
            className="btn btn-primary"
            disabled={busy || !current || !next || next !== repeat}
          >
            {busy ? 'Salvataggio…' : 'Salva password'}
          </button>
        </div>
      </form>
    </div>
  )
}
