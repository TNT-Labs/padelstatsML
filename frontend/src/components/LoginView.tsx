import { useState, type FC, type FormEvent } from 'react'
import { api, type Me } from '../api'

interface Props {
  onSignedIn: (me: Me) => void
}

export const LoginView: FC<Props> = ({ onSignedIn }) => {
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const submit = async (e: FormEvent) => {
    e.preventDefault()
    if (busy) return
    setBusy(true)
    setError(null)
    try {
      onSignedIn(await api.login(username.trim(), password))
    } catch (err) {
      setError((err as Error).message)
      setPassword('')
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="auth-page">
      <form className="card auth-card" onSubmit={submit}>
        <div className="auth-logo" aria-hidden>
          🎾
        </div>
        <h1>Padel Stats</h1>
        <p className="muted-note" style={{ marginBottom: '1.25rem' }}>
          Accedi con le credenziali ricevute dall'amministratore.
        </p>

        {error && (
          <div className="callout callout-error" role="alert" style={{ marginBottom: '1rem' }}>
            {error}
          </div>
        )}

        <div className="form-group">
          <label htmlFor="login-user">Nome utente</label>
          <input
            id="login-user"
            type="text"
            autoComplete="username"
            autoCapitalize="none"
            autoCorrect="off"
            spellCheck={false}
            value={username}
            onChange={e => setUsername(e.target.value)}
            required
            autoFocus
          />
        </div>
        <div className="form-group">
          <label htmlFor="login-pass">Password</label>
          <input
            id="login-pass"
            type="password"
            autoComplete="current-password"
            value={password}
            onChange={e => setPassword(e.target.value)}
            required
          />
        </div>
        <button
          className="btn btn-primary"
          style={{ width: '100%' }}
          disabled={busy || !username.trim() || !password}
        >
          {busy ? 'Accesso…' : 'Accedi'}
        </button>
      </form>
    </div>
  )
}
