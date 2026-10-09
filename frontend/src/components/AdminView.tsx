import { useCallback, useEffect, useState, type FC, type FormEvent } from 'react'
import { api, type AuditRow, type Me, type Role, type UserRow } from '../api'

interface Props {
  me: Me
  onBack: () => void
}

const EVENT_LABEL: Record<string, string> = {
  login: 'Accesso',
  login_fallito: 'Accesso fallito',
  logout: 'Uscita',
  cambio_password: 'Cambio password',
  cambio_password_fallito: 'Cambio password fallito',
  utente_creato: 'Utente creato',
  utente_modificato: 'Utente modificato',
  password_reimpostata: 'Password reimpostata',
  utente_sbloccato: 'Utente sbloccato',
  sessioni_chiuse: 'Sessioni chiuse',
  utente_eliminato: 'Utente eliminato',
  admin_ripristinato: 'Amministratore ripristinato',
}

const count = (n: number, one: string, many: string) => `${n} ${n === 1 ? one : many}`

const when = (iso: string | null) =>
  iso
    ? new Date(iso).toLocaleString('it-IT', {
        day: 'numeric',
        month: 'short',
        year: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
      })
    : 'mai'

/** Users and access log. Every rule (last administrator, own account, …)
 *  is enforced by the server; the buttons only avoid offering what it would
 *  refuse anyway. */
export const AdminView: FC<Props> = ({ me, onBack }) => {
  const [tab, setTab] = useState<'users' | 'audit'>('users')
  const [users, setUsers] = useState<UserRow[]>([])
  const [audit, setAudit] = useState<AuditRow[]>([])
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [newName, setNewName] = useState('')
  const [newRole, setNewRole] = useState<Role>('utente')
  const [secret, setSecret] = useState<{ username: string; password: string } | null>(null)
  const [confirmDelete, setConfirmDelete] = useState<string | null>(null)

  const loadUsers = useCallback(
    () => api.listUsers().then(setUsers).catch((e: Error) => setError(e.message)),
    [],
  )

  useEffect(() => {
    void loadUsers()
  }, [loadUsers])

  useEffect(() => {
    if (tab === 'audit') api.audit().then(setAudit).catch((e: Error) => setError(e.message))
  }, [tab])

  /** Run one action, then show the fresh list. */
  const act = async (action: () => Promise<unknown>) => {
    if (busy) return
    setBusy(true)
    setError(null)
    try {
      await action()
      await loadUsers()
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  const create = (e: FormEvent) => {
    e.preventDefault()
    void act(async () => {
      const created = await api.createUser(newName.trim(), newRole)
      setSecret({ username: created.user.username, password: created.temporary_password })
      setNewName('')
      setNewRole('utente')
    })
  }

  const reset = (user: UserRow) =>
    act(async () => {
      const result = await api.resetPassword(user.id)
      setSecret({ username: user.username, password: result.temporary_password })
    })

  return (
    <div className="layout" style={{ maxWidth: 860 }}>
      <div className="header">
        <h1>Gestione utenti</h1>
        <button className="btn btn-ghost btn-sm" onClick={onBack}>
          ← Torna alle partite
        </button>
      </div>

      <div className="tabs" role="tablist">
        <button role="tab" aria-selected={tab === 'users'} onClick={() => setTab('users')}>
          Utenti
        </button>
        <button role="tab" aria-selected={tab === 'audit'} onClick={() => setTab('audit')}>
          Registro accessi
        </button>
      </div>

      {error && (
        <div className="callout callout-error" role="alert" style={{ marginBottom: '1rem' }}>
          {error}
        </div>
      )}

      {secret && (
        <div className="callout callout-warn secret-box" style={{ marginBottom: '1rem' }}>
          <div>
            Password provvisoria di <strong>{secret.username}</strong>:
            <code className="secret">{secret.password}</code>
          </div>
          <div style={{ marginTop: '.35rem' }}>
            Comunicala all'utente: dovrà sceglierne una sua al primo accesso. Non verrà più
            mostrata.
          </div>
          <div className="action-row" style={{ marginTop: '.6rem' }}>
            <button
              className="btn btn-ghost btn-sm"
              onClick={() => void navigator.clipboard?.writeText(secret.password)}
            >
              Copia
            </button>
            <button className="btn btn-ghost btn-sm" onClick={() => setSecret(null)}>
              Fatto
            </button>
          </div>
        </div>
      )}

      {tab === 'users' && (
        <>
          <form className="card new-user" onSubmit={create} style={{ marginBottom: '1rem' }}>
            <div className="form-group" style={{ flex: 2 }}>
              <label htmlFor="nu-name">Nuovo utente</label>
              <input
                id="nu-name"
                type="text"
                placeholder="es. mario.rossi"
                autoCapitalize="none"
                autoCorrect="off"
                spellCheck={false}
                value={newName}
                onChange={e => setNewName(e.target.value)}
                maxLength={40}
              />
            </div>
            <div className="form-group" style={{ flex: 1 }}>
              <label htmlFor="nu-role">Ruolo</label>
              <select id="nu-role" value={newRole} onChange={e => setNewRole(e.target.value as Role)}>
                <option value="utente">Utente</option>
                <option value="admin">Amministratore</option>
              </select>
            </div>
            <button className="btn btn-primary" disabled={busy || newName.trim().length < 3}>
              Crea utente
            </button>
          </form>

          <div className="card">
            {users.map(user => {
              const self = user.id === me.id
              return (
                <div key={user.id} className="user-row">
                  <div style={{ minWidth: 0, flex: 1 }}>
                    <div className="match-title">
                      {user.username}
                      {self && <span className="muted-note"> (tu)</span>}
                    </div>
                    <div className="tag-row">
                      {user.role === 'admin' && <span className="badge badge-ready">Amministratore</span>}
                      {!user.active && <span className="badge badge-failed">Disattivato</span>}
                      {user.locked && <span className="badge badge-needs_calibration">Bloccato</span>}
                      {user.must_change_password && (
                        <span className="badge badge-uploading">Password provvisoria</span>
                      )}
                    </div>
                    <div className="muted-note">
                      Ultimo accesso: {when(user.last_login_at)} ·{' '}
                      {count(user.matches, 'partita', 'partite')}
                      {user.sessions > 0 &&
                        ` · ${count(user.sessions, 'sessione aperta', 'sessioni aperte')}`}
                    </div>
                  </div>
                  <div className="match-actions">
                    {!self && (
                      <button
                        className="btn btn-ghost btn-sm"
                        disabled={busy}
                        onClick={() =>
                          act(() =>
                            api.updateUser(user.id, { role: user.role === 'admin' ? 'utente' : 'admin' }),
                          )
                        }
                      >
                        {user.role === 'admin' ? 'Rendi utente' : 'Rendi admin'}
                      </button>
                    )}
                    {!self && (
                      <button
                        className="btn btn-ghost btn-sm"
                        disabled={busy}
                        onClick={() => act(() => api.updateUser(user.id, { active: !user.active }))}
                      >
                        {user.active ? 'Disattiva' : 'Riattiva'}
                      </button>
                    )}
                    {user.locked && (
                      <button className="btn btn-ghost btn-sm" disabled={busy} onClick={() => act(() => api.unlockUser(user.id))}>
                        Sblocca
                      </button>
                    )}
                    {!self && (
                      <button className="btn btn-ghost btn-sm" disabled={busy} onClick={() => reset(user)}>
                        Reimposta password
                      </button>
                    )}
                    {!self && user.sessions > 0 && (
                      <button className="btn btn-ghost btn-sm" disabled={busy} onClick={() => act(() => api.logoutUser(user.id))}>
                        Chiudi sessioni
                      </button>
                    )}
                    {!self && (
                      <button
                        className={`btn btn-sm ${confirmDelete === user.id ? 'btn-danger' : 'btn-danger-ghost'}`}
                        disabled={busy}
                        onBlur={() => setConfirmDelete(null)}
                        onClick={() =>
                          confirmDelete === user.id
                            ? act(async () => {
                                setConfirmDelete(null)
                                await api.deleteUser(user.id)
                              })
                            : setConfirmDelete(user.id)
                        }
                      >
                        {confirmDelete === user.id
                          ? user.matches
                            ? `Conferma: elimina con ${count(user.matches, 'partita', 'partite')}`
                            : 'Conferma eliminazione'
                          : 'Elimina'}
                      </button>
                    )}
                  </div>
                </div>
              )
            })}
          </div>
        </>
      )}

      {tab === 'audit' && (
        <div className="card">
          {audit.length === 0 && <p className="muted-note">Nessun evento registrato.</p>}
          {audit.map((row, i) => (
            <div key={i} className={`audit-row ${row.event.endsWith('fallito') ? 'audit-bad' : ''}`}>
              <span className="audit-when">{when(row.at)}</span>
              <span className="audit-event">{EVENT_LABEL[row.event] ?? row.event}</span>
              <span className="audit-who">{row.username ?? '—'}</span>
              <span className="audit-detail">
                {row.detail}
                {row.ip && (
                  <span className="muted-note">
                    {row.detail ? ' · ' : ''}
                    {row.ip}
                  </span>
                )}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
