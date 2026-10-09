import { useEffect, useRef, useState, type FC } from 'react'
import { BASE, type Me } from '../api'

interface Props {
  me: Me
  onPassword: () => void
  onAdmin: () => void
  onLogout: () => void
}

/** Thin bar above every screen: who is signed in, and the account menu. */
export const UserBar: FC<Props> = ({ me, onPassword, onAdmin, onLogout }) => {
  const [open, setOpen] = useState(false)
  const menu = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!open) return
    const outside = (e: PointerEvent) => {
      if (menu.current && !menu.current.contains(e.target as Node)) setOpen(false)
    }
    const escape = (e: KeyboardEvent) => e.key === 'Escape' && setOpen(false)
    document.addEventListener('pointerdown', outside)
    document.addEventListener('keydown', escape)
    return () => {
      document.removeEventListener('pointerdown', outside)
      document.removeEventListener('keydown', escape)
    }
  }, [open])

  const pick = (action: () => void) => () => {
    setOpen(false)
    action()
  }

  return (
    <header className="topbar">
      <div className="topbar-inner">
        {/* Served under a path of a shared domain: the other applications
            are one level up. */}
        {BASE.startsWith('/') && (
          <a className="topbar-link" href="/">
            ← Tutte le app
          </a>
        )}
        <div className="topbar-spacer" />
        <div className="user-menu" ref={menu}>
          <button
            className="user-btn"
            aria-haspopup="menu"
            aria-expanded={open}
            onClick={() => setOpen(o => !o)}
          >
            <span className="user-avatar" aria-hidden>
              {me.username.charAt(0).toUpperCase()}
            </span>
            <span className="user-name">{me.username}</span>
            {me.role === 'admin' && <span className="role-tag">admin</span>}
            <span aria-hidden>▾</span>
          </button>
          {open && (
            <div className="user-menu-list" role="menu">
              {me.role === 'admin' && (
                <button role="menuitem" onClick={pick(onAdmin)}>
                  Gestione utenti
                </button>
              )}
              <button role="menuitem" onClick={pick(onPassword)}>
                Cambia password
              </button>
              <button role="menuitem" className="danger" onClick={pick(onLogout)}>
                Esci
              </button>
            </div>
          )}
        </div>
      </div>
    </header>
  )
}
