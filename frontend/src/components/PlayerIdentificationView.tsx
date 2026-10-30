import { useState, type FC } from 'react'
import { api, type Hand, type MatchStats } from '../api'
import { playerPlace } from '../lib/format'
import { PlayerThumb } from './PlayerThumb'

interface Props {
  stats: MatchStats
  onConfirm: () => void
  onSkip: () => void
}

export const PlayerIdentificationView: FC<Props> = ({ stats, onConfirm, onSkip }) => {
  const players = Object.entries(stats.per_player).sort(([a], [b]) => Number(a) - Number(b))
  const [names, setNames] = useState<string[]>(
    players.map((_, i) => stats.player_names?.[i] ?? ''),
  )
  const [hands, setHands] = useState<Hand[]>(players.map((_, i) => stats.player_hands?.[i] ?? null))
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const setName = (index: number, value: string) =>
    setNames(current => current.map((n, i) => (i === index ? value : n)))
  // A second click on the chosen hand clears it: "not said" stays possible.
  const setHand = (index: number, value: Hand) =>
    setHands(current => current.map((h, i) => (i === index ? (h === value ? null : value) : h)))

  const save = async () => {
    setSaving(true)
    setError(null)
    try {
      // Persisted on the Pi, not kept in browser state: the names used to be
      // lost on reload and never reached a second device.
      await api.updateMatch(stats.match_id, { player_names: names, player_hands: hands })
      onConfirm()
    } catch (e) {
      setError((e as Error).message)
      setSaving(false)
    }
  }

  return (
    <div className="layout">
      <div className="header">
        <h1>Chi è chi?</h1>
        <button className="btn btn-ghost btn-sm" onClick={onSkip} disabled={saving}>
          Salta
        </button>
      </div>

      <p className="muted-note" style={{ marginBottom: '1.5rem', textAlign: 'center' }}>
        Associa un nome a ciascun giocatore riconosciuto e indica se gioca con la destra o con la
        sinistra: diritto e rovescio stanno dai due lati opposti per un mancino. Tutto resta
        salvato sul Pi.
      </p>

      <div className="grid-2" style={{ marginBottom: '2rem' }}>
        {players.map(([pid, player], index) => (
          <div key={pid} className="card" style={{ textAlign: 'center' }}>
            <PlayerThumb url={player.crop_url} alt={`Giocatore ${index + 1}`} className="identify-thumb" />
            <div className="muted-note" style={{ marginBottom: '.5rem' }}>
              {playerPlace(player.team, player.role)}
            </div>
            <input
              type="text"
              value={names[index]}
              placeholder={`Giocatore ${index + 1}`}
              onChange={e => setName(index, e.target.value)}
              style={{ width: '100%', textAlign: 'center' }}
            />
            <div className="hand-choice" role="group" aria-label={`Mano di ${names[index] || `Giocatore ${index + 1}`}`}>
              {(['R', 'L'] as const).map(hand => (
                <button
                  key={hand}
                  type="button"
                  className={`btn btn-sm ${hands[index] === hand ? 'btn-primary' : 'btn-ghost'}`}
                  aria-pressed={hands[index] === hand}
                  onClick={() => setHand(index, hand)}
                >
                  {hand === 'R' ? 'Destro' : 'Mancino'}
                </button>
              ))}
            </div>
          </div>
        ))}
      </div>

      {error && <p className="error-msg">{error}</p>}

      <div style={{ textAlign: 'center' }}>
        <button className="btn btn-primary" style={{ minWidth: 220 }} disabled={saving} onClick={save}>
          {saving ? 'Salvataggio…' : 'Vedi statistiche →'}
        </button>
      </div>
    </div>
  )
}
