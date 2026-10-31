/**
 * Where each player hit the ball from: a dot per hit on the court seen from
 * above (as CourtHeatmap: X 0..10 m, Y 0..20 m, net at 10 m, near end at the
 * bottom).
 */
import type { FC } from 'react'
import type { PlayerStats } from '../api'
import { PLAYER_COLORS } from '../lib/players'

const SERVICE_OFFSET = 6.95

interface Props {
  perPlayer: Record<string, PlayerStats>
  playerNames?: (string | null)[]
  width?: number
}

export const HitMap: FC<Props> = ({ perPlayer, playerNames, width = 240 }) => {
  const players = Object.entries(perPlayer)
    .filter(([, stats]) => stats.hit_spots?.length)
    .sort(([a], [b]) => Number(a) - Number(b))
  return (
    <div className="heatmap-wrap" style={{ width }}>
      <svg viewBox="0 0 100 200" width={width} height={width * 2} style={{ display: 'block' }}
        role="img" aria-label="Punti del campo da cui ogni giocatore ha colpito">
        <rect x={0} y={0} width={100} height={200} fill="#1d4e3f" />
        <g stroke="#ffffff" strokeOpacity={0.75} fill="none" strokeWidth={0.7}>
          <rect x={0.5} y={0.5} width={99} height={199} />
          <line x1={0} y1={100} x2={100} y2={100} strokeWidth={1.2} />
          <line x1={0} y1={(20 - (10 + SERVICE_OFFSET)) * 10} x2={100} y2={(20 - (10 + SERVICE_OFFSET)) * 10} />
          <line x1={0} y1={(20 - (10 - SERVICE_OFFSET)) * 10} x2={100} y2={(20 - (10 - SERVICE_OFFSET)) * 10} />
          <line x1={50} y1={(20 - (10 + SERVICE_OFFSET)) * 10} x2={50} y2={(20 - (10 - SERVICE_OFFSET)) * 10} />
        </g>
        {players.map(([pid, stats]) =>
          (stats.hit_spots ?? []).map(([x, y], i) => (
            <circle
              key={`${pid}-${i}`}
              cx={Math.min(100, Math.max(0, x * 10))}
              cy={Math.min(200, Math.max(0, (20 - y) * 10))}
              r={1.8}
              fill={PLAYER_COLORS[Number(pid) % 4]}
              stroke="#fff"
              strokeWidth={0.3}
              opacity={0.85}
            />
          )),
        )}
      </svg>
      <div className="heatmap-legend">
        {players.map(([pid, stats]) => (
          <span key={pid}>
            <i style={{ background: PLAYER_COLORS[Number(pid) % 4] }} />
            {playerNames?.[Number(pid)] || `G${Number(pid) + 1}`} · {stats.hit_spots?.length}
          </span>
        ))}
      </div>
    </div>
  )
}
