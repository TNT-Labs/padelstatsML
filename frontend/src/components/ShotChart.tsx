/**
 * Horizontal stacked bar chart showing shot distribution per player.
 * Pure SVG — no charting library dependency.
 */
import type { FC } from 'react'
import type { PlayerStats } from '../api'

const SHOT_COLORS: Record<string, string> = {
  smash:   '#ef4444',
  volley:  '#3b82f6',
  bandeja: '#f59e0b',
  other:   '#94a3b8',
}
const SHOT_LABELS: Record<string, string> = {
  smash: 'Smash', volley: 'Volée', bandeja: 'Bandeja', other: 'Altro',
}
const PLAYER_COLORS = ['#ef4444', '#3b82f6', '#f59e0b', '#8b5cf6']
const SHOT_ORDER = ['smash', 'volley', 'bandeja', 'other'] as const

interface Props {
  perPlayer: Record<string, PlayerStats>
  playerNames?: (string | null)[]
}

export const ShotChart: FC<Props> = ({ perPlayer, playerNames }) => {
  const players = Object.entries(perPlayer).sort(([a], [b]) => Number(a) - Number(b))
  const barH = 28
  const gap = 12
  const labelW = 72
  const barW = 300
  const legendH = 28
  const svgH = players.length * (barH + gap) + legendH + 16

  return (
    <svg viewBox={`0 0 ${labelW + barW + 10} ${svgH}`} style={{ width: '100%', display: 'block' }}>
      {players.map(([pid, ps], row) => {
        const idx = Number(pid)
        const total = Object.values(ps.shots).reduce((s, v) => s + v, 0) || 1
        const y = row * (barH + gap) + 4
        let xOff = labelW

        return (
          <g key={pid}>
            {/* Player name */}
            <text x={labelW - 6} y={y + barH / 2 + 4}
                  textAnchor="end" fontSize={11} fontWeight="600"
                  fill={PLAYER_COLORS[idx % 4]}>
              {playerNames?.[idx] || `P${idx + 1}`}
            </text>
            {/* Stacked segments */}
            {SHOT_ORDER.map((type) => {
              const count = ps.shots[type] ?? 0
              const w = (count / total) * barW
              const seg = (
                <g key={type}>
                  <rect x={xOff} y={y} width={w} height={barH} fill={SHOT_COLORS[type]} />
                  {w > 20 && (
                    <text x={xOff + w / 2} y={y + barH / 2 + 4}
                          textAnchor="middle" fontSize={9} fill="white" fontWeight="600">
                      {count}
                    </text>
                  )}
                </g>
              )
              xOff += w
              return seg
            })}
            {/* Total label */}
            <text x={labelW + barW + 6} y={y + barH / 2 + 4} fontSize={10} fill="#6b7280">
              {total}
            </text>
          </g>
        )
      })}

      {/* Legend */}
      {(() => {
        const ly = players.length * (barH + gap) + 12
        let lx = labelW
        return (
          <g>
            {SHOT_ORDER.map((type) => {
              const el = (
                <g key={type}>
                  <rect x={lx} y={ly} width={10} height={10} fill={SHOT_COLORS[type]} rx={2} />
                  <text x={lx + 14} y={ly + 9} fontSize={10} fill="#374151">
                    {SHOT_LABELS[type]}
                  </text>
                </g>
              )
              lx += SHOT_LABELS[type].length * 6.5 + 24
              return el
            })}
          </g>
        )
      })()}
    </svg>
  )
}
