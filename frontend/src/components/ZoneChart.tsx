/**
 * Court-position distribution per player: the share of time spent at the net,
 * in mid-court and at the back.
 *
 * This replaces the old shot-type chart. Shot types required ball tracking the
 * Pi cannot run, and were previously produced by a heuristic that guessed.
 * Court position is measured directly, is the most coached aspect of padel,
 * and is the metric this system can actually stand behind.
 */
import type { FC } from 'react'
import type { PlayerStats } from '../api'

const ZONES = [
  { key: 'net', label: 'Rete', color: '#16a34a' },
  { key: 'mid', label: 'Metà campo', color: '#f59e0b' },
  { key: 'back', label: 'Fondo', color: '#64748b' },
] as const

const PLAYER_COLORS = ['#ef4444', '#3b82f6', '#f59e0b', '#8b5cf6']

interface Props {
  perPlayer: Record<string, PlayerStats>
  playerNames?: (string | null)[]
}

export const ZoneChart: FC<Props> = ({ perPlayer, playerNames }) => {
  const players = Object.entries(perPlayer).sort(([a], [b]) => Number(a) - Number(b))
  const barH = 26
  const gap = 14
  const labelW = 80
  const barW = 280
  const legendH = 26
  const svgH = players.length * (barH + gap) + legendH + 12

  return (
    <svg viewBox={`0 0 ${labelW + barW + 40} ${svgH}`} style={{ width: '100%', display: 'block' }}>
      {players.map(([pid, stats], row) => {
        const index = Number(pid)
        const y = row * (barH + gap) + 4
        let x = labelW

        return (
          <g key={pid}>
            <text
              x={labelW - 8}
              y={y + barH / 2 + 4}
              textAnchor="end"
              fontSize={11}
              fontWeight="600"
              fill={PLAYER_COLORS[index % 4]}
            >
              {playerNames?.[index] || `G${index + 1}`}
            </text>

            {ZONES.map(zone => {
              const share = stats.zone_pct[zone.key] ?? 0
              const w = share * barW
              const segment = (
                <g key={zone.key}>
                  <rect x={x} y={y} width={w} height={barH} fill={zone.color} />
                  {w > 30 && (
                    <text
                      x={x + w / 2}
                      y={y + barH / 2 + 4}
                      textAnchor="middle"
                      fontSize={10}
                      fill="#fff"
                      fontWeight="600"
                    >
                      {Math.round(share * 100)}%
                    </text>
                  )}
                </g>
              )
              x += w
              return segment
            })}
          </g>
        )
      })}

      <g>
        {ZONES.map((zone, i) => (
          <g key={zone.key}>
            <rect x={labelW + i * 92} y={svgH - legendH + 6} width={10} height={10} rx={2} fill={zone.color} />
            <text x={labelW + i * 92 + 15} y={svgH - legendH + 15} fontSize={10} fill="#475569">
              {zone.label}
            </text>
          </g>
        ))}
      </g>
    </svg>
  )
}
