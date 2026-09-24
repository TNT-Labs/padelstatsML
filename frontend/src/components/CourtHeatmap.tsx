/**
 * Top-down padel court with per-player position heatmaps.
 *
 * Court coordinates: origin at the near-left corner, X = width 0..10 m,
 * Y = length 0..20 m, net at Y = 10 m. The SVG viewBox is 0 0 100 200, so one
 * SVG unit is 10 cm.
 *
 * Heatmap points are [x_m, y_m, weight] on a 0.5 m grid, weights summing to 1
 * per player.
 */
import type { FC } from 'react'
import { PLAYER_COLORS } from '../lib/players'

const CELL_M = 0.5
const SERVICE_OFFSET = 6.95

interface Props {
  heatmaps: Record<string, [number, number, number][]>
  playerNames?: (string | null)[]
  /** Restrict the drawing to one player; omit to overlay everybody. */
  only?: string
  width?: number
}

export const CourtHeatmap: FC<Props> = ({ heatmaps, playerNames, only, width = 260 }) => {
  const height = width * 2
  const entries = Object.entries(heatmaps).filter(([pid]) => only === undefined || pid === only)

  // Normalise against the busiest cell on screen so a single player's map is
  // not washed out when shown on its own.
  const peak = Math.max(
    ...entries.flatMap(([, points]) => points.map(p => p[2])),
    1e-6,
  )

  return (
    <div className="heatmap-wrap" style={{ width }}>
      <svg viewBox="0 0 100 200" width={width} height={height} style={{ display: 'block' }}>
        <defs>
          <filter id="heat-blur" x="-30%" y="-30%" width="160%" height="160%">
            <feGaussianBlur stdDeviation="2.4" />
          </filter>
        </defs>

        <rect x={0} y={0} width={100} height={200} fill="#1d4e3f" />

        <g filter="url(#heat-blur)">
          {entries.map(([pid, points]) => {
            const color = PLAYER_COLORS[Number(pid) % 4]
            return points.map(([x, y, weight], i) => (
              <circle
                key={`${pid}-${i}`}
                cx={x * 10}
                cy={(20 - y) * 10}
                r={CELL_M * 10 * 1.5}
                fill={color}
                opacity={Math.min(0.85, 0.12 + (weight / peak) * 0.8)}
              />
            ))
          })}
        </g>

        {/* Court markings, drawn over the heat so the geometry stays readable */}
        <g stroke="#ffffff" strokeOpacity={0.75} fill="none" strokeWidth={0.7}>
          <rect x={0.5} y={0.5} width={99} height={199} />
          <line x1={0} y1={100} x2={100} y2={100} strokeWidth={1.2} />
          <line x1={0} y1={(20 - (10 + SERVICE_OFFSET)) * 10} x2={100} y2={(20 - (10 + SERVICE_OFFSET)) * 10} />
          <line x1={0} y1={(20 - (10 - SERVICE_OFFSET)) * 10} x2={100} y2={(20 - (10 - SERVICE_OFFSET)) * 10} />
          <line
            x1={50}
            y1={(20 - (10 + SERVICE_OFFSET)) * 10}
            x2={50}
            y2={(20 - (10 - SERVICE_OFFSET)) * 10}
          />
        </g>
      </svg>

      {only === undefined && (
        <div className="heatmap-legend">
          {Object.keys(heatmaps).map(pid => (
            <span key={pid}>
              <i style={{ background: PLAYER_COLORS[Number(pid) % 4] }} />
              {playerNames?.[Number(pid)] || `G${Number(pid) + 1}`}
            </span>
          ))}
        </div>
      )}
    </div>
  )
}
