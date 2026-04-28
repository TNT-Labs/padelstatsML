/**
 * SVG top-down padel court with per-player position heatmaps.
 *
 * Court coordinate system: origin bottom-left, X = width 0..10m, Y = length 0..20m.
 * SVG viewBox 0 0 100 200 (1 unit = 0.1m).
 *
 * Heatmap data: [[x_m, y_m, weight], ...] — weights sum to 1 per player.
 * Blurred colored circles give a gaussian-like heat effect via CSS filter.
 */
import type { FC } from 'react'

const PLAYER_COLORS = ['#ef4444', '#3b82f6', '#f59e0b', '#8b5cf6']
const PLAYER_NAMES  = ['P1', 'P2', 'P3', 'P4']

interface Props {
  heatmaps: Record<string, [number, number, number][]>
  playerNames?: (string | null)[]
  width?: number
}

export const CourtHeatmap: FC<Props> = ({ heatmaps, playerNames, width = 260 }) => {
  const height = width * 2  // 1:2 aspect ratio

  // SVG viewBox is 0 0 100 200; court in metres → SVG: x*10, y*10
  const toSVG = (m: number) => m * 10

  return (
    <div className="heatmap-wrap" style={{ width, background: '#166534' }}>
      <svg
        viewBox="0 0 100 200"
        width={width}
        height={height}
        style={{ display: 'block' }}
      >
        {/* Court surface */}
        <rect x={0} y={0} width={100} height={200} fill="#2d6a4f" />

        {/* Heatmap blobs — rendered below court lines */}
        {Object.entries(heatmaps).map(([pid, points]) => {
          const color = PLAYER_COLORS[Number(pid) % 4]
          if (!points.length) return null
          return (
            <g key={pid} style={{ filter: 'blur(5px)', opacity: 0.75 }}>
              {points.map(([xm, ym, w], i) => (
                <circle
                  key={i}
                  cx={toSVG(xm)}
                  cy={toSVG(ym)}
                  r={Math.sqrt(w) * 30}
                  fill={color}
                  opacity={Math.min(w * 12, 0.9)}
                />
              ))}
            </g>
          )
        })}

        {/* Court lines */}
        {/* Outer boundary */}
        <rect x={1} y={1} width={98} height={198} fill="none" stroke="rgba(255,255,255,0.9)" strokeWidth={1.5} />
        {/* Net */}
        <line x1={0} y1={100} x2={100} y2={100} stroke="white" strokeWidth={2} />
        <line x1={2} y1={100} x2={98} y2={100} stroke="#d4d4d4" strokeWidth={0.5} strokeDasharray="2 2" />
        {/* Service lines */}
        <line x1={2} y1={70}  x2={98} y2={70}  stroke="rgba(255,255,255,0.75)" strokeWidth={1} />
        <line x1={2} y1={130} x2={98} y2={130} stroke="rgba(255,255,255,0.75)" strokeWidth={1} />
        {/* Center T */}
        <line x1={50} y1={70} x2={50} y2={130} stroke="rgba(255,255,255,0.75)" strokeWidth={1} />

        {/* Player labels at center of each quarter */}
        {Object.keys(heatmaps).map((pid) => {
          const idx = Number(pid)
          const color = PLAYER_COLORS[idx % 4]
          const name = playerNames?.[idx] ?? PLAYER_NAMES[idx]
          // bottom half = players 0,1 (Y > 100); top half = players 2,3 (Y < 100)
          const cy = idx < 2 ? 160 : 40
          const cx = idx % 2 === 0 ? 25 : 75
          return (
            <text key={pid} x={cx} y={cy} textAnchor="middle" fontSize={8}
                  fill={color} fontWeight="bold" style={{ userSelect: 'none' }}>
              {name}
            </text>
          )
        })}
      </svg>

      {/* Legend */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px', padding: '6px 8px', background: '#14532d' }}>
        {Object.keys(heatmaps).map((pid) => {
          const idx = Number(pid)
          return (
            <span key={pid} style={{ display: 'flex', alignItems: 'center', gap: 4, color: '#fff', fontSize: 11 }}>
              <span style={{ width: 10, height: 10, borderRadius: '50%', background: PLAYER_COLORS[idx % 4] }} />
              {playerNames?.[idx] ?? PLAYER_NAMES[idx]}
            </span>
          )
        })}
      </div>
    </div>
  )
}
