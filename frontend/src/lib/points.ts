/** A point marked watching the video: serve, end, the pair that won it. */
export type MarkedPoint = [number, number, [number, number] | null]

/** A point is reached for its winner this long after its end: the time to
 *  press the key once the point is over. */
export const WINNER_AFTER_S = 15

export function stamp(t: number): string {
  const m = Math.floor(t / 60)
  return `${m}:${(t - 60 * m).toFixed(2).padStart(5, '0')}`
}

const round = (t: number) => Math.round(t * 100) / 100

/**
 * The points with one more, from `start` to `end`, in time order; or why
 * it cannot be added.
 */
export function addPoint(
  points: MarkedPoint[], start: number, end: number,
): { points: MarkedPoint[] } | { error: string } {
  const a = round(start)
  const b = round(end)
  if (b <= a) return { error: 'La fine del punto viene prima della battuta: torna alla battuta e premi di nuovo B.' }
  const clash = points.find(([s, e]) => a < e && b > s)
  if (clash) {
    return { error: `Si sovrappone al punto ${stamp(clash[0])}–${stamp(clash[1])}: toglilo o cambia i tempi.` }
  }
  return { points: [...points, [a, b, null] as MarkedPoint].sort((x, y) => x[0] - y[0]) }
}

/**
 * The point the winner key is for at time `t`: the one being watched, or
 * the last one ended up to WINNER_AFTER_S before; -1 for none.
 */
export function pointAt(points: MarkedPoint[], t: number): number {
  const inside = points.findIndex(([s, e]) => s <= t && t <= e)
  if (inside >= 0) return inside
  let best = -1
  points.forEach(([, e], i) => {
    if (e < t && t - e <= WINNER_AFTER_S && (best < 0 || e > points[best][1])) best = i
  })
  return best
}

export function setWinner(points: MarkedPoint[], index: number, pair: [number, number] | null): MarkedPoint[] {
  return points.map((p, i) => (i === index ? [p[0], p[1], pair ? ([...pair].sort((x, y) => x - y) as [number, number]) : null] : p))
}

/** The two pairs, by the team each player is in: ids sorted, pair of the
 *  lowest id first. Null unless there are two pairs of two. */
export function pairsOf(teams: Record<string, { team: number }>): [[number, number], [number, number]] | null {
  const byTeam = new Map<number, number[]>()
  for (const [id, p] of Object.entries(teams)) {
    byTeam.set(p.team, [...(byTeam.get(p.team) ?? []), Number(id)])
  }
  const pairs = [...byTeam.values()].map(ids => ids.sort((a, b) => a - b))
  if (pairs.length !== 2 || pairs.some(ids => ids.length !== 2)) return null
  pairs.sort((a, b) => a[0] - b[0])
  return [pairs[0] as [number, number], pairs[1] as [number, number]]
}

/** The text "Copia i punti segnati" gives: what backend/scripts/eval_points.py reads. */
export function pointsText(title: string, points: MarkedPoint[], name: (pid: number) => string): string {
  return [
    `${title} · punti segnati ${points.length}`,
    ...points.map(([s, e, w]) =>
      [stamp(s), stamp(e), ...(w ? [`${name(w[0])} e ${name(w[1])}`, `${w[0]},${w[1]}`] : ['-'])].join('\t')),
  ].join('\n')
}
