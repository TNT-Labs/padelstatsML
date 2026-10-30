/** Formatting shared by the views. Pure, so it is unit-tested. */

/**
 * A time typed by a person — "95", "1:35", "1:01:35" — in seconds, or null
 * when it is not one. Seconds and minutes past the first field stay under 60.
 */
export function parseClock(text: string): number | null {
  const parts = text.trim().split(':')
  if (parts.length > 3 || parts.some(p => !/^\d+$/.test(p))) return null
  const values = parts.map(Number)
  if (values.slice(1).some(v => v >= 60)) return null
  return values.reduce((total, v) => total * 60 + v, 0)
}

/** Metres per second to km/h, the unit a padel player thinks in. */
export function kmh(metresPerSecond: number): string {
  return `${(metresPerSecond * 3.6).toFixed(1)} km/h`
}

/**
 * A URL the server returned for one of its own files (a player thumbnail),
 * made to point at the API this page actually talks to.
 *
 * Older results carry an absolute URL built on the server's API_BASE_URL,
 * which is wrong whenever the browser reached the Pi at a different address —
 * every thumbnail then failed to load. Only the path and query are kept, and
 * `base` (the configured API origin, empty when the page is served by the
 * API itself) is put in front.
 */
export function apiAsset(url: string, base: string): string {
  try {
    const parsed = new URL(url, 'http://placeholder')
    return `${base}${parsed.pathname}${parsed.search}`
  } catch {
    return url
  }
}

/**
 * Pair and side of a player, for the cards. The pair is the one on the half
 * near the camera, or far from it, where the analysis starts: at changeovers
 * the pairs trade ends and keep their ids.
 */
export function handLabel(hand?: 'R' | 'L' | null): string {
  return hand === 'L' ? 'mancino' : hand === 'R' ? 'destro' : ''
}

export function playerPlace(team: number, role?: 'drive' | 'reves' | null): string {
  const pair = team === 0 ? 'Coppia vicina a inizio analisi' : team === 1 ? 'Coppia lontana a inizio analisi' : '—'
  if (role === 'drive') return `${pair} · lato drive`
  if (role === 'reves') return `${pair} · lato revés`
  return pair
}

/** A player's high strokes with the racket hand, when it is known: the
 *  free arm raised to point at the ball is not a stroke. */
export function strokesOf(
  strokes: { t: number; kind: 'overhead' | 'lob'; wrists: string }[] | null | undefined,
  hand?: 'R' | 'L' | null,
) {
  if (!strokes) return null
  return hand ? strokes.filter(s => s.wrists.includes(hand)) : strokes
}

export const STROKE_LABELS: Record<string, string> = {
  smash: 'Smash',
  bandeja: 'Bandeja / víbora',
  lob: 'Lob',
  volee: 'Volée',
  altro: 'Altro colpo',
  nulla: 'Non è un colpo',
}

/** How the high strokes found fare against what a person saw: of those
 *  checked, how many were a stroke at all and how many of the right kind
 *  (smash or bandeja for "overhead", lob for "lob"). */
export function strokeCheck(
  strokes: { key: string; kind: 'overhead' | 'lob' }[],
  labels: Record<string, string> | null | undefined,
) {
  let checked = 0
  let real = 0
  let rightKind = 0
  for (const { key, kind } of strokes) {
    const label = labels?.[key]
    if (!label) continue
    checked++
    if (label !== 'nulla') real++
    if (kind === 'overhead' ? label === 'smash' || label === 'bandeja' : label === 'lob') rightKind++
  }
  return { checked, real, rightKind }
}
