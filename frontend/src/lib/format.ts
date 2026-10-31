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

export const STROKE_LABELS: Record<string, string> = {
  smash: 'Smash',
  bandeja: 'Bandeja / víbora',
  lob: 'Lob',
  volee: 'Volée',
  dritto: 'Dritto',
  rovescio: 'Rovescio',
  altro: 'Altro colpo',
  nulla: 'Non è un colpo',
}

/** How the high strokes found fare against what a person saw: of those
 *  checked, how many were a stroke at all and how many of the right kind
 *  (smash or bandeja for "overhead", lob for "lob"). */
export function strokeCheck(
  strokes: { key: string; kind: StrokeKind }[],
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
    if (RIGHT_LABELS[kind].includes(label)) rightKind++
  }
  return { checked, real, rightKind }
}

export type StrokeKind = 'overhead' | 'lob'

/** What a person may call each kind found, and be in agreement with it. */
const RIGHT_LABELS: Record<StrokeKind, string[]> = {
  overhead: ['smash', 'bandeja'],
  lob: ['lob'],
}

/** The name on a stroke's chip. */
export function strokeName(stroke: { kind: StrokeKind }): string {
  return stroke.kind === 'overhead' ? 'smash' : 'lob'
}
