/** A hit marked watching the video: [seconds, player id] or, once its kind
 *  is said, [seconds, player id, kind]. */
export type HitKind =
  | 'battuta' | 'dritto' | 'rovescio' | 'volee_dritto' | 'volee_rovescio'
  | 'smash' | 'bandeja' | 'lob' | 'parete' | 'altro'
export type MarkedHit = [number, number] | [number, number, HitKind]

/** The kinds, with the key that marks each: a volley's key beside its
 *  side's (D-F, R-E), the bandeja's beside the smash's (S-A). */
export const HIT_KINDS: { kind: HitKind; key: string; label: string }[] = [
  { kind: 'battuta', key: 'b', label: 'Battuta' },
  { kind: 'dritto', key: 'd', label: 'Dritto' },
  { kind: 'rovescio', key: 'r', label: 'Rovescio' },
  { kind: 'volee_dritto', key: 'f', label: 'Volée di dritto' },
  { kind: 'volee_rovescio', key: 'e', label: 'Volée di rovescio' },
  { kind: 'smash', key: 's', label: 'Smash' },
  { kind: 'bandeja', key: 'a', label: 'Bandeja' },
  { kind: 'lob', key: 'l', label: 'Lob' },
  { kind: 'parete', key: 'p', label: 'Uscita di parete' },
  { kind: 'altro', key: 'x', label: 'Altro' },
]

export function kindOfKey(key: string): HitKind | null {
  return HIT_KINDS.find(k => k.key === key.toLowerCase())?.kind ?? null
}

export function kindLabel(kind: HitKind | undefined): string {
  return HIT_KINDS.find(k => k.kind === kind)?.label ?? ''
}

/** The hits with the one at time `t` given `kind` (null: none). */
export function setKind(hits: MarkedHit[], t: number, kind: HitKind | null): MarkedHit[] {
  return hits.map(h => (h[0] === t ? (kind ? [h[0], h[1], kind] : [h[0], h[1]]) : h))
}

/** The first hit without a kind after time `t` (strictly), or the first
 *  one at all when none is after; null when every hit has one. */
export function nextUnkinded(hits: MarkedHit[], t: number): MarkedHit | null {
  const open = hits.filter(h => h.length < 3)
  return open.find(h => h[0] > t) ?? open[0] ?? null
}

/** The text "Copia i colpi segnati" gives: what training/hits reads. */
export function hitsText(title: string, hits: MarkedHit[], name: (pid: number) => string, stamp: (t: number) => string): string {
  const kinded = hits.filter(h => h.length === 3).length
  return [
    `${title} · colpi segnati ${hits.length}${kinded ? ` (con il tipo ${kinded})` : ''}`,
    ...hits.map(h => [stamp(h[0]), name(h[1]), String(h[1]), ...(h.length === 3 ? [h[2]] : [])].join('\t')),
  ].join('\n')
}
