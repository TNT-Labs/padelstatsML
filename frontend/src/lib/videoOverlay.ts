/** Placing the players' boxes on the video. Pure, so it is unit-tested. */

export type Box = [number, number, number, number]

/** Index of the last sample at or before `t`, or -1. `times` is sorted. */
export function lastAtOrBefore(times: number[], t: number): number {
  let lo = 0
  let hi = times.length - 1
  let found = -1
  while (lo <= hi) {
    const mid = (lo + hi) >> 1
    if (times[mid] <= t) {
      found = mid
      lo = mid + 1
    } else {
      hi = mid - 1
    }
  }
  return found
}

/**
 * A player's box at video time `t`, from samples taken a few times a second
 * (`times`, and `boxes` flattened four numbers per sample).
 *
 * Between two samples less than `maxGap` apart the box is interpolated, so it
 * glides with the player instead of jumping five times a second. Further
 * apart, the player was lost in between: the nearer sample is shown only
 * within half the gap limit of it, and nothing otherwise.
 */
export function boxAt(times: number[], boxes: number[], t: number, maxGap: number): Box | null {
  const i = lastAtOrBefore(times, t)
  const at = (k: number): Box => [boxes[4 * k], boxes[4 * k + 1], boxes[4 * k + 2], boxes[4 * k + 3]]
  const next = i + 1 < times.length ? i + 1 : -1
  if (i >= 0 && next >= 0 && times[next] - times[i] <= maxGap) {
    const f = (t - times[i]) / (times[next] - times[i])
    const a = at(i)
    const b = at(next)
    return [0, 1, 2, 3].map(k => a[k] + (b[k] - a[k]) * f) as Box
  }
  const before = i >= 0 ? t - times[i] : Infinity
  const after = next >= 0 ? times[next] - t : Infinity
  if (Math.min(before, after) > maxGap / 2) return null
  return before <= after ? at(i) : at(next)
}

/** Limbs of the body's 17 points (COCO order), drawn as lines. */
export const SKELETON: [number, number][] = [
  [5, 7], [7, 9], [6, 8], [8, 10], [5, 6], [5, 11], [6, 12], [11, 12],
  [11, 13], [13, 15], [12, 14], [14, 16], [0, 5], [0, 6],
]
/** A point is drawn when the network is at least this sure it is visible (0-100). */
export const MIN_VISIBLE = 50

/**
 * A player's body points at video time `t` (51 numbers: x, y, visibility),
 * as boxAt does for the box: interpolated between two close samples that
 * both have them, the nearer one for a moment past a gap, null otherwise.
 */
export function pointsAt(
  times: number[], points: (number[] | null)[] | undefined, t: number, maxGap: number,
): number[] | null {
  if (!points) return null
  const i = lastAtOrBefore(times, t)
  const next = i + 1 < times.length ? i + 1 : -1
  const a = i >= 0 ? points[i] : null
  const b = next >= 0 ? points[next] : null
  if (a && b && times[next] - times[i] <= maxGap) {
    const f = (t - times[i]) / (times[next] - times[i])
    return a.map((v, k) => (k % 3 === 2 ? Math.min(v, b[k]) : v + (b[k] - v) * f))
  }
  const before = a ? t - times[i] : Infinity
  const after = b ? times[next] - t : Infinity
  if (Math.min(before, after) > maxGap / 2) return null
  return before <= after ? a : b
}

/** A name tag: left edge, bottom edge (where it would sit, on the box's
 *  top), width and height, in canvas pixels. */
export interface Tag {
  x: number
  bottom: number
  w: number
  h: number
}

/**
 * Bottom edges for the name tags so that none covers another: a player
 * just behind another has his box's top at the same height, and his tag
 * hid the other's name. Tags are placed top first; one that would overlap
 * an earlier one moves down a tag's height at a time, over its own box.
 */
export function placeTags(tags: Tag[]): number[] {
  const placed: { x: number; top: number; w: number; h: number }[] = []
  const out = new Array<number>(tags.length)
  const order = tags.map((_, i) => i).sort((a, b) => tags[a].bottom - tags[b].bottom)
  for (const i of order) {
    const tag = tags[i]
    let bottom = tag.bottom
    const overlaps = () =>
      placed.some(p => tag.x < p.x + p.w && p.x < tag.x + tag.w && bottom - tag.h < p.top + p.h && p.top < bottom)
    for (let step = 0; step < 8 && overlaps(); step++) bottom += tag.h
    placed.push({ x: tag.x, top: bottom - tag.h, w: tag.w, h: tag.h })
    out[i] = bottom
  }
  return out
}

/** Where a frame of `frameW`×`frameH` lands in a `boxW`×`boxH` element
 *  showing it whole (object-fit: contain, as <video> does). */
export function fitRect(frameW: number, frameH: number, boxW: number, boxH: number) {
  const scale = Math.min(boxW / frameW, boxH / frameH)
  return { scale, offsetX: (boxW - frameW * scale) / 2, offsetY: (boxH - frameH * scale) / 2 }
}

/** m:ss, for seek buttons. */
export function clock(seconds: number): string {
  const total = Math.max(0, Math.floor(seconds))
  return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, '0')}`
}
