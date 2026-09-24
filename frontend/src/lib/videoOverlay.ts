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
