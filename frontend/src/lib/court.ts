/**
 * Client-side court geometry.
 *
 * The calibration screen solves the same homography the backend will solve,
 * so it can draw the resulting court model — net, service lines, outline —
 * straight onto the keyframe while the user drags the corners. Seeing the
 * predicted net land on the real net is what turns manual calibration from a
 * chore into a two-second check, and it catches a misplaced corner before
 * an hour of analysis is spent on it.
 */

export const COURT_WIDTH_M = 10
export const COURT_LENGTH_M = 20
export const NET_Y_M = 10
export const SERVICE_OFFSET_M = 6.95

export type Pt = [number, number]

/** Court corners in metres, in the order the UI asks the user to place them. */
export const COURT_CORNERS_M: Pt[] = [
  [0, COURT_LENGTH_M], // far-left
  [COURT_WIDTH_M, COURT_LENGTH_M], // far-right
  [COURT_WIDTH_M, 0], // near-right
  [0, 0], // near-left
]

export const CORNER_LABELS = [
  'Fondo sinistra',
  'Fondo destra',
  'Vicino destra',
  'Vicino sinistra',
]

export type Matrix3 = number[] // row-major, length 9

/**
 * Exact 4-point homography via the direct linear transform.
 * Returns null when the system is singular (degenerate quadrilateral).
 */
export function homographyFromQuad(src: Pt[], dst: Pt[]): Matrix3 | null {
  if (src.length !== 4 || dst.length !== 4) return null

  const a: number[][] = []
  const b: number[] = []
  for (let i = 0; i < 4; i++) {
    const [x, y] = src[i]
    const [u, v] = dst[i]
    a.push([x, y, 1, 0, 0, 0, -u * x, -u * y])
    b.push(u)
    a.push([0, 0, 0, x, y, 1, -v * x, -v * y])
    b.push(v)
  }

  const h = solve(a, b)
  if (!h) return null
  return [...h, 1]
}

/** Gaussian elimination with partial pivoting. */
function solve(a: number[][], b: number[]): number[] | null {
  const n = b.length
  const m = a.map((row, i) => [...row, b[i]])

  for (let col = 0; col < n; col++) {
    let pivot = col
    for (let r = col + 1; r < n; r++) {
      if (Math.abs(m[r][col]) > Math.abs(m[pivot][col])) pivot = r
    }
    if (Math.abs(m[pivot][col]) < 1e-10) return null
    ;[m[col], m[pivot]] = [m[pivot], m[col]]

    for (let r = 0; r < n; r++) {
      if (r === col) continue
      const factor = m[r][col] / m[col][col]
      if (factor === 0) continue
      for (let c = col; c <= n; c++) m[r][c] -= factor * m[col][c]
    }
  }

  const out = new Array(n)
  for (let i = 0; i < n; i++) {
    out[i] = m[i][n] / m[i][i]
    if (!Number.isFinite(out[i])) return null
  }
  return out
}

export function applyHomography(h: Matrix3, p: Pt): Pt | null {
  const w = h[6] * p[0] + h[7] * p[1] + h[8]
  if (Math.abs(w) < 1e-9) return null
  return [
    (h[0] * p[0] + h[1] * p[1] + h[2]) / w,
    (h[3] * p[0] + h[4] * p[1] + h[5]) / w,
  ]
}

/** Signed area; positive when the points run clockwise on screen (Y down). */
export function signedArea(pts: Pt[]): number {
  let sum = 0
  for (let i = 0; i < pts.length; i++) {
    const [x1, y1] = pts[i]
    const [x2, y2] = pts[(i + 1) % pts.length]
    sum += x1 * y2 - x2 * y1
  }
  return sum / 2
}

export function isConvex(pts: Pt[]): boolean {
  let sign = 0
  for (let i = 0; i < pts.length; i++) {
    const a = pts[i]
    const b = pts[(i + 1) % pts.length]
    const c = pts[(i + 2) % pts.length]
    const cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
    if (Math.abs(cross) < 1e-6) return false
    const current = Math.sign(cross)
    if (sign === 0) sign = current
    else if (current !== sign) return false
  }
  return true
}

export interface QuadCheck {
  valid: boolean
  reason?: string
}

/**
 * Mirrors the server-side validation so the user is told what is wrong while
 * dragging, instead of after pressing save.
 */
export function checkQuad(pts: Pt[], frameW: number, frameH: number): QuadCheck {
  if (pts.length !== 4) return { valid: false, reason: 'Servono 4 angoli.' }

  for (let i = 0; i < 4; i++) {
    for (let j = i + 1; j < 4; j++) {
      const dx = pts[i][0] - pts[j][0]
      const dy = pts[i][1] - pts[j][1]
      if (Math.hypot(dx, dy) < 8) return { valid: false, reason: 'Due angoli coincidono.' }
    }
  }
  if (!isConvex(pts)) {
    return { valid: false, reason: 'Il perimetro si incrocia: segui i lati del campo.' }
  }
  const area = signedArea(pts)
  if (area < 0) {
    return { valid: false, reason: 'Ordine invertito: il campo risulterebbe specchiato.' }
  }
  if (Math.abs(area) < 0.02 * frameW * frameH) {
    return { valid: false, reason: 'Il campo occupa una porzione troppo piccola del frame.' }
  }
  return { valid: true }
}

/** Court-model polylines (in metres) drawn over the keyframe as a check. */
export const COURT_LINES_M: Pt[][] = [
  // Outline
  [
    [0, 0],
    [COURT_WIDTH_M, 0],
    [COURT_WIDTH_M, COURT_LENGTH_M],
    [0, COURT_LENGTH_M],
    [0, 0],
  ],
  // Net
  [
    [0, NET_Y_M],
    [COURT_WIDTH_M, NET_Y_M],
  ],
  // Service lines
  [
    [0, NET_Y_M - SERVICE_OFFSET_M],
    [COURT_WIDTH_M, NET_Y_M - SERVICE_OFFSET_M],
  ],
  [
    [0, NET_Y_M + SERVICE_OFFSET_M],
    [COURT_WIDTH_M, NET_Y_M + SERVICE_OFFSET_M],
  ],
  // Centre service line, between the two service lines only
  [
    [COURT_WIDTH_M / 2, NET_Y_M - SERVICE_OFFSET_M],
    [COURT_WIDTH_M / 2, NET_Y_M + SERVICE_OFFSET_M],
  ],
]

/** Project the court model into image pixels. Returns [] if unprojectable. */
export function projectCourtLines(corners: Pt[]): Pt[][] {
  const h = homographyFromQuad(COURT_CORNERS_M, corners)
  if (!h) return []
  const out: Pt[][] = []
  for (const line of COURT_LINES_M) {
    const projected: Pt[] = []
    for (const point of line) {
      const p = applyHomography(h, point)
      if (!p || !Number.isFinite(p[0]) || !Number.isFinite(p[1])) return []
      projected.push(p)
    }
    out.push(projected)
  }
  return out
}

/** Distance, in metres, between a marked net line and the true Y = 10 m. */
export function netErrorMetres(corners: Pt[], netPx: Pt[]): number | null {
  const toCourt = homographyFromQuad(corners, COURT_CORNERS_M)
  if (!toCourt || netPx.length !== 2) return null
  let worst = 0
  for (const point of netPx) {
    const court = applyHomography(toCourt, point)
    if (!court) return null
    worst = Math.max(worst, Math.abs(court[1] - NET_Y_M))
  }
  return worst
}
