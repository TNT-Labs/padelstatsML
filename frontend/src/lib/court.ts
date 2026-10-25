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

/**
 * How far outside the frame a corner may lie, as a fraction of the frame size
 * on each side. Mirrors MAX_OFF_FRAME in app/ml/court.py.
 */
export const MAX_OFF_FRAME = 1.0

/** True when the point lies outside the video frame. */
export function isOffFrame(p: Pt, frameW: number, frameH: number): boolean {
  return p[0] < 0 || p[0] > frameW || p[1] < 0 || p[1] > frameH
}

export const CORNER_LABELS = [
  'Fondo sinistra',
  'Fondo destra',
  'Vicino destra',
  'Vicino sinistra',
]

export type Matrix3 = number[] // row-major, length 9

/**
 * Lens correction for a wide-angle lens, as on the server (app/ml/court.py):
 * straight = c + (p - c)(1 + k1 r^2), r = |p - c| / half diagonal.
 */
export interface Lens {
  k1: number
  frameW: number
  frameH: number
}

/** Highest correction offered: a phone's 0.6x needs about 0.1-0.3. */
export const MAX_K1 = 0.5

export function undistortPoint(p: Pt, lens: Lens): Pt {
  if (!lens.k1) return p
  const cx = lens.frameW / 2
  const cy = lens.frameH / 2
  const radius = Math.hypot(lens.frameW, lens.frameH) / 2
  const dx = p[0] - cx
  const dy = p[1] - cy
  const scale = 1 + (lens.k1 * (dx * dx + dy * dy)) / (radius * radius)
  return [cx + dx * scale, cy + dy * scale]
}

export function distortPoint(p: Pt, lens: Lens): Pt {
  if (!lens.k1) return p
  const cx = lens.frameW / 2
  const cy = lens.frameH / 2
  const radius = Math.hypot(lens.frameW, lens.frameH) / 2
  const dx = p[0] - cx
  const dy = p[1] - cy
  const ru = Math.hypot(dx, dy) / radius
  if (ru < 1e-12) return p
  let rd = ru
  for (let i = 0; i < 30; i++) rd -= (rd + lens.k1 * rd ** 3 - ru) / (1 + 3 * lens.k1 * rd * rd)
  const scale = rd / ru
  return [cx + dx * scale, cy + dy * scale]
}

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
export function checkQuad(raw: Pt[], frameW: number, frameH: number, k1 = 0): QuadCheck {
  if (raw.length !== 4) return { valid: false, reason: 'Servono 4 angoli.' }

  const mx = MAX_OFF_FRAME * frameW
  const my = MAX_OFF_FRAME * frameH
  if (raw.some(([x, y]) => x < -mx || x > frameW + mx || y < -my || y > frameH + my)) {
    return { valid: false, reason: "Un angolo è troppo lontano dall'inquadratura." }
  }
  // The shape is judged on the straightened picture, as the server does.
  const pts = raw.map(p => undistortPoint(p, { k1, frameW, frameH }))

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
export function projectCourtLines(corners: Pt[], lens?: Lens): Pt[][] {
  const bent = !!lens && lens.k1 > 0
  const straight = bent ? corners.map(p => undistortPoint(p, lens)) : corners
  const h = homographyFromQuad(COURT_CORNERS_M, straight)
  if (!h) return []
  const out: Pt[][] = []
  for (const line of COURT_LINES_M) {
    // Through a wide-angle lens a straight line on the court is a curve in
    // the picture: sampled along each segment, then bent.
    const samples: Pt[] = []
    for (let i = 0; i < line.length - 1; i++) {
      const steps = bent ? 24 : 1
      for (let k = 0; k <= steps; k++) {
        if (i > 0 && k === 0) continue
        const f = k / steps
        samples.push([line[i][0] + (line[i + 1][0] - line[i][0]) * f, line[i][1] + (line[i + 1][1] - line[i][1]) * f])
      }
    }
    const projected: Pt[] = []
    for (const point of samples) {
      const p = applyHomography(h, point)
      if (!p || !Number.isFinite(p[0]) || !Number.isFinite(p[1])) return []
      projected.push(bent ? distortPoint(p, lens!) : p)
    }
    out.push(projected)
  }
  return out
}

/** Distance, in metres, between a marked net line and the true Y = 10 m. */
export function netErrorMetres(corners: Pt[], netPx: Pt[], lens?: Lens): number | null {
  const straighten = (p: Pt) => (lens ? undistortPoint(p, lens) : p)
  const toCourt = homographyFromQuad(corners.map(straighten), COURT_CORNERS_M)
  if (!toCourt || netPx.length !== 2) return null
  let worst = 0
  for (const point of netPx.map(straighten)) {
    const court = applyHomography(toCourt, point)
    if (!court) return null
    worst = Math.max(worst, Math.abs(court[1] - NET_Y_M))
  }
  return worst
}
