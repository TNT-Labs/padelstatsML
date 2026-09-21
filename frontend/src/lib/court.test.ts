import { describe, expect, it } from 'vitest'
import {
  COURT_CORNERS_M,
  NET_Y_M,
  applyHomography,
  checkQuad,
  homographyFromQuad,
  netErrorMetres,
  projectCourtLines,
  signedArea,
  type Pt,
} from './court'

const FRAME_W = 1920
const FRAME_H = 1080
// Camera behind the baseline, elevated: the usual padel recording setup.
const GOOD: Pt[] = [
  [600, 300],
  [1320, 300],
  [1800, 980],
  [120, 980],
]

describe('homography', () => {
  it('maps the four court corners onto the four image corners', () => {
    const h = homographyFromQuad(COURT_CORNERS_M, GOOD)!
    expect(h).not.toBeNull()
    COURT_CORNERS_M.forEach((courtPoint, i) => {
      const projected = applyHomography(h, courtPoint)!
      expect(projected[0]).toBeCloseTo(GOOD[i][0], 4)
      expect(projected[1]).toBeCloseTo(GOOD[i][1], 4)
    })
  })

  it('round-trips a point through both directions', () => {
    const toImage = homographyFromQuad(COURT_CORNERS_M, GOOD)!
    const toCourt = homographyFromQuad(GOOD, COURT_CORNERS_M)!
    const original: Pt = [3.5, 14.2]
    const back = applyHomography(toCourt, applyHomography(toImage, original)!)!
    expect(back[0]).toBeCloseTo(original[0], 6)
    expect(back[1]).toBeCloseTo(original[1], 6)
  })

  it('returns null for a degenerate quadrilateral', () => {
    const collinear: Pt[] = [
      [0, 0],
      [100, 0],
      [200, 0],
      [300, 0],
    ]
    expect(homographyFromQuad(COURT_CORNERS_M, collinear)).toBeNull()
  })

  it('agrees with the backend on where the net lands', () => {
    // Points genuinely on Y = 10 m must report ~zero error.
    const toImage = homographyFromQuad(COURT_CORNERS_M, GOOD)!
    const onNet: Pt[] = [
      applyHomography(toImage, [0, NET_Y_M])!,
      applyHomography(toImage, [10, NET_Y_M])!,
    ]
    expect(netErrorMetres(GOOD, onNet)!).toBeLessThan(1e-6)
  })

  it('measures how far a misplaced net line is', () => {
    const toImage = homographyFromQuad(COURT_CORNERS_M, GOOD)!
    const off: Pt[] = [
      applyHomography(toImage, [0, 13])!,
      applyHomography(toImage, [10, 13])!,
    ]
    expect(netErrorMetres(GOOD, off)!).toBeCloseTo(3, 4)
  })
})

describe('quad validation', () => {
  it('accepts a realistic court quadrilateral', () => {
    expect(checkQuad(GOOD, FRAME_W, FRAME_H).valid).toBe(true)
  })

  it('rejects coincident corners', () => {
    const bad: Pt[] = [GOOD[0], [602, 301], GOOD[2], GOOD[3]]
    expect(checkQuad(bad, FRAME_W, FRAME_H).reason).toMatch(/coincidono/)
  })

  it('rejects a self-intersecting perimeter', () => {
    const bowtie: Pt[] = [GOOD[0], GOOD[2], GOOD[1], GOOD[3]]
    expect(checkQuad(bowtie, FRAME_W, FRAME_H).valid).toBe(false)
  })

  it('rejects the reversed order that would mirror the court', () => {
    const reversed: Pt[] = [GOOD[0], GOOD[3], GOOD[2], GOOD[1]]
    const check = checkQuad(reversed, FRAME_W, FRAME_H)
    expect(check.valid).toBe(false)
    expect(check.reason).toMatch(/specchiat/)
  })

  it('rejects a court covering too little of the frame', () => {
    const tiny: Pt[] = [
      [100, 100],
      [160, 100],
      [160, 150],
      [100, 150],
    ]
    expect(checkQuad(tiny, FRAME_W, FRAME_H).reason).toMatch(/piccola/)
  })

  it('treats the expected corner order as clockwise on screen', () => {
    expect(signedArea(GOOD)).toBeGreaterThan(0)
  })
})

describe('court overlay', () => {
  it('projects outline, net, service lines and centre line', () => {
    const lines = projectCourtLines(GOOD)
    expect(lines).toHaveLength(5)
    expect(lines.every(line => line.every(p => Number.isFinite(p[0]) && Number.isFinite(p[1])))).toBe(true)
  })

  it('draws the net between the two baselines', () => {
    const [outline, net] = projectCourtLines(GOOD)
    const netY = (net[0][1] + net[1][1]) / 2
    const farY = Math.min(...outline.map(p => p[1]))
    const nearY = Math.max(...outline.map(p => p[1]))
    expect(netY).toBeGreaterThan(farY)
    expect(netY).toBeLessThan(nearY)
  })

  it('returns nothing for an unprojectable quad', () => {
    expect(projectCourtLines([[0, 0], [100, 0], [200, 0], [300, 0]])).toEqual([])
  })
})
