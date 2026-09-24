import { describe, expect, it } from 'vitest'
import { boxAt, clock, fitRect, lastAtOrBefore } from './videoOverlay'

describe('lastAtOrBefore', () => {
  const times = [0, 0.2, 0.4, 1.0]
  it('finds the sample at or just before a time', () => {
    expect(lastAtOrBefore(times, 0.2)).toBe(1)
    expect(lastAtOrBefore(times, 0.3)).toBe(1)
    expect(lastAtOrBefore(times, 5)).toBe(3)
  })
  it('is -1 before the first sample', () => {
    expect(lastAtOrBefore(times, -0.1)).toBe(-1)
    expect(lastAtOrBefore([], 1)).toBe(-1)
  })
})

describe('boxAt', () => {
  // Samples at 5 Hz, then the player lost for two seconds.
  const times = [0, 0.2, 2.2]
  const boxes = [0, 0, 10, 20, 10, 0, 20, 20, 100, 0, 110, 20]
  const gap = 0.5

  it('glides between two close samples', () => {
    expect(boxAt(times, boxes, 0.1, gap)).toEqual([5, 0, 15, 20])
  })
  it('keeps the nearer sample for a moment past a gap', () => {
    expect(boxAt(times, boxes, 0.3, gap)).toEqual([10, 0, 20, 20])
    expect(boxAt(times, boxes, 2.0, gap)).toEqual([100, 0, 110, 20])
  })
  it('shows nothing while the player was not tracked', () => {
    expect(boxAt(times, boxes, 1.2, gap)).toBeNull()
    expect(boxAt(times, boxes, -1, gap)).toBeNull()
    expect(boxAt(times, boxes, 9, gap)).toBeNull()
  })
})

describe('fitRect', () => {
  it('letterboxes a wide frame in a tall element', () => {
    expect(fitRect(1920, 1080, 960, 1080)).toEqual({ scale: 0.5, offsetX: 0, offsetY: 270 })
  })
  it('pillarboxes a frame in a wider element', () => {
    expect(fitRect(1920, 1080, 2000, 540)).toEqual({ scale: 0.5, offsetX: 520, offsetY: 0 })
  })
})

describe('clock', () => {
  it('writes minutes and seconds', () => {
    expect(clock(0)).toBe('0:00')
    expect(clock(754.9)).toBe('12:34')
  })
})
