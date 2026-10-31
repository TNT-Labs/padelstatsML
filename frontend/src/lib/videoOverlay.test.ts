import { describe, expect, it } from 'vitest'
import { ballAt, boxAt, clock, fitRect, lastAtOrBefore, placeTags, pointsAt } from './videoOverlay'

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

describe('pointsAt', () => {
  const times = [0, 0.2, 2.2]
  const p = (x: number, v: number) => Array.from({ length: 17 }, () => [x, 10, v]).flat()
  const points = [p(0, 90), p(10, 60), null]
  const gap = 0.5

  it('glides between two close samples, as sure as the less sure one', () => {
    const mid = pointsAt(times, points, 0.1, gap)!
    expect(mid.slice(0, 3)).toEqual([5, 10, 60])
  })
  it('keeps the nearer sample for a moment, then nothing', () => {
    expect(pointsAt(times, points, 0.3, gap)!.slice(0, 3)).toEqual([10, 10, 60])
    expect(pointsAt(times, points, 1.2, gap)).toBeNull()
  })
  it('is null without the posture', () => {
    expect(pointsAt(times, undefined, 0.1, gap)).toBeNull()
  })
})

describe('placeTags', () => {
  it('leaves apart tags where they are', () => {
    expect(placeTags([{ x: 0, bottom: 20, w: 40, h: 16 }, { x: 100, bottom: 20, w: 40, h: 16 }])).toEqual([20, 20])
  })
  it('moves down a tag that would cover another', () => {
    // A far player just behind a near one: tops at nearly the same height.
    const [near, far] = placeTags([{ x: 100, bottom: 50, w: 50, h: 16 }, { x: 120, bottom: 48, w: 50, h: 16 }])
    expect(far).toBe(48)
    expect(near).toBe(66)
  })
})

describe('ballAt', () => {
  const flights = [
    { t: [1.0, 1.033, 1.067, 1.1], x: [10, 20, 30, 40], y: [5, 5, 5, 5] },
    { t: [3.0, 3.033], x: [500, 510], y: [100, 101] },
  ]

  it('shows the ball with the path it came along', () => {
    const ball = ballAt(flights, 1.07)
    expect(ball?.at).toEqual([30, 5])
    expect(ball?.trail).toEqual([[10, 5], [20, 5], [30, 5]])
  })

  it('shows nothing between flights or before the first', () => {
    expect(ballAt(flights, 0.5)).toBeNull()
    expect(ballAt(flights, 2.0)).toBeNull()
    expect(ballAt([], 2.0)).toBeNull()
  })

  it('never draws a trail across two flights', () => {
    expect(ballAt(flights, 3.04)?.trail).toEqual([[500, 100], [510, 101]])
  })
})
