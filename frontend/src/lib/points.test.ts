import { describe, expect, it } from 'vitest'
import { addPoint, pairsOf, pointAt, pointsText, setWinner, type MarkedPoint } from './points'

describe('addPoint', () => {
  it('keeps the points in time order, to the hundredth', () => {
    const one = addPoint([[30, 40, null]], 10.004, 20.456)
    expect(one).toEqual({ points: [[10, 20.46, null], [30, 40, null]] })
  })

  it('refuses an end before the serve and a point over another', () => {
    expect('error' in addPoint([], 20, 10)).toBe(true)
    expect('error' in addPoint([], 10, 10)).toBe(true)
    expect('error' in addPoint([[10, 20, null]], 19, 25)).toBe(true)
    expect('error' in addPoint([[10, 20, null]], 5, 30)).toBe(true)
    // Touching is not overlapping.
    expect(addPoint([[10, 20, null]], 20, 25)).toEqual({ points: [[10, 20, null], [20, 25, null]] })
  })
})

describe('pointAt', () => {
  const points: MarkedPoint[] = [[10, 20, null], [40, 50, null]]
  it('is the point watched, or the last one just ended', () => {
    expect(pointAt(points, 15)).toBe(0)
    expect(pointAt(points, 25)).toBe(0)
    expect(pointAt(points, 45)).toBe(1)
    expect(pointAt(points, 60)).toBe(1)
  })
  it('is none before the first point or long after the last', () => {
    expect(pointAt(points, 5)).toBe(-1)
    expect(pointAt(points, 80)).toBe(-1)
    expect(pointAt([], 10)).toBe(-1)
  })
})

describe('setWinner', () => {
  it('sets, sorts and clears the pair of one point only', () => {
    const points: MarkedPoint[] = [[10, 20, null], [40, 50, null]]
    const won = setWinner(points, 1, [3, 1])
    expect(won).toEqual([[10, 20, null], [40, 50, [1, 3]]])
    expect(setWinner(won, 1, null)).toEqual(points)
  })
})

describe('pairsOf', () => {
  it('groups the players by team', () => {
    expect(pairsOf({ 0: { team: 1 }, 1: { team: 0 }, 2: { team: 1 }, 3: { team: 0 } })).toEqual([[0, 2], [1, 3]])
  })
  it('is null without two pairs of two', () => {
    expect(pairsOf({ 0: { team: 0 }, 1: { team: 0 }, 2: { team: 1 } })).toBeNull()
    expect(pairsOf({})).toBeNull()
  })
})

describe('pointsText', () => {
  it('writes what the evaluation script reads', () => {
    const text = pointsText('Giovedì', [[62.3, 69.75, [1, 2]], [80, 91.1, null]], pid => ['A', 'B', 'C', 'D'][pid])
    expect(text.split('\n')).toEqual([
      'Giovedì · punti segnati 2',
      '1:02.30\t1:09.75\tB e C\t1,2',
      '1:20.00\t1:31.10\t-',
    ])
  })
})
