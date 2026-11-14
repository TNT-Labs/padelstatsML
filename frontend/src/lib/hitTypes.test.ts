import { describe, expect, it } from 'vitest'
import { HIT_KINDS, hitsText, kindOfKey, nextUnkinded, setKind, type MarkedHit } from './hitTypes'

describe('kinds', () => {
  it('have a key each, no two alike, none of the marker\'s other keys', () => {
    const keys = HIT_KINDS.map(k => k.key)
    expect(new Set(keys).size).toBe(keys.length)
    for (const taken of ['1', '2', '3', '4', ',', '.', ' ', 'n']) expect(keys).not.toContain(taken)
    expect(kindOfKey('F')).toBe('volee_dritto')
    expect(kindOfKey('q')).toBeNull()
  })
})

describe('setKind', () => {
  it('gives and takes the kind of one hit', () => {
    const hits: MarkedHit[] = [[10, 0], [11, 2, 'lob']]
    expect(setKind(hits, 10, 'battuta')).toEqual([[10, 0, 'battuta'], [11, 2, 'lob']])
    expect(setKind(hits, 11, null)).toEqual([[10, 0], [11, 2]])
  })
})

describe('nextUnkinded', () => {
  const hits: MarkedHit[] = [[10, 0], [11, 1, 'dritto'], [12, 2], [13, 3, 'lob']]
  it('is the next hit still without a kind, from the start when none is after', () => {
    expect(nextUnkinded(hits, 10)).toEqual([12, 2])
    expect(nextUnkinded(hits, 12.5)).toEqual([10, 0])
    expect(nextUnkinded(setKind(setKind(hits, 10, 'smash'), 12, 'rovescio'), 0)).toBeNull()
  })
})

describe('hitsText', () => {
  it('writes the kind last, when there is one', () => {
    const text = hitsText('Set', [[65.86, 2], [67.16, 1, 'volee_rovescio']], pid => ['A', 'B', 'C'][pid], t => t.toFixed(2))
    expect(text.split('\n')).toEqual(['Set · colpi segnati 2 (con il tipo 1)', '65.86\tC\t2', '67.16\tB\t1\tvolee_rovescio'])
  })
})
