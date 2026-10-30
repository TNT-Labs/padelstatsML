import { describe, expect, it } from 'vitest'
import { apiAsset, kmh, parseClock, playerPlace, strokeCheck, strokeName } from './format'

describe('parseClock', () => {
  it('reads seconds, minutes and hours', () => {
    expect(parseClock('95')).toBe(95)
    expect(parseClock('1:35')).toBe(95)
    expect(parseClock(' 0:05 ')).toBe(5)
    expect(parseClock('1:01:35')).toBe(3695)
  })

  it('refuses what is not a time', () => {
    expect(parseClock('')).toBeNull()
    expect(parseClock('1:75')).toBeNull()
    expect(parseClock('a:10')).toBeNull()
    expect(parseClock('1:2:3:4')).toBeNull()
    expect(parseClock('-5')).toBeNull()
  })
})

describe('kmh', () => {
  it('converts metres per second', () => {
    expect(kmh(1)).toBe('3.6 km/h')
    expect(kmh(5.5)).toBe('19.8 km/h')
    expect(kmh(0)).toBe('0.0 km/h')
  })
})

describe('apiAsset', () => {
  it('keeps a relative path, under the configured API origin', () => {
    expect(apiAsset('/api/matches/a/crops/0?v=3', '')).toBe('/api/matches/a/crops/0?v=3')
    expect(apiAsset('/api/matches/a/crops/0?v=3', 'http://pi5:8000')).toBe(
      'http://pi5:8000/api/matches/a/crops/0?v=3',
    )
  })

  it('drops the host of an absolute URL built on a wrong API_BASE_URL', () => {
    expect(apiAsset('http://padelpi.local:8000/api/matches/a/crops/1', '')).toBe(
      '/api/matches/a/crops/1',
    )
  })
})

describe('playerPlace', () => {
  it('names the pair by where it started, and the side when known', () => {
    expect(playerPlace(0, 'reves')).toBe('Coppia vicina a inizio analisi · lato revés')
    expect(playerPlace(1, 'drive')).toBe('Coppia lontana a inizio analisi · lato drive')
  })

  it('leaves the side out for results found by linking alone', () => {
    expect(playerPlace(0)).toBe('Coppia vicina a inizio analisi')
    expect(playerPlace(1, null)).toBe('Coppia lontana a inizio analisi')
  })
})

describe('strokeCheck', () => {
  const strokes = [
    { key: '0:10', kind: 'overhead' as const },
    { key: '0:20', kind: 'overhead' as const },
    { key: '1:30', kind: 'lob' as const },
    { key: '1:40', kind: 'lob' as const },
  ]
  it('measures the checked ones only', () => {
    const labels = { '0:10': 'bandeja', '0:20': 'nulla', '1:30': 'smash' }
    expect(strokeCheck(strokes, labels)).toEqual({ checked: 3, real: 2, rightKind: 1 })
    expect(strokeCheck(strokes, null)).toEqual({ checked: 0, real: 0, rightKind: 0 })
  })
})

describe('strokes on the ground', () => {
  it('are named, volleys too', () => {
    expect(strokeName({ kind: 'forehand' })).toBe('dritto')
    expect(strokeName({ kind: 'backhand', volley: true })).toBe('rovescio volée')
  })
  it('agree with what a person saw', () => {
    const strokes = [
      { key: '0:1', kind: 'forehand' as const },
      { key: '0:2', kind: 'backhand' as const, volley: true },
      { key: '0:3', kind: 'backhand' as const },
    ]
    const labels = { '0:1': 'dritto', '0:2': 'volee', '0:3': 'dritto' }
    expect(strokeCheck(strokes, labels)).toEqual({ checked: 3, real: 3, rightKind: 2 })
  })
})
