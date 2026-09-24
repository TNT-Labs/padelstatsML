import { describe, expect, it } from 'vitest'
import { apiAsset, kmh, playerPlace } from './format'

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
    expect(playerPlace(0, 'reves')).toBe('Coppia vicina a inizio video · lato revés')
    expect(playerPlace(1, 'drive')).toBe('Coppia lontana a inizio video · lato drive')
  })

  it('leaves the side out for results found by linking alone', () => {
    expect(playerPlace(0)).toBe('Coppia vicina a inizio video')
    expect(playerPlace(1, null)).toBe('Coppia lontana a inizio video')
  })
})
