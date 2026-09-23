import { describe, expect, it } from 'vitest'
import { apiAsset, kmh } from './format'

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
