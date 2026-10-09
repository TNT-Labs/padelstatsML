import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { uploadInPieces, uploadTuning } from './api'

/**
 * A fake Pi: what it has received, and what each attempt to send a piece does.
 * `plan` decides the fate of each PUT in order: 'ok', 'drop' (network error,
 * nothing arrives), 'lost' (the piece arrives but the answer does not).
 */
function fakeServer(plan: Array<'ok' | 'drop' | 'lost'>, uploadable = true) {
  const state = { received: 0, sent: [] as number[], completed: false }

  vi.stubGlobal('fetch', async (url: string, init?: RequestInit) => {
    const json = (body: unknown, status = 200) =>
      new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } })
    if (url.endsWith('/upload')) return json({ received_bytes: uploadable ? state.received : 0, chunk_size_bytes: 4 })
    if (url.endsWith('/upload/complete') && init?.method === 'POST') {
      state.completed = true
      return json({ id: 'm', status: 'needs_calibration' })
    }
    return json({ detail: 'inatteso' }, 500)
  })

  class FakeXHR {
    status = 0
    responseText = ''
    upload: { onprogress?: (e: { loaded: number }) => void } = {}
    onload?: () => void
    onerror?: () => void
    onabort?: () => void
    ontimeout?: () => void
    onloadend?: () => void
    withCredentials = false
    private url = ''
    open(_method: string, url: string) {
      this.url = url
    }
    setRequestHeader() {}
    abort() {}
    async send(piece: Blob) {
      const offset = Number(new URL(this.url, 'http://x').searchParams.get('offset'))
      state.sent.push(offset)
      const fate = plan.shift() ?? 'ok'
      await Promise.resolve()
      // Like the real server: offset 0 starts the file over.
      if (uploadable && offset === 0) state.received = 0
      if (!uploadable || offset !== state.received) {
        this.status = 409
        this.responseText = JSON.stringify({ detail: 'Posizione non valida' })
        this.onload?.()
      } else if (fate === 'drop') {
        this.onerror?.()
      } else {
        state.received += piece.size
        if (fate === 'lost') this.onerror?.()
        else {
          this.status = 200
          this.responseText = JSON.stringify({ received_bytes: state.received, chunk_size_bytes: 4 })
          this.onload?.()
        }
      }
      this.onloadend?.()
    }
  }
  vi.stubGlobal('XMLHttpRequest', FakeXHR)
  return state
}

const video = () => new File([new Uint8Array(10)], 'partita.mp4')

describe('uploadInPieces', () => {
  beforeEach(() => {
    uploadTuning.retryBaseMs = 0
  })
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('sends the file in pieces and completes it', async () => {
    const server = fakeServer([])
    const fractions: number[] = []
    await uploadInPieces('m', video(), p => fractions.push(p.fraction))
    expect(server.sent).toEqual([0, 4, 8])
    expect(server.received).toBe(10)
    expect(server.completed).toBe(true)
    expect(fractions[fractions.length - 1]).toBe(1)
  })

  it('tries a dropped piece again', async () => {
    const server = fakeServer(['ok', 'drop'])
    const retries: (number | null)[] = []
    await uploadInPieces('m', video(), p => retries.push(p.retrying))
    expect(server.sent).toEqual([0, 4, 4, 8])
    expect(retries).toContain(1)
    expect(server.completed).toBe(true)
  })

  it('carries on from what the server has when an answer is lost', async () => {
    const server = fakeServer(['ok', 'lost'])
    await uploadInPieces('m', video(), () => {})
    // The piece at 4 arrived: it is not sent twice.
    expect(server.sent).toEqual([0, 4, 8])
    expect(server.received).toBe(10)
  })

  it('resumes where an earlier upload stopped', async () => {
    const server = fakeServer([])
    server.received = 8
    await uploadInPieces('m', video(), () => {})
    expect(server.sent).toEqual([8])
  })

  it('starts over when asked to, whatever the server holds', async () => {
    const server = fakeServer([])
    server.received = 8
    await uploadInPieces('m', video(), () => {}, { fromZero: true })
    expect(server.sent).toEqual([0, 4, 8])
  })

  it('gives up on a conflict that is not about the position', async () => {
    const server = fakeServer([], false)
    await expect(uploadInPieces('m', video(), () => {})).rejects.toThrow('Posizione non valida')
    expect(server.completed).toBe(false)
  })

  it('gives up after repeated network failures', async () => {
    const server = fakeServer(['drop', 'drop', 'drop', 'drop', 'drop', 'drop'])
    await expect(uploadInPieces('m', video(), () => {})).rejects.toThrow('rete')
    expect(server.sent).toHaveLength(6)
  })
})
