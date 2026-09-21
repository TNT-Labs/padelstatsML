import { useCallback, useEffect, useRef, useState } from 'react'
import { api, localiseUploadUrl, type Match, type MatchStatus } from '../api'

const POLL_MS = 5_000
/** Analysis on a Pi 5 is measured in hours, not minutes; this only guards
 *  against a worker that died without updating the row. */
const MAX_POLL_HOURS = 6

const ACTIVE: MatchStatus[] = ['queued', 'analyzing']

/** Poll a single match while its analysis is running. */
export function useMatch(matchId: string | null) {
  const [match, setMatch] = useState<Match | null>(null)
  const [error, setError] = useState<string | null>(null)
  const timer = useRef<number | null>(null)
  const started = useRef<number>(Date.now())

  const refresh = useCallback(async () => {
    if (!matchId) return null
    try {
      const fresh = await api.getMatch(matchId)
      setMatch(fresh)
      setError(null)
      return fresh
    } catch (e) {
      // A transient network blip must not wipe the last known state; the
      // Pi is on the same LAN and usually comes straight back.
      setError((e as Error).message)
      return null
    }
  }, [matchId])

  useEffect(() => {
    if (!matchId) {
      setMatch(null)
      return
    }
    started.current = Date.now()
    let cancelled = false

    const stop = () => {
      if (timer.current !== null) {
        window.clearInterval(timer.current)
        timer.current = null
      }
    }

    const tick = async () => {
      const fresh = await refresh()
      if (cancelled) return
      if (fresh && !ACTIVE.includes(fresh.status)) stop()
      if (Date.now() - started.current > MAX_POLL_HOURS * 3_600_000) {
        stop()
        setError(`L'analisi supera le ${MAX_POLL_HOURS} ore: controlla il worker.`)
      }
    }

    void tick()
    timer.current = window.setInterval(tick, POLL_MS)
    return () => {
      cancelled = true
      stop()
    }
  }, [matchId, refresh])

  return { match, error, refresh }
}

export type UploadPhase = 'idle' | 'creating' | 'uploading' | 'done' | 'error'

export interface UploadState {
  phase: UploadPhase
  progress: number
  matchId: string | null
  error: string | null
}

/** Create a match and stream the video to the Pi. Analysis is NOT started
 *  here: the court has to be calibrated first. */
export function useUpload() {
  const [state, setState] = useState<UploadState>({
    phase: 'idle',
    progress: 0,
    matchId: null,
    error: null,
  })

  const upload = useCallback(async (file: File, title: string) => {
    setState({ phase: 'creating', progress: 0, matchId: null, error: null })
    let matchId: string | null = null
    try {
      const init = await api.createMatch(title, file.size)
      matchId = init.match_id
      setState(s => ({ ...s, phase: 'uploading', matchId }))

      await api.uploadVideo(localiseUploadUrl(init.upload_url, init.match_id), file, p =>
        setState(s => ({ ...s, progress: p })),
      )
      setState({ phase: 'done', progress: 1, matchId, error: null })
      return matchId
    } catch (e) {
      setState({ phase: 'error', progress: 0, matchId, error: (e as Error).message })
      return null
    }
  }, [])

  const reset = useCallback(
    () => setState({ phase: 'idle', progress: 0, matchId: null, error: null }),
    [],
  )

  return { state, upload, reset }
}
