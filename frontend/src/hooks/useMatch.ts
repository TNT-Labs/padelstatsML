import { useCallback, useEffect, useRef, useState } from 'react'
import { api, uploadInPieces, type Match, type MatchStatus } from '../api'

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

export type UploadPhase = 'idle' | 'creating' | 'uploading' | 'checking' | 'done' | 'error'

export interface UploadState {
  phase: UploadPhase
  progress: number
  /** Attempt number while a piece waits to be sent again. */
  retrying: number | null
  matchId: string | null
  error: string | null
}

const IDLE: UploadState = { phase: 'idle', progress: 0, retrying: null, matchId: null, error: null }

/** Which file an unfinished upload was made of, so that resuming it with a
 *  different file starts over instead of gluing two videos together. */
function fileKey(file: File): string {
  return `${file.name}|${file.size}|${file.lastModified}`
}
function rememberFile(matchId: string, file: File) {
  try {
    localStorage.setItem(`padel-upload-${matchId}`, fileKey(file))
  } catch {
    /* private mode: resuming just starts over */
  }
}
function sameFile(matchId: string, file: File): boolean {
  try {
    return localStorage.getItem(`padel-upload-${matchId}`) === fileKey(file)
  } catch {
    return false
  }
}
function forgetFile(matchId: string) {
  try {
    localStorage.removeItem(`padel-upload-${matchId}`)
  } catch {
    /* nothing to forget */
  }
}

/** Create a match (or take an unfinished one) and send the video to the Pi
 *  in pieces. Analysis is NOT started here: the court has to be calibrated
 *  first. */
export function useUpload() {
  const [state, setState] = useState<UploadState>(IDLE)
  const abort = useRef<AbortController | null>(null)

  // Leaving the screen stops the transfer; the pieces already sent stay on
  // the Pi and the upload can be resumed from the match list.
  useEffect(() => () => abort.current?.abort(), [])

  const upload = useCallback(async (file: File, title: string, resumeId?: string) => {
    setState({ ...IDLE, phase: 'creating' })
    let matchId: string | null = resumeId ?? null
    const controller = new AbortController()
    abort.current = controller
    try {
      if (!matchId) matchId = (await api.createMatch(title, file.size)).match_id
      const id = matchId
      // Resuming with a file that is not provably the one started with:
      // begin again from byte 0 rather than glue two videos together.
      const fromZero = resumeId !== undefined && !sameFile(id, file)
      rememberFile(id, file)
      setState(s => ({ ...s, phase: 'uploading', matchId: id }))

      await uploadInPieces(
        id,
        file,
        p =>
          setState(s => ({
            ...s,
            progress: p.fraction,
            retrying: p.retrying,
            phase: p.fraction >= 1 ? 'checking' : 'uploading',
          })),
        { signal: controller.signal, fromZero },
      )
      forgetFile(id)
      setState({ phase: 'done', progress: 1, retrying: null, matchId: id, error: null })
      return id
    } catch (e) {
      if (e instanceof DOMException && e.name === 'AbortError') return null
      setState({ ...IDLE, phase: 'error', matchId, error: (e as Error).message })
      return null
    }
  }, [])

  const reset = useCallback(() => setState(IDLE), [])

  return { state, upload, reset }
}
