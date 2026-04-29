/**
 * Hook that manages the complete analysis flow: create → upload → start → poll.
 * Exposes unified state + progress (0..1).
 */
import * as FileSystem from 'expo-file-system'
import { useCallback, useRef, useState } from 'react'
import { api, Match, MatchStats } from '../services/api'

export type Phase = 'idle' | 'creating' | 'uploading' | 'processing' | 'done' | 'error'

interface State {
  phase: Phase
  progress: number
  match: Match | null
  stats: MatchStats | null
  error: string | null
}

const POLL_INTERVAL_MS = 5_000
const MAX_POLL_MIN     = 25
const MAX_POLLS        = (MAX_POLL_MIN * 60_000) / POLL_INTERVAL_MS

export function useMatchAnalysis() {
  const [state, setState] = useState<State>({
    phase: 'idle', progress: 0, match: null, stats: null, error: null,
  })
  const pollRef   = useRef<ReturnType<typeof setInterval> | null>(null)
  const pollCount = useRef(0)

  const stopPolling = () => {
    if (pollRef.current) {
      clearInterval(pollRef.current)
      pollRef.current = null
    }
    pollCount.current = 0
  }

  const analyze = useCallback(async (videoUri: string, title: string, players?: string[]) => {
    stopPolling()
    setState({ phase: 'creating', progress: 0, match: null, stats: null, error: null })

    try {
      // 1. Get file size for server-side validation before creating the match
      const fileInfo = await FileSystem.getInfoAsync(videoUri, { size: true })
      const fileSizeBytes = fileInfo.exists && 'size' in fileInfo ? fileInfo.size : undefined

      // 2. Create match + get presigned upload URL (API rejects oversized videos here)
      const init = await api.createMatch(title, players, fileSizeBytes)

      // 2. Upload directly to S3
      setState(s => ({ ...s, phase: 'uploading', progress: 0 }))
      await api.uploadVideo(init.upload_url, videoUri, p =>
        setState(s => ({ ...s, progress: p }))
      )

      // 3. Trigger ML processing
      const match = await api.startProcessing(init.match_id)
      setState(s => ({ ...s, phase: 'processing', progress: 0, match }))

      // 4. Poll until completion or failure
      pollRef.current = setInterval(async () => {
        pollCount.current += 1

        if (pollCount.current > MAX_POLLS) {
          stopPolling()
          setState(s => ({
            ...s,
            phase: 'error',
            error: `Timeout: analisi oltre ${MAX_POLL_MIN} minuti`,
          }))
          return
        }

        try {
          const m = await api.getMatch(init.match_id)
          setState(s => ({ ...s, match: m, progress: m.progress / 100 }))

          if (m.status === 'completed') {
            stopPolling()
            const stats = await api.getStats(init.match_id)
            setState(s => ({ ...s, phase: 'done', stats, progress: 1 }))
          } else if (m.status === 'failed') {
            stopPolling()
            setState(s => ({
              ...s,
              phase: 'error',
              error: m.error_message ?? 'Processing failed',
            }))
          }
        } catch {
          // Transient network error during polling — keep trying
          console.warn('Poll network error, retrying…')
        }
      }, POLL_INTERVAL_MS)

    } catch (e) {
      stopPolling()
      setState(s => ({ ...s, phase: 'error', error: (e as Error).message }))
    }
  }, [])

  const reset = useCallback(() => {
    stopPolling()
    setState({ phase: 'idle', progress: 0, match: null, stats: null, error: null })
  }, [])

  return { state, analyze, reset }
}
