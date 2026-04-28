import { useCallback, useRef, useState } from 'react'
import { api, Match, MatchStats } from '../api'

export type Phase = 'idle' | 'creating' | 'uploading' | 'processing' | 'done' | 'error'

export interface AnalysisState {
  phase: Phase
  progress: number  // 0..1
  match: Match | null
  stats: MatchStats | null
  error: string | null
}

const POLL_MS      = 5_000
const MAX_POLL_MIN = 25            // abort after 25 minutes of processing
const MAX_POLLS    = (MAX_POLL_MIN * 60_000) / POLL_MS

export function useMatchAnalysis() {
  const [state, setState] = useState<AnalysisState>({
    phase: 'idle', progress: 0, match: null, stats: null, error: null,
  })
  const pollRef   = useRef<number | null>(null)
  const pollCount = useRef(0)

  const stopPoll = () => {
    if (pollRef.current !== null) {
      clearInterval(pollRef.current)
      pollRef.current = null
    }
    pollCount.current = 0
  }

  const analyze = useCallback(async (file: File, title: string, players?: string[]) => {
    stopPoll()
    setState({ phase: 'creating', progress: 0, match: null, stats: null, error: null })

    try {
      const init = await api.createMatch(title, players)

      setState(s => ({ ...s, phase: 'uploading', progress: 0 }))
      await api.uploadVideo(init.upload_url, file, p =>
        setState(s => ({ ...s, progress: p }))
      )

      const match = await api.startProcessing(init.match_id)
      setState(s => ({ ...s, phase: 'processing', progress: 0, match }))

      pollRef.current = window.setInterval(async () => {
        pollCount.current += 1

        // Safety timeout: abort after MAX_POLL_MIN minutes
        if (pollCount.current > MAX_POLLS) {
          stopPoll()
          setState(s => ({
            ...s,
            phase: 'error',
            error: `Timeout: l'analisi ha superato ${MAX_POLL_MIN} minuti`,
          }))
          return
        }

        try {
          const m = await api.getMatch(init.match_id)
          setState(s => ({ ...s, match: m, progress: m.progress / 100 }))

          if (m.status === 'completed') {
            stopPoll()
            const stats = await api.getStats(init.match_id)
            setState(s => ({ ...s, phase: 'done', stats, progress: 1 }))
          } else if (m.status === 'failed') {
            stopPoll()
            setState(s => ({
              ...s,
              phase: 'error',
              error: m.error_message ?? 'Analisi fallita',
            }))
          }
        } catch {
          // Transient network glitch — keep polling; error counted against timeout
        }
      }, POLL_MS)

    } catch (e) {
      stopPoll()
      setState(s => ({ ...s, phase: 'error', error: (e as Error).message }))
    }
  }, [])

  const reset = useCallback(() => {
    stopPoll()
    setState({ phase: 'idle', progress: 0, match: null, stats: null, error: null })
  }, [])

  return { state, analyze, reset }
}
