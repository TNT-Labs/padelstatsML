/**
 * Hook che incapsula il flow completo: createMatch -> upload -> start -> poll.
 * Espone stato + progress unificato.
 */
import { useCallback, useRef, useState } from 'react';
import { api, Match, MatchStats } from '../services/api';

export type Phase = 'idle' | 'creating' | 'uploading' | 'processing' | 'done' | 'error';

interface State {
  phase: Phase;
  progress: number; // 0..1
  match: Match | null;
  stats: MatchStats | null;
  error: string | null;
}

const POLL_INTERVAL_MS = 5000;

export function useMatchAnalysis() {
  const [state, setState] = useState<State>({
    phase: 'idle',
    progress: 0,
    match: null,
    stats: null,
    error: null,
  });
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const stopPolling = () => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  };

  const analyze = useCallback(async (videoUri: string, title: string, players?: string[]) => {
    stopPolling();
    setState({ phase: 'creating', progress: 0, match: null, stats: null, error: null });

    try {
      // 1. Create match + get presigned URL
      const init = await api.createMatch(title, players);

      // 2. Upload video direttamente a S3
      setState((s) => ({ ...s, phase: 'uploading', progress: 0 }));
      await api.uploadVideo(init.upload_url, videoUri, (p) => {
        setState((s) => ({ ...s, progress: p }));
      });

      // 3. Trigger processing
      const match = await api.startProcessing(init.match_id);
      setState((s) => ({ ...s, phase: 'processing', progress: 0, match }));

      // 4. Poll status
      pollRef.current = setInterval(async () => {
        try {
          const m = await api.getMatch(init.match_id);
          setState((s) => ({ ...s, match: m, progress: m.progress / 100 }));

          if (m.status === 'completed') {
            stopPolling();
            const stats = await api.getStats(init.match_id);
            setState((s) => ({ ...s, phase: 'done', stats, progress: 1 }));
          } else if (m.status === 'failed') {
            stopPolling();
            setState((s) => ({
              ...s,
              phase: 'error',
              error: m.error_message ?? 'Processing failed',
            }));
          }
        } catch (e) {
          // network glitch durante polling: continua, non fallire subito
          console.warn('Poll error:', e);
        }
      }, POLL_INTERVAL_MS);
    } catch (e) {
      stopPolling();
      setState((s) => ({ ...s, phase: 'error', error: (e as Error).message }));
    }
  }, []);

  const reset = useCallback(() => {
    stopPolling();
    setState({ phase: 'idle', progress: 0, match: null, stats: null, error: null });
  }, []);

  return { state, analyze, reset };
}
