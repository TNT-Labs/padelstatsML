const BASE = import.meta.env.VITE_API_URL ?? ''

export type MatchStatus =
  | 'uploading'
  | 'needs_calibration'
  | 'ready'
  | 'queued'
  | 'analyzing'
  | 'completed'
  | 'failed'

export interface Match {
  id: string
  title: string
  status: MatchStatus
  progress: number
  progress_message: string | null
  error_message: string | null
  duration_seconds: number | null
  fps: number | null
  width: number | null
  height: number | null
  player_names: string[] | null
  calibrated: boolean
  created_at: string
  updated_at: string
}

export interface UploadInit {
  match_id: string
  upload_url: string
}

export type Point = [number, number]

export interface CalibrationSuggestion {
  corners_px: Point[]
  method: 'lines' | 'default' | 'saved'
  confidence: number
  frame_size: [number, number]
  corner_labels: string[]
}

export interface CalibrationResult {
  ok: boolean
  corners_px: Point[]
  net_error_m: number | null
  preset_id: string | null
  message: string | null
}

export interface CameraPreset {
  id: string
  name: string
  corners_px: Point[]
  frame_width: number
  frame_height: number
  net_px: Point[] | null
  times_used: number
  created_at: string
}

export interface ZoneShare {
  net: number
  mid: number
  back: number
}

export interface PlayerStats {
  team: number
  samples: number
  tracked_ratio: number
  distance_m: number
  distance_rally_m: number
  avg_speed_ms: number
  peak_speed_ms: number
  coverage_m2: number
  zone_pct: ZoneShare
  rejected_steps: number
  source_tracklets: number
  /** Side within the pair; absent when players were found by linking alone. */
  role?: 'drive' | 'reves' | null
  crop_url?: string | null
}

export interface MatchSummary {
  analysed_s: number
  rallies_count: number
  total_rally_s: number
  avg_rally_s: number
  median_rally_s: number
  longest_rally_s: number
  active_ratio: number
  players_found: number
}

export interface IdentityCue {
  cue: 'reid' | 'colore'
  same?: number
  different?: number
  veto?: number
  reason?: string
  /** How the four players were found: by their place on court, or by linking tracks. */
  method?: 'ruoli' | 'collegamento'
}

export interface DataQuality {
  calibration_source: string
  net_error_m: number | null
  sample_hz: number
  frames_sampled: number
  players_found: number
  clusters_found: number
  side_changes: number
  detector_model: string
  /** Absent in results produced before re-identification existed. */
  identity_cue?: IdentityCue
  metrics_tier: number
  excluded_metrics: string[]
  warnings: string[]
}

export interface Rally {
  index: number
  start_s: number
  end_s: number
  duration_s: number
  players_tracked: number
}

export interface MatchStats {
  match_id: string
  title: string
  player_names: string[] | null
  per_player: Record<string, PlayerStats>
  heatmaps: Record<string, [number, number, number][]>
  rallies: Rally[]
  summary: MatchSummary
  data_quality: DataQuality
}

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message)
  }
}

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    ...init,
    headers: init?.body ? { 'Content-Type': 'application/json', ...init?.headers } : init?.headers,
  })
  if (!res.ok) throw new ApiError(res.status, await readError(res))
  if (res.status === 204) return undefined as T
  return (await res.json()) as T
}

/** FastAPI reports failures as `detail`, which is what the user needs to read. */
async function readError(res: Response): Promise<string> {
  try {
    const body = await res.json()
    if (typeof body.detail === 'string') return body.detail
    if (Array.isArray(body.detail)) return body.detail.map((d: any) => d.msg).join(', ')
    return JSON.stringify(body)
  } catch {
    return `Errore ${res.status}`
  }
}

export const api = {
  listMatches: () => req<Match[]>('/api/matches'),
  getMatch: (id: string) => req<Match>(`/api/matches/${id}`),
  getStats: (id: string) => req<MatchStats>(`/api/matches/${id}/stats`),

  createMatch: (title: string, fileSizeBytes?: number) =>
    req<UploadInit>('/api/matches', {
      method: 'POST',
      body: JSON.stringify({ title, file_size_bytes: fileSizeBytes }),
    }),

  updateMatch: (id: string, patch: { title?: string; player_names?: string[] }) =>
    req<Match>(`/api/matches/${id}`, { method: 'PATCH', body: JSON.stringify(patch) }),

  startAnalysis: (id: string) => req<Match>(`/api/matches/${id}/start`, { method: 'POST' }),
  deleteMatch: (id: string) => req<void>(`/api/matches/${id}`, { method: 'DELETE' }),

  keyframeUrl: (id: string) => `${BASE}/api/matches/${id}/keyframe`,

  getSuggestion: (id: string) =>
    req<CalibrationSuggestion>(`/api/matches/${id}/calibration/suggestion`),

  submitCalibration: (
    id: string,
    body: {
      corners_px?: Point[]
      net_px?: Point[] | null
      preset_id?: string
      save_as_preset?: string
    },
  ) =>
    req<CalibrationResult>(`/api/matches/${id}/calibration`, {
      method: 'POST',
      body: JSON.stringify(body),
    }),

  listPresets: () => req<CameraPreset[]>('/api/presets'),
  deletePreset: (id: string) => req<void>(`/api/presets/${id}`, { method: 'DELETE' }),

  /** XHR rather than fetch: only XHR reports upload progress, and these
   *  files are hundreds of megabytes over Wi-Fi. */
  uploadVideo(uploadUrl: string, file: File, onProgress?: (fraction: number) => void) {
    return new Promise<void>((resolve, reject) => {
      const xhr = new XMLHttpRequest()
      xhr.open('PUT', uploadUrl)
      xhr.setRequestHeader('Content-Type', 'application/octet-stream')
      if (onProgress) {
        xhr.upload.onprogress = e => {
          if (e.lengthComputable) onProgress(e.loaded / e.total)
        }
      }
      xhr.onload = () => {
        if (xhr.status < 300) return resolve()
        let message = `Upload fallito (${xhr.status})`
        try {
          const detail = JSON.parse(xhr.responseText)?.detail
          if (typeof detail === 'string') message = detail
        } catch {
          /* keep the generic message */
        }
        reject(new ApiError(xhr.status, message))
      }
      xhr.onerror = () => reject(new ApiError(0, 'Errore di rete durante il caricamento'))
      xhr.send(file)
    })
  },
}

/** Upload URLs are absolute (built from API_BASE_URL on the Pi). When the UI
 *  is served by the same host we prefer the relative path, so the app keeps
 *  working over whatever hostname or IP the browser actually used. */
export function localiseUploadUrl(url: string, matchId: string): string {
  try {
    const parsed = new URL(url, window.location.origin)
    if (parsed.origin !== window.location.origin) {
      return `${BASE}/api/matches/${matchId}/video`
    }
    return parsed.pathname
  } catch {
    return `${BASE}/api/matches/${matchId}/video`
  }
}
