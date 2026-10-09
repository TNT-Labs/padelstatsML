/**
 * Where the API is. Empty VITE_API_URL (the normal case) means "the server
 * this page came from", under whatever path it is mounted at: '' on a Pi
 * reached directly, '/padel' behind shopbeautylab.it. The page is built with
 * relative asset URLs, so the same build works in both places.
 */
export const BASE = (import.meta.env.VITE_API_URL || appPath()).replace(/\/$/, '')

function appPath(): string {
  try {
    return new URL('.', document.baseURI).pathname
  } catch {
    return ''
  }
}

/** Every request that changes data carries it: the server refuses writes
 *  without it, which is what stops another site from forging them. */
const CSRF_HEADER = { 'X-Padel': '1' }

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
  /** Only for administrators, who see everyone's matches. */
  owner?: string | null
  created_at: string
  updated_at: string
}

export interface UploadInit {
  match_id: string
  upload_url: string
  chunk_size_bytes: number
}

export interface UploadStatus {
  received_bytes: number
  chunk_size_bytes: number
}

export type Role = 'admin' | 'utente'

export interface Me {
  id: string
  username: string
  role: Role
  must_change_password: boolean
}

export interface UserRow {
  id: string
  username: string
  role: Role
  active: boolean
  must_change_password: boolean
  locked: boolean
  last_login_at: string | null
  created_at: string
  matches: number
  sessions: number
}

export interface TemporaryPassword {
  user: UserRow
  temporary_password: string
}

export interface AuditRow {
  at: string
  username: string | null
  event: string
  detail: string | null
  ip: string | null
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

/** Each player's box through the video, sampled a few times a second. */
export interface PlayerBoxes {
  id: number
  team: number
  role?: 'drive' | 'reves' | null
  /** Video time of each sample, in seconds. */
  t: number[]
  /** x1, y1, x2, y2 of each sample, in the video's pixels, one after another. */
  box: number[]
}

export interface PlayerTracks {
  sample_hz: number
  frame: { width: number | null; height: number | null }
  players: PlayerBoxes[]
  /** Seconds at which the pairs changed ends. */
  changeovers: number[]
  /** False when the boxes were recomputed with a newer version of the code
   *  than the statistics on screen: the numbering may differ. */
  matches_stats: boolean
}

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message)
  }
}

/** Called when the server no longer recognises the session (401) or wants a
 *  new password first (403 on a data route): the app goes back to sign-in. */
let onSessionLost: (() => void) | null = null
export function setSessionLostHandler(handler: (() => void) | null) {
  onSessionLost = handler
}

async function req<T>(path: string, init?: RequestInit & { quiet?: boolean }): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    ...init,
    credentials: 'include',
    headers: {
      ...CSRF_HEADER,
      ...(init?.body ? { 'Content-Type': 'application/json' } : {}),
      ...init?.headers,
    },
  })
  if (!res.ok) {
    const message = await readError(res)
    if (!init?.quiet && (res.status === 401 || (res.status === 403 && /password/i.test(message)))) {
      onSessionLost?.()
    }
    throw new ApiError(res.status, message)
  }
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
  videoUrl: (id: string) => `${BASE}/api/matches/${id}/video`,
  getPlayerTracks: (id: string) => req<PlayerTracks>(`/api/matches/${id}/tracks`),

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

  getUploadStatus: (id: string) => req<UploadStatus>(`/api/matches/${id}/upload`),
  completeUpload: (id: string, totalBytes: number) =>
    req<Match>(`/api/matches/${id}/upload/complete`, {
      method: 'POST',
      body: JSON.stringify({ total_bytes: totalBytes }),
    }),

  // ── Account ──
  me: () => req<Me>('/api/auth/me', { quiet: true }),
  login: (username: string, password: string) =>
    req<Me>('/api/auth/login', {
      method: 'POST',
      quiet: true,
      body: JSON.stringify({ username, password }),
    }),
  logout: () => req<void>('/api/auth/logout', { method: 'POST', quiet: true }),
  changePassword: (currentPassword: string, newPassword: string) =>
    req<Me>('/api/auth/password', {
      method: 'POST',
      quiet: true,
      body: JSON.stringify({ current_password: currentPassword, new_password: newPassword }),
    }),

  // ── Administration ──
  listUsers: () => req<UserRow[]>('/api/admin/users'),
  createUser: (username: string, role: Role) =>
    req<TemporaryPassword>('/api/admin/users', {
      method: 'POST',
      body: JSON.stringify({ username, role }),
    }),
  updateUser: (id: string, patch: { role?: Role; active?: boolean }) =>
    req<UserRow>(`/api/admin/users/${id}`, { method: 'PATCH', body: JSON.stringify(patch) }),
  resetPassword: (id: string) =>
    req<TemporaryPassword>(`/api/admin/users/${id}/reset-password`, { method: 'POST' }),
  unlockUser: (id: string) => req<UserRow>(`/api/admin/users/${id}/unlock`, { method: 'POST' }),
  logoutUser: (id: string) => req<UserRow>(`/api/admin/users/${id}/logout`, { method: 'POST' }),
  deleteUser: (id: string) => req<void>(`/api/admin/users/${id}`, { method: 'DELETE' }),
  audit: (limit = 200) => req<AuditRow[]>(`/api/admin/audit?limit=${limit}`),
}

/** HTTP statuses worth retrying a piece on: the network, the tunnel or the
 *  Pi hiccuped, the piece itself is fine. */
const RETRYABLE = new Set([0, 408, 429, 500, 502, 503, 504, 520, 522, 523, 524])
const MAX_TRIES = 6

/** PUT one piece with XHR: only XHR reports upload progress. */
function putPiece(
  url: string,
  piece: Blob,
  onProgress: (loaded: number) => void,
  signal?: AbortSignal,
): Promise<UploadStatus> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest()
    xhr.open('PUT', url)
    xhr.withCredentials = true
    xhr.setRequestHeader('Content-Type', 'application/octet-stream')
    xhr.setRequestHeader('X-Padel', '1')
    xhr.upload.onprogress = e => onProgress(e.loaded)
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          return resolve(JSON.parse(xhr.responseText) as UploadStatus)
        } catch {
          return reject(new ApiError(0, 'Risposta non valida dal server'))
        }
      }
      let message = `Upload fallito (${xhr.status})`
      try {
        const detail = JSON.parse(xhr.responseText)?.detail
        if (typeof detail === 'string') message = detail
      } catch {
        /* keep the generic message */
      }
      if (xhr.status === 401) onSessionLost?.()
      reject(new ApiError(xhr.status, message))
    }
    xhr.onerror = () => reject(new ApiError(0, 'Errore di rete durante il caricamento'))
    xhr.ontimeout = xhr.onerror
    const abort = () => xhr.abort()
    signal?.addEventListener('abort', abort, { once: true })
    xhr.onabort = () => reject(new DOMException('Caricamento annullato', 'AbortError'))
    xhr.onloadend = () => signal?.removeEventListener('abort', abort)
    xhr.send(piece)
  })
}

/** First wait before trying a piece again; doubles at each attempt. */
export const uploadTuning = { retryBaseMs: 1000 }

const wait = (ms: number) => new Promise(resolve => setTimeout(resolve, ms))

export interface UploadProgress {
  /** 0..1 of the whole file. */
  fraction: number
  /** Set while waiting to try a piece again after a network problem. */
  retrying: number | null
}

/**
 * Send the video in pieces (Cloudflare refuses any request over 100 MB, and a
 * dropped connection then costs one piece, not the whole upload). Starts from
 * whatever the server already has, so it also resumes an interrupted upload.
 */
export async function uploadInPieces(
  matchId: string,
  file: File,
  onProgress: (p: UploadProgress) => void,
  { signal, fromZero = false }: { signal?: AbortSignal; fromZero?: boolean } = {},
): Promise<Match> {
  const status = await api.getUploadStatus(matchId)
  const pieceSize = Math.max(1, status.chunk_size_bytes)
  // offset=0 makes the server discard whatever it holds and start over.
  let offset = !fromZero && status.received_bytes <= file.size ? status.received_bytes : 0
  let tries = 0

  while (offset < file.size) {
    const end = Math.min(offset + pieceSize, file.size)
    const url = `${BASE}/api/matches/${matchId}/upload?offset=${offset}`
    const from = offset
    try {
      const result = await putPiece(
        url,
        file.slice(from, end),
        loaded => onProgress({ fraction: (from + loaded) / file.size, retrying: null }),
        signal,
      )
      offset = result.received_bytes
      tries = 0
      onProgress({ fraction: offset / file.size, retrying: null })
    } catch (e) {
      if (e instanceof DOMException && e.name === 'AbortError') throw e
      const err = e as ApiError
      // 409: the server has a different amount than we thought (the answer
      // to a piece was lost): carry on from what it actually has.
      if (err.status !== 409 && !RETRYABLE.has(err.status)) throw err
      if (err.status !== 409 && ++tries >= MAX_TRIES) throw err
      if (err.status !== 409) {
        onProgress({ fraction: offset / file.size, retrying: tries })
        await wait(Math.min(30_000, uploadTuning.retryBaseMs * 2 ** (tries - 1)))
      }
      const fresh = await api.getUploadStatus(matchId)
      const next = fresh.received_bytes <= file.size ? fresh.received_bytes : 0
      // A conflict that does not move the position is not about the
      // position (e.g. the match no longer accepts an upload): give up.
      if (err.status === 409 && next === offset) throw err
      offset = next
    }
  }
  return api.completeUpload(matchId, file.size)
}
