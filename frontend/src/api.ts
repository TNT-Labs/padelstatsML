const BASE = import.meta.env.VITE_API_URL ?? ''

export type MatchStatus = 'uploading' | 'queued' | 'processing' | 'completed' | 'failed'

export interface Match {
  id: string
  title: string
  status: MatchStatus
  progress: number
  error_message: string | null
  duration_seconds: number | null
  player_names: string[] | null
  created_at: string
  updated_at: string
}

export interface UploadInit {
  match_id: string
  upload_url: string
  s3_key: string
}

export interface PlayerStats {
  distance_m: number
  winners: number
  errors: number
  shots: { smash: number; volley: number; bandeja: number; other: number }
  crop_url?: string
}

export interface MatchStats {
  match_id: string
  per_player: Record<string, PlayerStats>
  heatmaps: Record<string, [number, number, number][]>
  rallies_count: number
  total_shots: number
}

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    ...init,
    headers: { 'Content-Type': 'application/json', ...init?.headers },
  })
  if (!res.ok) {
    const txt = await res.text().catch(() => String(res.status))
    throw new Error(`${res.status}: ${txt}`)
  }
  if (res.status === 204) return undefined as T
  return res.json() as Promise<T>
}

export const api = {
  listMatches: () => req<Match[]>('/matches'),
  getMatch:    (id: string) => req<Match>(`/matches/${id}`),
  getStats:    (id: string) => req<MatchStats>(`/matches/${id}/stats`),

  createMatch: (title: string, players?: string[], fileSizeBytes?: number) =>
    req<UploadInit>('/matches', {
      method: 'POST',
      body: JSON.stringify({
        title,
        player_names: players?.length ? players : undefined,
        file_size_bytes: fileSizeBytes,
      }),
    }),

  startProcessing: (id: string) =>
    req<Match>(`/matches/${id}/start`, { method: 'POST' }),

  deleteMatch: (id: string) =>
    req<void>(`/matches/${id}`, { method: 'DELETE' }),

  async uploadVideo(
    uploadUrl: string,
    file: File,
    onProgress?: (p: number) => void,
  ): Promise<void> {
    await new Promise<void>((resolve, reject) => {
      const xhr = new XMLHttpRequest()
      xhr.open('PUT', uploadUrl)
      xhr.setRequestHeader('Content-Type', 'video/mp4')
      if (onProgress) {
        xhr.upload.onprogress = (e) => {
          if (e.lengthComputable) onProgress(e.loaded / e.total)
        }
      }
      xhr.onload = () => (xhr.status < 300 ? resolve() : reject(new Error(`Upload ${xhr.status}`)))
      xhr.onerror = () => reject(new Error('Upload network error'))
      xhr.send(file)
    })
  },
}
