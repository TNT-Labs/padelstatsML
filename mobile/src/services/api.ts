/**
 * API client. Centralizza chiamate al backend.
 * Pattern: ogni chiamata typed, error handling delegato al caller.
 */
import * as FileSystem from 'expo-file-system';

const API_BASE = process.env.EXPO_PUBLIC_API_URL ?? 'http://localhost:8000';

export type MatchStatus = 'uploading' | 'queued' | 'processing' | 'completed' | 'failed';

export interface Match {
  id: string;
  title: string;
  status: MatchStatus;
  progress: number;
  error_message: string | null;
  duration_seconds: number | null;
  player_names: string[] | null;
  created_at: string;
  updated_at: string;
}

export interface UploadInit {
  match_id: string;
  upload_url: string;
  s3_key: string;
}

export interface PlayerStats {
  distance_m: number;
  winners: number;
  errors: number;
  shots: { smash: number; volley: number; bandeja: number; other: number };
}

export interface MatchStats {
  match_id: string;
  per_player: Record<string, PlayerStats>;
  heatmaps: Record<string, [number, number, number][]>;
  rallies_count: number;
  total_shots: number;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: { 'Content-Type': 'application/json', ...init?.headers },
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`API ${res.status}: ${text}`);
  }
  return res.json();
}

export const api = {
  createMatch: (title: string, players?: string[], fileSizeBytes?: number) =>
    request<UploadInit>('/matches', {
      method: 'POST',
      body: JSON.stringify({ title, player_names: players, file_size_bytes: fileSizeBytes }),
    }),

  startProcessing: (matchId: string) =>
    request<Match>(`/matches/${matchId}/start`, { method: 'POST' }),

  getMatch: (matchId: string) => request<Match>(`/matches/${matchId}`),

  getStats: (matchId: string) => request<MatchStats>(`/matches/${matchId}/stats`),

  listMatches: () => request<Match[]>('/matches'),

  /**
   * Upload diretto a S3 via presigned URL. Usa expo-file-system per
   * gestire upload chunked di file grandi senza caricare tutto in memoria.
   */
  async uploadVideo(uploadUrl: string, localUri: string, onProgress?: (p: number) => void) {
    const task = FileSystem.createUploadTask(
      uploadUrl,
      localUri,
      {
        httpMethod: 'PUT',
        uploadType: FileSystem.FileSystemUploadType.BINARY_CONTENT,
        headers: { 'Content-Type': 'video/mp4' },
      },
      (progress) => {
        if (onProgress && progress.totalBytesExpectedToSend > 0) {
          onProgress(progress.totalBytesSent / progress.totalBytesExpectedToSend);
        }
      },
    );
    const result = await task.uploadAsync();
    if (!result || result.status >= 300) {
      throw new Error(`Upload failed: ${result?.status}`);
    }
  },
};
