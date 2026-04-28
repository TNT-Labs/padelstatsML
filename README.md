# Padel Stats — Cloud Async MVP

Sistema di analisi automatica di partite di padel tramite computer vision.
L'app mobile registra/carica un video, il backend lo processa in modo asincrono
e restituisce statistiche aggregate (heatmap, vincenti/errori, tipi di colpo).

## Architettura

```
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│  React Native   │       │    FastAPI      │       │  Celery Worker  │
│      App        │──────▶│     API         │──────▶│   ML Pipeline   │
└─────────────────┘  HTTP └─────────────────┘ Redis └─────────────────┘
        │                          │                          │
        │                          ▼                          ▼
        │                 ┌─────────────────┐       ┌─────────────────┐
        └────────────────▶│   S3 / MinIO    │       │   PostgreSQL    │
            (upload)      │  (video files)  │       │  (stats, jobs)  │
                          └─────────────────┘       └─────────────────┘
```

## Stack

### Backend
- **FastAPI** — API REST, validazione Pydantic
- **Celery + Redis** — job queue async per analisi video
- **PostgreSQL** — metadata, partite, statistiche
- **MinIO/S3** — storage video raw + analizzati
- **SQLAlchemy 2.0** — ORM async

### ML Pipeline
- **YOLOv8** (Ultralytics) — player detection + pose
- **ByteTrack** — multi-object tracking robusto (4 giocatori)
- **TrackNetV2** — ball tracking specifico padel
- **OpenCV** — court detection via Hough lines + omografia
- **PyTorch** — runtime modelli

### Mobile
- **React Native + Expo** — cross-platform
- **expo-camera** — registrazione video
- **expo-file-system** — gestione file locali
- **TanStack Query** — fetching/caching API

## Pipeline di analisi (worker)

1. `download_video()` — pull da S3 in tmp locale
2. `detect_court()` — Hough lines → 4 corner → omografia (coordinate reali in metri)
3. `track_players()` — YOLOv8 per frame + ByteTrack per ID consistenti
4. `track_ball()` — TrackNetV2 su finestra di 3 frame consecutivi
5. `detect_events()` — rule-based su trajectory: rimbalzi (cambio direzione Y),
   colpi (proximity ball↔player + cambio velocità), vetri (proximity ball↔muro)
6. `classify_shots()` — pose keypoints + ball trajectory → smash/volée/bandeja
7. `aggregate_stats()` — heatmap, distanza, vincenti/errori per player
8. `persist()` — scrivi su DB + carica overlay video opzionale

## Setup rapido

```bash
# Backend
cd backend
docker-compose up -d  # postgres + redis + minio
pip install -r requirements.txt
alembic upgrade head
uvicorn app.main:app --reload

# Worker
celery -A app.workers.celery_app worker --loglevel=info

# Mobile
cd mobile
npm install
npx expo start
```

## Roadmap

- [x] Fase 1 — Cloud async MVP (questo repo)
- [ ] Fase 2 — On-device post-match analysis
- [ ] Fase 3 — Real-time on-device

## Note critiche di sviluppo

**Inquadratura camera:** il modello assume camera fissa, elevata, dietro al campo,
con tutto il campo visibile. L'app DEVE guidare l'utente nel posizionamento
(setup wizard con overlay del campo) altrimenti la court detection fallisce e
tutto il resto crolla.

**Qualità video minima:** 1080p @ 30fps. Sotto questa soglia la palla diventa
indistinguibile dal rumore.

**Costi GPU:** un match di 60 minuti richiede ~10-15 min su RTX 3060.
Per produzione: serverless GPU (RunPod, Modal, Replicate) a consumo.
