# Installazione su Raspberry Pi 5 — Guida completa

**Target:** Raspberry Pi 5 · 8 GB RAM · ARM64 (Cortex-A76)  
**OS consigliato:** Raspberry Pi OS Lite 64-bit (Bookworm, Debian 12)  
**Tempo stimato:** 45–60 minuti (la build Docker richiede ~20 min la prima volta)

---

## Indice

1. [Requisiti](#1-requisiti)
2. [Preparazione del sistema operativo](#2-preparazione-del-sistema-operativo)
3. [Installazione Docker](#3-installazione-docker)
4. [Clonare il repository](#4-clonare-il-repository)
5. [Configurazione variabili d'ambiente](#5-configurazione-variabili-dambiente)
6. [Storage: SSD locale vs MinIO](#6-storage-ssd-locale-vs-minio)
7. [Prima build e avvio](#7-prima-build-e-avvio)
8. [Migrazione database](#8-migrazione-database)
9. [Ottimizzazione YOLO → ONNX (raccomandato)](#9-ottimizzazione-yolo--onnx-raccomandato)
10. [Pesi TrackNet (opzionale)](#10-pesi-tracknet-opzionale)
11. [Verifica installazione](#11-verifica-installazione)
12. [Configurare l'app mobile](#12-configurare-lapp-mobile)
13. [Avvio automatico al boot](#13-avvio-automatico-al-boot)
14. [Comandi operativi](#14-comandi-operativi)
15. [Risoluzione problemi](#15-risoluzione-problemi)

---

## 1. Requisiti

### Hardware
| Componente | Minimo | Consigliato |
|---|---|---|
| Modello | Raspberry Pi 5 4 GB | **Raspberry Pi 5 8 GB** |
| MicroSD | 32 GB Class 10 | **64 GB+ SSD via PCIe/USB** |
| Alimentatore | USB-C 5V 3A | **USB-C 5V 5A ufficiale** |
| Rete | Wi-Fi | **Ethernet** (upload video ~500 MB) |
| Dissipatore | Qualsiasi | **Active Cooler ufficiale** (ML scalda la CPU) |

### Software richiesto sul Pi
- Raspberry Pi OS Lite **64-bit** (ARM64 obbligatorio per PyTorch)
- Docker Engine 26+
- Docker Compose v2 (`docker compose`, non `docker-compose`)
- Git

---

## 2. Preparazione del sistema operativo

### 2a. Flash della microSD

Usa **Raspberry Pi Imager** (scaricabile da raspberrypi.com/software):

1. **OS** → "Raspberry Pi OS (other)" → **"Raspberry Pi OS Lite (64-bit)"**
2. Clicca l'icona ⚙️ (impostazioni avanzate) e configura:
   - Nome host: `padelpi`
   - Abilita SSH con chiave pubblica (o password)
   - Wi-Fi (se non usi Ethernet)
   - Fuso orario e lingua
3. Scrivi sulla SD e inseriscila nel Pi

### 2b. Primo accesso e aggiornamento

```bash
# Connettiti via SSH
ssh pi@padelpi.local

# Aggiorna il sistema
sudo apt update && sudo apt full-upgrade -y
sudo reboot
```

### 2c. Configurazione memoria swap

Il worker ML può richiedere fino a 3.5 GB. Con swap si evitano OOM kill:

```bash
# Disabilita il vecchio swap (troppo piccolo)
sudo dphys-swapfile swapoff
sudo systemctl disable dphys-swapfile

# Crea un file di swap da 4 GB
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Rendilo permanente
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

# Riduci la swappiness (usa RAM prima della swap)
echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
```

### 2d. Abilitare cgroups v2 (richiesto da Docker)

```bash
sudo nano /boot/firmware/cmdline.txt
```

Aggiungi alla fine della riga (tutto su una riga):
```
cgroup_enable=memory cgroup_memory=1 cgroup_enable=cpuset
```

```bash
sudo reboot
```

### 2e. Aumentare i file descriptor aperti (opzionale ma consigliato)

```bash
echo '* soft nofile 65536' | sudo tee -a /etc/security/limits.conf
echo '* hard nofile 65536' | sudo tee -a /etc/security/limits.conf
```

---

## 3. Installazione Docker

```bash
# Script ufficiale Docker
curl -fsSL https://get.docker.com | sudo sh

# Aggiungi l'utente al gruppo docker (evita sudo ogni volta)
sudo usermod -aG docker $USER

# Riconnettiti per applicare il gruppo
exit
# poi: ssh pi@padelpi.local

# Verifica installazione
docker --version          # Docker Engine 26.x.x o superiore
docker compose version    # Docker Compose v2.x.x
```

---

## 4. Clonare il repository

```bash
cd ~
git clone https://github.com/tnt-labs/padelstatsML.git
cd padelstatsML
```

---

## 5. Configurazione variabili d'ambiente

> ⚠️ **Cambia le password** prima di esporre il Pi sulla rete locale.

```bash
cd ~/padelstatsML/backend
cp .env.example .env   # se non esiste, crealo da zero:
nano .env
```

Contenuto minimo per il Pi:

```env
# ── Database ─────────────────────────────────────────
DATABASE_URL=postgresql+asyncpg://padel:CAMBIA_QUESTA_PASSWORD@postgres:5432/padel
SYNC_DATABASE_URL=postgresql://padel:CAMBIA_QUESTA_PASSWORD@postgres:5432/padel

# ── Redis ─────────────────────────────────────────────
REDIS_URL=redis://redis:6379/0
CELERY_BROKER_URL=redis://redis:6379/1
CELERY_RESULT_BACKEND=redis://redis:6379/2

# ── S3 / MinIO ────────────────────────────────────────
S3_ENDPOINT=http://minio:9000
S3_ACCESS_KEY=padel
S3_SECRET_KEY=CAMBIA_QUESTA_PASSWORD
S3_BUCKET_VIDEOS=padel-videos
S3_REGION=us-east-1

# ── ML ────────────────────────────────────────────────
ML_DEVICE=cpu
TORCH_NUM_THREADS=4
PLAYER_STRIDE=3              # 3 su Pi (2 su GPU)
YOLO_WEIGHTS=weights/yolov8n.onnx   # dopo export ONNX (step 8)
TRACKNET_WEIGHTS=weights/tracknet_padel.pth

# ── TrackNet auto-download (lascia vuoto per usare MOG2 fallback) ──
# TRACKNET_WEIGHTS_URL=https://github.com/.../releases/download/v1.0/tracknet_padel.pth

# ── Video ─────────────────────────────────────────────
MAX_VIDEO_SIZE_MB=2048

# ── API ───────────────────────────────────────────────
CORS_ORIGINS=["*"]           # restringi in produzione
```

Aggiorna le stesse password anche nel file docker-compose per i servizi interni:

```bash
# Sostituisci "padelpadel" con la tua password MinIO
sed -i 's/padelpadel/NUOVA_PASSWORD_MINIO/g' docker-compose.yml docker-compose.pi.yml

# Sostituisci la password postgres
sed -i 's/POSTGRES_PASSWORD: padel/POSTGRES_PASSWORD: NUOVA_PASSWORD_PG/g' docker-compose.yml
```

---

## 6. Storage: SSD locale vs MinIO

Il sistema supporta due modalità di storage per i video:

| | **SSD locale** (consigliato per Pi) | **MinIO** (default) |
|---|---|---|
| RAM usata | ~4.2 GB totali | ~4.4 GB (+256 MB) |
| Semplicità | Nessun container extra | Richiede MinIO |
| Upload | Passa per l'API (nginx → FastAPI) | Diretto da client a MinIO |
| Backup | `cp` o `rsync` sulla directory | Snapshot volume Docker |

### Opzione A — SSD locale (raccomandato)

#### 6a. Montare l'SSD

Collega l'SSD al Pi (via USB 3.0 o PCIe M.2 HAT), poi:

```bash
# Trova il dispositivo
lsblk
# Esempio: /dev/sda o /dev/nvme0n1

# Formatta se è nuovo (attenzione: cancella tutto)
sudo mkfs.ext4 /dev/sda1

# Crea punto di mount
sudo mkdir -p /mnt/ssd

# Monta
sudo mount /dev/sda1 /mnt/ssd

# Rendi permanente (trova l'UUID del dispositivo)
sudo blkid /dev/sda1
# Copia l'UUID, poi:
echo 'UUID=xxxx-xxxx  /mnt/ssd  ext4  defaults,noatime  0  2' | sudo tee -a /etc/fstab

# Crea la directory dei video
sudo mkdir -p /mnt/ssd/padel-videos
sudo chown $USER:$USER /mnt/ssd/padel-videos
```

#### 6b. Configurare il .env per lo storage locale

Aggiungi/modifica queste righe nel `.env`:

```env
STORAGE_BACKEND=local
VIDEOS_DIR=/data/videos
# L'IP del Pi — deve essere raggiungibile dal client mobile/web
API_BASE_URL=http://192.168.1.42
```

#### 6c. Abilitare il volume nel docker-compose.pi.yml

Apri `docker-compose.pi.yml` e decommenta le righe indicate per `api` e `worker`:

```yaml
# Nei servizi api e worker, decommenta:
volumes:
  - /mnt/ssd/padel-videos:/data/videos

# Nelle variabili d'ambiente di api e worker, decommenta:
environment:
  STORAGE_BACKEND: local
  VIDEOS_DIR: /data/videos
  API_BASE_URL: "http://192.168.1.42"   # oppure http://padelpi.local
```

#### 6d. Avvio SENZA MinIO

Con lo storage locale MinIO non serve. Avvia solo i servizi necessari:

```bash
docker compose -f docker-compose.yml -f docker-compose.pi.yml \
  up -d postgres redis api worker frontend
```

---

### Opzione B — MinIO (comportamento predefinito)

Nessuna modifica necessaria. Lascia `STORAGE_BACKEND=s3` nel `.env` (o non impostarlo) e includi `minio` nel comando di avvio:

```bash
docker compose -f docker-compose.yml -f docker-compose.pi.yml \
  up -d postgres redis minio api worker frontend
```

---

## 7. Prima build e avvio

> La prima build scarica ~3 GB di immagini Docker e compila PyTorch wheels ARM64.
> Con connessione 50 Mbit/s richiede circa **15–25 minuti**.

```bash
cd ~/padelstatsML/backend

# Build con overlay Pi (memory limits + CPU-only torch)
docker compose -f docker-compose.yml -f docker-compose.pi.yml build

# Avvia i servizi infrastrutturali per primi
docker compose -f docker-compose.yml -f docker-compose.pi.yml \
  up -d postgres redis minio

# Attendi che siano healthy (~20 secondi)
docker compose ps
```

Output atteso:
```
NAME       STATUS
postgres   running (healthy)
redis      running (healthy)
minio      running (healthy)
```

---

## 8. Migrazione database

Esegui **una sola volta** per creare lo schema:

```bash
cd ~/padelstatsML/backend

docker compose -f docker-compose.yml -f docker-compose.pi.yml \
  run --rm migrate
```

Output atteso:
```
INFO  [alembic.runtime.migration] Running upgrade -> 0001, initial schema
```

---

## 9. Ottimizzazione YOLO → ONNX (raccomandato)

YOLOv8 in formato ONNX è **3–5× più veloce** su ARM64 rispetto al formato PyTorch nativo. L'export si fa una volta sola.

```bash
cd ~/padelstatsML/backend

# Entra nel container worker (che ha ultralytics installato)
docker compose -f docker-compose.yml -f docker-compose.pi.yml \
  run --rm --entrypoint bash worker

# Dentro il container:
python scripts/export_onnx.py --weights yolov8n.pt --out weights/yolov8n.onnx
exit
```

Il file `weights/yolov8n.onnx` viene creato nella cartella `backend/weights/` (montata come volume nel container).

Verifica:
```bash
ls -lh weights/yolov8n.onnx   # deve essere ~12 MB
```

Assicurati che nel `.env` sia impostato:
```env
YOLO_WEIGHTS=weights/yolov8n.onnx
```

---

## 10. Pesi TrackNet (opzionale)

Se hai a disposizione un file `.pth` di pesi TrackNetV2 addestrato su padel:

**Opzione A — copia manuale:**
```bash
# Dal tuo computer locale:
scp tracknet_padel.pth pi@padelpi.local:~/padelstatsML/backend/weights/
```

**Opzione B — download automatico al primo avvio del worker:**
```bash
# Aggiungi al .env:
TRACKNET_WEIGHTS_URL=https://INDIRIZZO_DEL_FILE/tracknet_padel.pth
```
Il worker scaricherà il file automaticamente la prima volta che si avvia. Senza pesi il sistema usa il fallback MOG2 (meno preciso ma funzionante).

---

## 10. Avvio completo del sistema

```bash
cd ~/padelstatsML/backend

docker compose -f docker-compose.yml -f docker-compose.pi.yml \
  up -d postgres redis minio api worker frontend
```

Attendi ~30 secondi poi verifica:

```bash
# Stato di tutti i container
docker compose -f docker-compose.yml -f docker-compose.pi.yml ps
```

Output atteso:
```
NAME       STATUS
postgres   running (healthy)
redis      running (healthy)
minio      running (healthy)
api        running (healthy)
worker     running
frontend   running
```

### Health check dettagliato

```bash
curl http://localhost/health | python3 -m json.tool
```

Output atteso:
```json
{
  "status": "ok",
  "db": "ok",
  "redis": "ok",
  "s3": "ok"
}
```

### Verifica interfaccia web

Apri nel browser del Pi (o da qualsiasi PC sulla stessa rete):
```
http://padelpi.local
```

Dovresti vedere la schermata principale di Padel Stats.

---

## 12. Configurare l'app mobile

L'app mobile deve sapere l'indirizzo IP del Pi sulla rete locale.

### Trova l'IP del Pi

```bash
hostname -I | awk '{print $1}'
# esempio: 192.168.1.42
```

### Configura l'URL nell'app

Nel file `mobile/.env` (crea se non esiste):
```env
EXPO_PUBLIC_API_URL=http://192.168.1.42:8000
```

Oppure, se il frontend nginx è attivo sulla porta 80:
```env
EXPO_PUBLIC_API_URL=http://192.168.1.42
```

Ricompila l'app Expo dopo la modifica:
```bash
cd ~/padelstatsML/mobile
npx expo start
```

> **Suggerimento:** Assegna un IP fisso al Pi dal router (DHCP reservation tramite MAC address) così l'URL non cambia mai.

---

## 13. Avvio automatico al boot

Crea un servizio systemd che avvia Docker Compose al riavvio del Pi:

```bash
sudo nano /etc/systemd/system/padelstats.service
```

```ini
[Unit]
Description=Padel Stats ML Stack
Requires=docker.service
After=docker.service network-online.target
Wants=network-online.target

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/home/pi/padelstatsML/backend
ExecStart=/usr/bin/docker compose -f docker-compose.yml -f docker-compose.pi.yml up -d postgres redis minio api worker frontend
ExecStop=/usr/bin/docker compose -f docker-compose.yml -f docker-compose.pi.yml down
StandardOutput=journal
User=pi

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable padelstats.service
sudo systemctl start padelstats.service

# Verifica
sudo systemctl status padelstats.service
```

---

## 14. Comandi operativi

### Avvio / Arresto

```bash
cd ~/padelstatsML/backend

# Avvia tutto
docker compose -f docker-compose.yml -f docker-compose.pi.yml \
  up -d postgres redis minio api worker frontend

# Arresta tutto (i dati sono preservati nei volumi)
docker compose -f docker-compose.yml -f docker-compose.pi.yml down

# Riavvia solo il worker (es. dopo cambio .env)
docker compose -f docker-compose.yml -f docker-compose.pi.yml restart worker
```

### Log in tempo reale

```bash
# Tutti i servizi
docker compose -f docker-compose.yml -f docker-compose.pi.yml logs -f

# Solo il worker ML
docker compose -f docker-compose.yml -f docker-compose.pi.yml logs -f worker

# Solo l'API
docker compose -f docker-compose.yml -f docker-compose.pi.yml logs -f api
```

### Monitoraggio risorse

```bash
# Uso RAM e CPU dei container
docker stats

# Temperatura CPU (importante durante l'analisi)
watch -n2 vcgencmd measure_temp

# Memoria disponibile
free -h
```

### Aggiornamento applicazione

```bash
cd ~/padelstatsML
git pull

cd backend
docker compose -f docker-compose.yml -f docker-compose.pi.yml build --no-cache
docker compose -f docker-compose.yml -f docker-compose.pi.yml up -d
docker compose -f docker-compose.yml -f docker-compose.pi.yml run --rm migrate
```

### Backup dati

```bash
# Backup database
docker exec padelstats-postgres-1 pg_dump -U padel padel > backup_$(date +%Y%m%d).sql

# Backup video su storage esterno
docker run --rm -v padelstats_miniodata:/data \
  -v /media/usb:/backup alpine \
  tar czf /backup/minio_$(date +%Y%m%d).tar.gz /data
```

---

## 15. Risoluzione problemi

### Container non si avviano

```bash
# Controlla i log di errore
docker compose -f docker-compose.yml -f docker-compose.pi.yml logs --tail=50

# Verifica spazio disco
df -h
```

### `OOMKilled` — worker ucciso per memoria

Il worker ha superato il limite di 3.5 GB. Soluzioni:

```bash
# 1. Aumenta PLAYER_STRIDE nel .env (meno frame = meno memoria)
PLAYER_STRIDE=4

# 2. Verifica che il file swap sia attivo
swapon --show

# 3. Se necessario, aumenta il limite nel docker-compose.pi.yml
#    worker.deploy.resources.limits.memory: 4000M
```

### Analisi lenta (>30 minuti per partita)

```bash
# Verifica che ONNX sia usato (deve apparire "onnxruntime" nei log)
docker compose logs worker | grep -i onnx

# Se non esportato, esegui l'export (step 8)

# Aumenta PLAYER_STRIDE per ridurre i frame processati
PLAYER_STRIDE=4   # o 5 per video molto lunghi

# Verifica temperatura — throttling termico rallenta la CPU
vcgencmd measure_temp   # sopra 80°C → servono più ventilazione
vcgencmd get_throttled  # 0x0 = ok, qualsiasi altro valore = problema
```

### Errore `court calibration failed`

La camera non è posizionata correttamente. Requisiti:
- Posizione fissa sopraelevata (almeno 3–4 m)
- Tutto il campo visibile (4 linee + 2 reti laterali)
- Nessun movimento durante la ripresa
- Buona illuminazione

### Errore `S3 connection refused`

```bash
# Verifica che MinIO sia up
docker compose -f docker-compose.yml -f docker-compose.pi.yml ps minio

# Se è "unhealthy", controlla i log
docker compose logs minio
```

### Frontend non risponde (porta 80)

```bash
# Verifica che nginx sia up
docker compose ps frontend

# Controlla la configurazione CORS se usi l'IP invece di padelpi.local
# Aggiungi l'IP al .env:
CORS_ORIGINS=["http://192.168.1.42","http://padelpi.local"]
```

### Ripartire da zero (reset completo)

```bash
cd ~/padelstatsML/backend

# ATTENZIONE: elimina tutti i dati (DB, video, code)
docker compose -f docker-compose.yml -f docker-compose.pi.yml down -v --remove-orphans
docker system prune -af --volumes

# Poi ripeti dalla build (step 6)
```

---

## Struttura porte esposte

| Porta | Servizio | Note |
|---|---|---|
| **80** | Frontend web (nginx) | Accesso principale |
| **8000** | API FastAPI | Accessibile anche direttamente |
| **5432** | PostgreSQL | Solo rete interna Docker |
| **6379** | Redis | Solo rete interna Docker |
| **9000** | MinIO API | Storage S3 |
| **9001** | MinIO Console | Dashboard web MinIO |

> Per sicurezza, blocca le porte 5432, 6379, 9000, 9001 con `ufw` se il Pi è accessibile da Internet:
> ```bash
> sudo ufw allow 22    # SSH
> sudo ufw allow 80    # Frontend
> sudo ufw allow 8000  # API (opzionale se usi solo il frontend)
> sudo ufw enable
> ```

---

## Riepilogo comandi essenziali

```bash
# Avvia tutto
docker compose -f docker-compose.yml -f docker-compose.pi.yml up -d postgres redis minio api worker frontend

# Controlla stato
docker compose -f docker-compose.yml -f docker-compose.pi.yml ps

# Health check
curl http://localhost/health

# Log worker in tempo reale
docker compose -f docker-compose.yml -f docker-compose.pi.yml logs -f worker

# Arresta tutto
docker compose -f docker-compose.yml -f docker-compose.pi.yml down
```



Checklist primo test
Fase 1 — Test pipeline ML standalone (senza Docker, 10 minuti)
Questo è il modo più rapido per verificare che il codice ML funzioni con un video reale, prima di montare tutta l'infrastruttura:

cd ~/padelstatsML/backend
pip install -r requirements.txt           # una volta

python scripts/test_pipeline.py video.mp4 --stride 2
Stampa a terminale ogni stage con percentuale e tempo. Se questo funziona, il grosso è fatto.

Fase 2 — Stack completo su Pi
Passo	Comando	Note
1. Build	docker compose -f docker-compose.yml -f docker-compose.pi.yml build	~20 min prima volta
2. Avvia infrastruttura	up -d postgres redis api worker frontend (+ minio se non usi SSD)	
3. Migra DB	docker compose run --rm migrate	Una volta sola
4. Export ONNX	dentro il container worker: python scripts/export_onnx.py	Una volta sola, 3-5× più veloce
5. Health check	curl http://localhost/health	Deve tornare {"status":"ok"}
Fase 3 — Test con video reale
Requisiti del video:

Camera fissa su cavalletto o appoggio stabile
Altezza ≥ 3 m (tribuna, balcone, vetro superiore)
Tutto il campo visibile — 4 linee di fondo + 2 reti laterali
Buona illuminazione, no controluce
Almeno 5 secondi di campo vuoto all'inizio prima della battuta
Formato MP4, < 2 GB
Per un primo test usa un video corto: 5–10 minuti di gioco (non tutta la partita — elaborazione ~15–20 min su Pi, meno su PC).

Fase 4 — App mobile
cd ~/padelstatsML/mobile
npx expo install          # installa async-storage e le altre dipendenze
# configura EXPO_PUBLIC_API_URL=http://<IP-del-PI> in .env
npx expo start
Pesi TrackNet (opzionale ma migliora il tracciamento palla)
Senza pesi il sistema usa il fallback MOG2 — funziona, ma rileva meno colpi veloci. Per il primo test va benissimo così; si aggiunge dopo se i risultati sul tracciamento palla sono scadenti.

Ordine consigliato: testa prima con test_pipeline.py su un video da 5 minuti. Se le statistiche stampate hanno senso (rally contati, giocatori tracciati, colpi classificati) → avvia lo stack Docker e testa l'interfaccia.
