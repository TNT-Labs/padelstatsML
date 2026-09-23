# Installazione su Raspberry Pi 5

**Target:** Raspberry Pi 5 · 8 GB RAM · ARM64
**OS:** Raspberry Pi OS Lite 64-bit (Bookworm)
**Tempo:** ~40 minuti, di cui ~15 di build Docker

Il sistema gira in **due container** (`api` e `worker`) che condividono una
sola immagine, un file SQLite e una cartella sull'SSD. Non servono database
server, broker, object store o reverse proxy.

---

## Indice

1. [Hardware](#1-hardware)
2. [Preparazione del sistema](#2-preparazione-del-sistema)
3. [Docker](#3-docker)
4. [SSD e cartella dati](#4-ssd-e-cartella-dati)
5. [Configurazione](#5-configurazione)
6. [Esportare il modello ONNX](#6-esportare-il-modello-onnx)
7. [Build e avvio (Docker)](#7-build-e-avvio-docker)
7b. [Installazione senza Docker](#7b-installazione-senza-docker)
8. [Verifica](#8-verifica)
9. [Prima analisi](#9-prima-analisi)
10. [Avvio automatico al boot](#10-avvio-automatico-al-boot)
11. [Comandi operativi](#11-comandi-operativi)
12. [Verificare che il tracciamento sia corretto](#12-verificare-che-il-tracciamento-sia-corretto)
13. [Taratura velocità/accuratezza](#13-taratura-velocitàaccuratezza)
14. [Backup](#14-backup)
15. [Risoluzione problemi](#15-risoluzione-problemi)

---

## 1. Hardware

| Componente | Minimo | Consigliato |
|---|---|---|
| Scheda | Pi 5 4 GB | **Pi 5 8 GB** |
| Storage | microSD 32 GB | **SSD NVMe o USB 3.0** |
| Alimentazione | USB-C 5V 3A | **USB-C 5V 5A ufficiale** |
| Rete | Wi-Fi | **Ethernet** (i video pesano centinaia di MB) |
| Raffreddamento | qualsiasi | **Active Cooler ufficiale** |

Il raffreddamento non è opzionale: l'analisi tiene i quattro core al 100% per
un'ora e senza dissipazione attiva il Pi riduce la frequenza, allungando i
tempi del 30-40%.

Un SSD è fortemente consigliato: il database usa il journal WAL, che su
microSD provoca usura rapida, e la decodifica video legge in sequenza
centinaia di MB.

---

## 2. Preparazione del sistema

Con **Raspberry Pi Imager**, scrivi *Raspberry Pi OS Lite (64-bit)* e
configura hostname (`padelpi`), SSH e rete nelle impostazioni avanzate.

```bash
ssh pi@padelpi.local
sudo apt update && sudo apt full-upgrade -y
sudo reboot
```

### Swap

Il worker può arrivare a ~3 GB di picco. Con 8 GB non serve, ma un po' di
swap evita l'OOM killer durante la build Docker:

```bash
sudo dphys-swapfile swapoff
sudo sed -i 's/^CONF_SWAPSIZE=.*/CONF_SWAPSIZE=2048/' /etc/dphys-swapfile
sudo dphys-swapfile setup && sudo dphys-swapfile swapon
```

### Boot da NVMe (se usi un HAT PCIe)

```bash
sudo raspi-config     # Advanced Options → Boot Order → NVMe/USB
```

---

## 3. Docker

```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER
newgrp docker
docker --version && docker compose version
```

Serve Docker Compose v2 (comando `docker compose`, non `docker-compose`).

---

## 4. SSD e cartella dati

Monta l'SSD e crea la cartella dei dati:

```bash
lsblk                               # individua il device, es. /dev/nvme0n1p1
sudo mkdir -p /mnt/ssd
sudo mount /dev/nvme0n1p1 /mnt/ssd

# Montaggio permanente
sudo blkid /dev/nvme0n1p1           # copia lo UUID
echo 'UUID=<uuid>  /mnt/ssd  ext4  defaults,noatime  0  2' | sudo tee -a /etc/fstab

sudo mkdir -p /mnt/ssd/padelstats
sudo chown -R $USER:$USER /mnt/ssd/padelstats
```

Dentro `/mnt/ssd/padelstats` finiranno database, video, keyframe e anteprime.

---

## 5. Configurazione

```bash
git clone <repo> ~/padelstats
cd ~/padelstats
cp .env.example .env
nano .env
```

I due valori da impostare:

```ini
DATA_VOLUME=/mnt/ssd/padelstats
API_BASE_URL=http://padelpi.local:8000
```

`API_BASE_URL` deve essere raggiungibile dal telefono o dal PC: se
`padelpi.local` non si risolve sulla tua rete, usa l'IP
(`http://192.168.1.42:8000`).

---

## 6. Esportare il modello ONNX

Il Pi esegue l'inferenza con **onnxruntime** e non installa PyTorch. Il
modello va esportato una volta sola; il file risultante è portabile.

### Sul Pi (consigliato)

```bash
cd ~/padelstats

# Il venv va sull'SSD, non in /tmp: lì lo spazio è poco.
python3 -m venv .export-venv && source .export-venv/bin/activate
pip install --upgrade pip
pip install -r backend/requirements.export.txt      # ~10 minuti, ~2 GB
python backend/scripts/export_yolo_onnx.py --imgsz 480 --out weights/yolov8n.onnx
deactivate && rm -rf .export-venv                    # lo spazio torna libero
```

Il virtualenv non è opzionale: Bookworm segue la PEP 668 e un `pip install`
di sistema fallisce con `externally-managed-environment`.

### Da un altro computer

Il file ONNX è portabile, quindi puoi produrlo altrove e copiarlo con `scp`
in `~/padelstats/weights/`. Attenzione a due cose:

- `torch==2.4.1` ha wheel solo per **Python 3.8-3.12**. Su una macchina con
  un Python più recente (3.13, 3.14) `pip install` fallisce con
  "No matching distribution found", e serve affiancare un Python più vecchio.
- Su Windows i comandi qui sopra vanno tradotti in PowerShell: `;` al posto
  di `&&`, e `.\.export-venv\Scripts\Activate.ps1` per attivare il venv.

Per questo la strada sul Pi resta la più semplice.

> **Se l'export finisce con `PermissionError` scrivendo in `weights/`:** la
> cartella appartiene a root perché l'ha creata Docker montandola prima che
> esistesse. Il modello è comunque stato prodotto nella cartella corrente:
> ```bash
> sudo chown -R $(id -u):$(id -g) weights
> mv yolov8n.onnx weights/
> ```

Lo script verifica l'export ricaricandolo con onnxruntime, esattamente come
farà il worker.

> `--imgsz` deve coincidere con `DETECTOR_IMGSZ` nel `.env`. 480 è il
> compromesso migliore sul Cortex-A76: circa 1,8× più veloce di 640 con una
> perdita di recall contenuta sui giocatori di fondo campo.

---

## 7. Build e avvio (Docker)

```bash
cd ~/padelstats
docker compose up -d --build
```

La prima build richiede 10-15 minuti (npm install + pip install). Tutte le
dipendenze Python hanno wheel precompilate per aarch64: non viene compilato
nulla.

---

## 7b. Installazione senza Docker

Alternativa al passo 7, non aggiuntiva: scegli l'una o l'altra. Senza Docker
si risparmiano circa 400 MB di immagini e l'avvio è più rapido, ma i due
servizi vanno installati a mano.

### Prerequisiti

```bash
sudo apt install -y python3-venv python3-dev libglib2.0-0 nodejs npm
```

`libglib2.0-0` serve a opencv-python-headless; tutto il resto delle
dipendenze Python ha wheel precompilate per aarch64, quindi non viene
compilato nulla.

### Installazione

```bash
cd ~/padelstats
make install
```

Il target crea un virtualenv in `.venv`, installa le dipendenze pinnate e
compila la UI in `backend/web`.

> Il virtualenv non è opzionale: Raspberry Pi OS Bookworm segue la PEP 668 e
> un `pip install` nell'interprete di sistema fallisce con
> `externally-managed-environment`.

> La UI **deve** finire in `backend/web`: è lì che FastAPI la cerca. Se salti
> `make web`, l'API risponde regolarmente ma il browser riceve una pagina
> vuota, e nei log compare `UI non presente in … — modalità solo API`.

### Prova in foreground

Due terminali, per vedere subito i log:

```bash
make run      # terminale 1 — API su :8000
make worker   # terminale 2 — worker di analisi
```

Verifica da un altro terminale:

```bash
make check
```

### Servizi systemd (avvio automatico)

```bash
sudo cp deploy/systemd/padelstats-*.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now padelstats-api padelstats-worker
```

Le unit presuppongono il repository in `/home/pi/padelstats`, l'utente `pi` e
i dati in `/mnt/ssd/padelstats`. Se il tuo assetto è diverso, adatta
`User`, `WorkingDirectory`, `ExecStart` e `ReadWritePaths` prima di copiarle
(`ReadWritePaths` deve combaciare con `DATA_VOLUME`, altrimenti i servizi
partono ma non riescono a scrivere).

Il worker gira con `Nice=5` e l'API con `Nice=-5`: durante un'analisi i
quattro core sono saturi, e senza questa differenza la pagina di stato smette
di aggiornarsi proprio quando serve guardarla.

### Comandi operativi

```bash
sudo systemctl status padelstats-api padelstats-worker
sudo journalctl -u padelstats-worker -f     # log dell'analisi
sudo systemctl restart padelstats-worker

# Dopo un git pull
cd ~/padelstats && git pull && make install
sudo systemctl restart padelstats-api padelstats-worker
```

Il worker termina il job in corso prima di uscire, quindi un `restart`
durante un'analisi può richiedere qualche minuto; se viene interrotto prima,
il job torna in coda da solo al riavvio successivo.

---

## 8. Verifica

```bash
# Docker
docker compose ps
# Senza Docker
sudo systemctl status padelstats-api padelstats-worker

# In entrambi i casi
curl -s http://localhost:8000/api/health | python3 -m json.tool
```

Atteso:

```json
{
  "status": "ok",
  "checks": {
    "database": "ok",
    "storage": "ok · 421.3 GB liberi",
    "detector": "ok · yolov8n.onnx"
  }
}
```

Se `detector` riporta `mancante`, il file ONNX non è in `./weights/` oppure
`DETECTOR_MODEL` non corrisponde. Lo stato `503` è voluto: senza modello il
sistema non può analizzare nulla e lo dichiara invece di ripiegare in
silenzio su un risultato inventato.

Log del worker:

```bash
docker compose logs -f worker   # Docker
sudo journalctl -u padelstats-worker -f                  # senza Docker
# Worker avviato · 4 thread di inferenza
```

---

## 9. Prima analisi

Apri `http://padelpi.local:8000` dal browser.

1. **Nuova partita** → titolo e video → *Carica video*.
2. **Calibrazione**: trascina le quattro maniglie sugli angoli del campo,
   partendo dal fondo a sinistra e proseguendo in senso orario. Le linee
   bianche mostrano rete e linee di servizio calcolate: se coincidono con
   quelle reali, la calibrazione è corretta.
3. Dai un nome alla posizione camera (es. *Campo 2 — tripode angolo nord*) per
   riusarla con un clic nelle partite successive.
4. **Conferma calibrazione** → l'analisi parte. Puoi chiudere la pagina: il
   lavoro prosegue sul Pi.

Consiglio: la prima volta, prova con una clip di 2-3 minuti. Il ciclo
completo dura pochi minuti e ti permette di verificare calibrazione e
tracciamento prima di impegnare un'ora su una partita intera.

---

## 10. Avvio automatico al boot

Docker Compose riavvia i container da solo (`restart: unless-stopped`).
Perché avvengano anche dopo un reboot:

```bash
sudo systemctl enable docker
```

Per un controllo più esplicito:

```bash
sudo tee /etc/systemd/system/padelstats.service > /dev/null <<'UNIT'
[Unit]
Description=Padel Stats
Requires=docker.service
After=docker.service network-online.target

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/home/pi/padelstats
ExecStart=/usr/bin/docker compose up -d
ExecStop=/usr/bin/docker compose down

[Install]
WantedBy=multi-user.target
UNIT

sudo systemctl daemon-reload && sudo systemctl enable padelstats
```

---

## 11. Comandi operativi

```bash
cd ~/padelstats
C="docker compose"

$C ps                      # stato
$C logs -f worker          # log dell'analisi
$C restart worker          # riavvio del solo worker
$C down && $C up -d        # riavvio completo
$C up -d --build           # aggiornamento dopo un git pull

docker stats --no-stream   # RAM e CPU
vcgencmd measure_temp      # temperatura (sotto 80°C sotto carico)
df -h /mnt/ssd             # spazio disco
```

Un worker riavviato durante un'analisi rimette il job in coda da solo: il job
viene recuperato dall'heartbeat scaduto e riprovato al successivo avvio.

---

## 12. Verificare che il tracciamento sia corretto

**Fallo alla prima partita, prima di fidarti di qualunque numero.** Le
statistiche sono sempre plausibili: non si distingue un'analisi corretta da
una in cui due giocatori sono stati scambiati guardando solo i totali.

Ogni analisi salva le osservazioni grezze in
`/mnt/ssd/padelstats/artifacts/<match-id>/` (~30 MB per ora di video). Da
quelle si genera un video annotato senza rieseguire il detector:

```bash
cd ~/padelstats
C="docker compose exec worker"

# Primi 3 minuti annotati, ridotti a metà risoluzione
$C python scripts/overlay.py latest --to 180 --scale 0.5 --out /data/check.mp4

# Oppure una manciata di immagini, più rapide da sfogliare
$C python scripts/overlay.py latest --stills /data/frames --count 12
```

Il file finisce in `/mnt/ssd/padelstats/` sull'host. Copialo e guardalo.

**Cosa controllare, in quest'ordine:**

1. **Il campo disegnato sta sul campo vero?** La linea gialla deve cadere
   sulla rete reale e le linee tratteggiate sulle linee di servizio. Se no,
   la calibrazione è sbagliata e ogni metrica in metri lo è con lei:
   riapri la partita e ricalibra.
2. **Ogni giocatore mantiene lo stesso colore?** Se un colore salta fra due
   persone c'è un errore di identità. La mini-mappa in alto a destra lo rende
   evidente: un giocatore che attraversa la rete di colpo è uno scambio di
   identità, non un movimento.
3. **I segmenti verdi in basso coincidono con i punti reali?** Se gli scambi
   sono troppi o troppo pochi, tara la soglia con `retune.py` (sotto).

Non serve conoscere l'ID della partita: `latest` indica l'ultima analizzata.
Va bene anche un prefisso, per esempio `4f21e100`. L'ID completo compare nel
pannello *Affidabilità dei dati* della pagina statistiche, oppure con:

```bash
curl -s http://localhost:8000/api/matches | python3 -c "
import json,sys
for m in json.load(sys.stdin):
    print(f\"{m['id']}  {m['status']:<18} {m['title']}\")
"
```

---

## 13. Taratura velocità/accuratezza

`SAMPLE_HZ` nel `.env` è la leva principale.

| SAMPLE_HZ | Tempo per 60 min di video | Effetto |
|---|---|---|
| 3.0 | ~35 min | distanza sottostimata ~15% (segnalata nei warning) |
| **5.0** | **~60 min** | **default, buon compromesso** |
| 8.0 | ~100 min | guadagno marginale |

Altre leve:

- `DETECTOR_IMGSZ=416` — ~25% più veloce, perde qualche giocatore sul fondo.
- `DETECTOR_CONF=0.25` — più rilevazioni, più falsi positivi (filtrati poi
  dal confine del campo).
- `MAX_ANALYSIS_MINUTES` — tronca i video lunghi; la troncatura viene
  dichiarata nel risultato.

Dopo ogni modifica: `docker compose up -d`.

### Tarare le soglie senza rianalizzare

L'inferenza è l'unico stadio costoso. Tutto ciò che viene dopo — identità,
scambi, metriche — gira sulle osservazioni salvate in millisecondi, quindi
una soglia si valuta subito invece di costare un'ora:

```bash
C="docker compose exec worker"

# Risultato con le impostazioni attuali
$C python scripts/retune.py latest

# Confronta più valori della soglia di scambio
$C python scripts/retune.py latest --sweep rally-speed 0.8 1.0 1.2 1.4 1.6
```

Scegli il valore i cui scambi corrispondono ai punti reali (verificalo
nell'overlay), poi mettilo nel `.env` e riavvia. Parametri disponibili:
`rally-speed`, `rally-min`, `rally-gap`, `max-speed`.

Lo stesso vale per il tracker: `retrack.py` lo riesegue sulle osservazioni
salvate, senza detector, e confronta il risultato con l'analisi originale.
Dopo un aggiornamento che tocca il tracking, dice se la partita migliora
prima di spendere un'ora a rianalizzarla:

```bash
$C python scripts/retrack.py latest
```

Se i giocatori risultano tracciati per una piccola parte della partita,
`diagnose_identity.py` mostra cosa tiene fuori le rilevazioni scartate:

```bash
$C python scripts/diagnose_identity.py latest
```

Se lo spazio disco diventa critico puoi disattivare gli artefatti con
`KEEP_ARTIFACTS=false`, rinunciando però a overlay e taratura.

---

## 14. Backup

Tutto lo stato sta in una cartella:

```bash
# A container fermi, per un backup coerente del database
docker compose stop
tar czf padel-backup-$(date +%F).tar.gz -C /mnt/ssd padelstats
docker compose start
```

Solo il database (i video pesano molto di più):

```bash
sqlite3 /mnt/ssd/padelstats/padel.db ".backup '/tmp/padel-$(date +%F).db'"
```

---

## 15. Risoluzione problemi

**`/api/health` risponde 503 con `detector: mancante`**
Il file ONNX non è in `./weights/`. Rifai il [passo 6](#6-esportare-il-modello-onnx)
e verifica che `DETECTOR_MODEL` nel `.env` corrisponda al percorso montato.

**"Impossibile aprire il video" subito dopo il caricamento**
Il contenitore non è leggibile da OpenCV (spesso HEVC di iPhone). Converti:
`ffmpeg -i input.mov -c:v libx264 -preset fast -crf 23 output.mp4`.

**"Nessun giocatore rilevato"**
Nel 90% dei casi la calibrazione non corrisponde al video: riapri la partita,
ricalibra e controlla che la rete disegnata coincida con quella reale.
Altrimenti il campo è troppo piccolo nel fotogramma o la ripresa è troppo
bassa e i giocatori si coprono a vicenda.

**I numeri sembrano sbagliati ma non capisci perché**
Genera l'overlay ([passo 12](#12-verificare-che-il-tracciamento-sia-corretto)).
Distanze gonfiate indicano quasi sempre scambi di identità; distanze troppo
basse indicano un tracciamento con molti buchi (guarda *copertura del
tracciamento* nel pannello di affidabilità).

**"Artefatti non trovati" eseguendo overlay.py o retune.py**
La partita è stata analizzata con `KEEP_ARTIFACTS=false`, oppure prima che la
funzione esistesse. Rilancia l'analisi dalla pagina della partita.

**"Rilevati solo N giocatori invece di 4"**
Un angolo del campo è fuori inquadratura, oppure un giocatore resta occluso
per gran parte della partita. Il pannello *Affidabilità dei dati* indica
quali giocatori sono stati seguiti poco.

**L'analisi è molto più lenta del previsto**
Controlla la temperatura (`vcgencmd measure_temp`): sopra gli 80°C il Pi
riduce la frequenza. Verifica anche di non aver messo i dati su microSD.

**Il worker viene terminato (OOM)**
Riduci `DETECTOR_IMGSZ` a 416, verifica lo swap del [passo 2](#swap) e che
nessun altro servizio pesante giri sul Pi.

**La pagina web non si apre dal telefono**
`API_BASE_URL` deve contenere un hostname o IP raggiungibile dal telefono,
non `localhost`. Verifica con `curl http://<ip-del-pi>:8000/api/health` da un
altro dispositivo della rete.

**Il database è bloccato**
Solo il worker scrive a lungo; l'API usa un `busy_timeout` di 10 secondi. Se
l'errore persiste, quasi sempre ci sono due worker attivi:
`docker compose ps` deve mostrarne uno solo.
