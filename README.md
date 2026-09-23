# Padel Stats — analisi partite su Raspberry Pi 5

Sistema self-hosted che analizza il video di una partita di padel e produce
statistiche di **posizione e movimento** dei quattro giocatori: distanza
percorsa, occupazione del campo, zone, scambi.

Gira interamente su un Raspberry Pi 5. Nessun cloud, nessuna GPU, nessun
acceleratore, nessun dato che esce di casa.

```
┌──────────────┐        ┌───────────────────────────────┐
│   Browser    │◀──────▶│  Raspberry Pi 5 (2 container) │
│ (PC o phone) │  HTTP  │                               │
└──────────────┘        │  api      FastAPI + web UI    │
                        │  worker   pipeline di analisi │
                        │                               │
                        │  SQLite  ·  SSD (video/foto)  │
                        └───────────────────────────────┘
```

---

## Cosa misura, e cosa no

Questa è la parte più importante del progetto. Ogni numero prodotto è
espresso in metri e deriva da un'omografia **confermata da una persona**.
Nulla viene stimato di nascosto.

### Misurato

| Metrica | Come |
|---|---|
| Distanza percorsa (totale e negli scambi) | somma dei passi in coordinate campo |
| Velocità media e di punta | derivata dalle posizioni, 95° percentile per la punta |
| Occupazione del campo (heatmap 0,5 m) | istogramma delle posizioni dei piedi |
| Quota di tempo a rete / metà campo / fondo | distanza dalla linea di rete |
| Area di campo coperta | celle occupate × 0,25 m² |
| Numero, durata e distribuzione degli scambi | segmentazione dal movimento dei giocatori |
| Tempo di gioco effettivo | rapporto scambi / durata |
| Copertura del tracciamento per giocatore | quanto della partita ciascuno è stato seguito |

### Non misurato — e perché

Tipo di colpo (smash, bandeja, volée), vincenti ed errori, velocità della
palla, punteggio.

Tutte e quattro richiedono il **tracciamento della palla**. Un Pi 5 senza
acceleratore non lo può fare a un frame rate utile: una pallina da padel a
90 km/h percorre 80 cm per frame a 30 fps e occupa pochi pixel, quindi serve
inferenza su ogni frame — ore di elaborazione per ogni partita, con modelli
che pesano più del detector di persone.

La versione precedente li riportava comunque, prodotti da euristiche
(picchi di accelerazione su una traiettoria già filtrata, un albero
decisionale a soglie fisse, "palla vicino al bordo = errore"). I numeri
avevano l'aspetto di misure e non lo erano. Sono stati rimossi: **meglio
sette metriche vere che venti plausibili.**

Se in futuro si aggiunge un acceleratore (per esempio un Hailo-8L su HAT+),
il tracciamento della palla torna alla portata e queste metriche possono
rientrare — con un modello addestrato e delle metriche di validazione, non
con delle soglie.

---

## Come funziona

### 1. Calibrazione del campo — obbligatoria

L'utente trascina quattro maniglie sugli angoli del campo in un fotogramma
del video. Mentre trascina, l'interfaccia proietta il modello del campo
(rete, linee di servizio, perimetro) sull'immagine: se la rete calcolata
coincide con quella reale, la calibrazione è corretta. Serve mezzo minuto.

La calibrazione si salva come **preset camera** e si riusa con un clic per
ogni partita successiva ripresa dalla stessa posizione.

Il rilevamento automatico esiste solo come *proposta* da correggere, mai
come calibrazione. Su un campo da padel — vetri, rete metallica, poche linee
dipinte, superfici riflettenti — non è affidabile, e senza conferma umana
l'analisi non parte.

### 2. Analisi — una sola decodifica del video

```
decode (una volta, con frame-skip)
   └─▶ detector persone (ONNX, ritagliato sul campo)
        └─▶ tracking in metri sul piano del campo
             └─▶ ricostruzione identità: 4 giocatori, 2 coppie
                  └─▶ segmentazione scambi dal movimento
                       └─▶ metriche + qualità del dato
```

Punti chiave:

- **Una sola passata.** Il video viene decodificato una volta sola. Sul Pi il
  decode H.264 costa più della rete neurale alle frequenze di campionamento
  sostenibili; i frame non campionati usano `grab()` invece di `read()`.
- **Associazione in metri, non in pixel.** A 5 Hz due box consecutive di un
  giocatore in corsa quasi non si sovrappongono, quindi l'IoU fallisce
  proprio quando conta. Il gate è `velocità_max × Δt`, un vincolo fisico
  uniforme su tutto il campo.
- **Tracklet corte e ricucite dopo.** Le tracce muoiono e rinascono
  liberamente; l'identità viene ricostruita a valle con l'istogramma colore
  della maglia più la raggiungibilità fisica. Una persona occlusa per tre
  secondi resta la stessa persona.
- **Scambi dal movimento dei giocatori,** non dalla palla: durante il punto
  tutti e quattro si muovono, fra un punto e l'altro no.

### 3. Verifica e taratura

I numeri hanno sempre un aspetto plausibile: una distanza di 2 480 m e una
heatmap sbilanciata verso la rete sono indistinguibili da quelle giuste,
anche se due giocatori sono stati scambiati per metà partita. L'unico modo
per saperlo è **guardare i fotogrammi**.

Ogni analisi salva le osservazioni grezze (~30 MB per ora di video). Da quelle:

```bash
# Guarda cosa ha visto la pipeline: campo proiettato, ID giocatori,
# velocità, mini-mappa dall'alto, scambi sulla timeline
python scripts/overlay.py latest --out check.mp4
python scripts/overlay.py latest --stills /tmp/frames --from 120 --to 180

# Ri-tara una soglia in un secondo, senza rifare l'inferenza
python scripts/retune.py latest --sweep rally-speed 0.8 1.0 1.2 1.4 1.6

# Perché le tracce si spezzano: detector che perde il giocatore,
# o associazione che rifiuta di ricongiungere?
python scripts/diagnose_tracking.py latest

# Riesegue il tracker (senza detector, in secondi) e confronta con
# l'analisi salvata: misura una modifica al tracker prima di rianalizzare
python scripts/retrack.py latest

# Perché delle rilevazioni restano fuori dai 4 giocatori: per ogni cluster
# escluso, cosa gli impedisce di unirsi a ciascun giocatore
python scripts/diagnose_identity.py latest
```

Ogni script accetta `latest`, un prefisso dell'id (`4f21e100`) o l'id intero.
L'id compare anche nel pannello di affidabilità della pagina statistiche.

L'inferenza è l'unico stadio costoso: tutto ciò che viene dopo gira sulle
osservazioni salvate in millisecondi. È la differenza fra tarare una soglia
in un secondo e pagare un'ora di rianalisi per ogni tentativo — cioè fra
misurare un miglioramento e sperarci.

Cosa guardare nell'overlay, in ordine:
1. Il campo disegnato sta sul campo vero, e la linea gialla sulla rete vera?
   Se no, ricalibra: ogni numero in metri è sbagliato.
2. Ogni giocatore mantiene lo stesso colore per tutta la partita? Un colore
   che salta fra due persone è un errore di identità — sulla mini-mappa è
   evidente.
3. I segmenti verdi sulla timeline coincidono con i punti reali?

### 4. Qualità del dato, sempre visibile

Ogni risultato porta con sé `data_quality`: origine della calibrazione,
scarto della rete, frequenza di campionamento, quanti giocatori sono stati
trovati, per quanta parte della partita ciascuno è stato seguito, e la lista
esplicita di cosa non è stato misurato. L'interfaccia mostra il pannello in
cima alle statistiche.

---

## Stack

**Backend** — FastAPI, SQLAlchemy 2, SQLite (WAL), onnxruntime, OpenCV, NumPy,
SciPy.
Niente PyTorch, niente ultralytics, niente Celery, niente Redis, niente
Postgres, niente MinIO, niente Alembic: l'immagine passa da ~3 GB a ~600 MB e
l'avvio del worker da ~25 s a ~3 s.

**Frontend** — React 18 + Vite, SVG puro per grafici e campo. Servito dalla
stessa FastAPI: niente nginx.

**Coda** — una tabella `jobs` su SQLite, drenata da un singolo processo
worker. Il deployment esegue un'analisi alla volta su una macchina: un broker
aggiungerebbe due servizi e ~200 MB di RAM per una concorrenza mai usata.

---

## Avvio rapido

Due modi, entrambi supportati. In entrambi i casi servono un `.env` e il
modello ONNX del detector.

```bash
git clone <repo> && cd padelstatsML
cp .env.example .env        # imposta DATA_DIR/DATA_VOLUME e API_BASE_URL

# Esporta il detector una volta sola, direttamente sul Pi
python3 -m venv .export-venv && source .export-venv/bin/activate
pip install -r backend/requirements.export.txt
python backend/scripts/export_yolo_onnx.py --imgsz 480 --out weights/yolov8n.onnx
deactivate && rm -rf .export-venv
```

Il file ONNX è portabile, quindi in teoria puoi produrlo altrove — ma
`torch==2.4.1` ha wheel solo per Python 3.8-3.12, e su una macchina con un
Python più recente l'installazione fallisce. Sul Pi, dove Bookworm porta
Python 3.11, il problema non si pone.

### Senza Docker

```bash
make install     # venv + dipendenze + build della UI in backend/web
make run         # API su :8000
make worker      # in un secondo terminale
```

`make install` crea un virtualenv in `.venv`: su Raspberry Pi OS Bookworm non
è una preferenza ma un obbligo (PEP 668 rifiuta `pip install` nell'interprete
di sistema).

La UI viene compilata in `backend/web`, che è dove FastAPI la cerca: senza
quel passo l'API funziona ma il browser non vede nulla. Per un avvio
automatico al boot ci sono due unit systemd pronte in
[`deploy/systemd/`](deploy/systemd/).

### Con Docker

```bash
docker compose up -d --build
```

Due container (`api` e `worker`) che condividono un'immagine; il build della
UI è uno stage del Dockerfile, quindi qui non serve `make web`.

In entrambi i casi apri `http://padelpi.local:8000` e verifica con
`curl http://localhost:8000/api/health`.

Guida completa passo per passo: [INSTALL_RASPBERRY.md](INSTALL_RASPBERRY.md).

---

## Ripresa del video

L'accuratezza dipende quasi interamente dalla ripresa.

- **Camera fissa** su treppiede. Se si muove, la calibrazione decade.
- **In alto e dietro il fondo campo**, 3-4 m di altezza, tutto il campo nel
  fotogramma compresi i quattro angoli.
- **1080p a 30 fps** è sufficiente; il 4K non aggiunge nulla e rallenta il
  decode.
- **Evita zoom e stabilizzazione elettronica**: deformano la prospettiva
  fotogramma per fotogramma.
- Maglie di colore diverso fra i due della stessa coppia aiutano molto la
  ricostruzione dell'identità dopo le occlusioni.

---

## Tempi sul Pi 5

A `SAMPLE_HZ=5` l'analisi procede all'incirca in tempo reale: una partita di
60 minuti richiede circa 60 minuti. È un lavoro batch — si lancia e si guarda
il risultato dopo. `SAMPLE_HZ=3` scende a ~35 minuti, al prezzo di una
sottostima della distanza percorsa di circa il 15% (segnalata nei warning).

---

## Sviluppo

```bash
make install      # venv, dipendenze, build della UI
make test         # 124 backend + 14 frontend
make run          # API con reload
make worker       # worker, in un altro terminale
make check        # interroga /api/health
```

Installa **sempre** le dipendenze pinnate (`make` lo fa): pacchetti non
pinnati in locale producono test verdi che falliscono in CI. Per i soli
strumenti di test aggiungi `.venv/bin/pip install -r backend/requirements.dev.txt`.

Per lavorare sul frontend con hot reload serve il dev server di Vite, che
gira su un'altra porta e quindi richiede CORS:

```bash
echo 'CORS_ORIGINS=http://localhost:5173' >> .env
cd frontend && npm run dev
```

I test coprono calibrazione e validazione geometrica, campionamento video,
geometria del detector, tracking, ricostruzione identità, segmentazione
scambi, metriche, semantica della coda, round-trip degli artefatti e il
flusso API completo, più una prova end-to-end della pipeline su una partita
sintetica — inclusa la verifica che **il replay dagli artefatti riproduca
l'analisi originale**, senza la quale ogni soglia tarata con `retune.py`
sarebbe tarata su un sistema che non esiste.
