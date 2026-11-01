# Padel Stats — analisi partite su Raspberry Pi 5

Sistema self-hosted che analizza il video di una partita di padel e produce
statistiche di **posizione e movimento** dei quattro giocatori: distanza
percorsa, occupazione del campo, zone, scambi.

Gira interamente su un Raspberry Pi 5. Nessun cloud, nessuna GPU, nessun
dato che esce di casa. Con un AI HAT+ (Hailo) montato, riconosciuto da solo,
il rilevatore dei giocatori gira sulla scheda e si aggiungono la **postura**
(smash e bandeja) e la **palla**; senza, tutto il resto gira sul processore
come prima (INSTALL_RASPBERRY.md §6b).

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
| Distanza percorsa (totale e negli scambi) | somma dei passi sul percorso filtrato (Kalman), al netto del tremolio del rilevatore |
| Velocità media e di punta | dal percorso filtrato, 95° percentile per la punta |
| Occupazione del campo (heatmap 0,5 m) | istogramma delle posizioni dei piedi |
| Quota di tempo a rete / metà campo / fondo | distanza dalla linea di rete |
| Area di campo coperta | celle occupate per almeno 1 s × 0,25 m² |
| Numero, durata e distribuzione degli scambi | segmentazione dal movimento dei giocatori, ogni scambio aperto dalla battuta |
| Tempo di gioco effettivo | rapporto scambi / durata |
| Copertura del tracciamento per giocatore | quanto della partita ciascuno è stato seguito |
| Smash e bandeja *(con Hailo)* | braccio della racchetta sopra la spalla durante lo scambio, dalla postura; si distinguono dal braccio libero e dalla distanza dalla rete, secondo la mano di ciascuno (destro/mancino). Verificabili colpo per colpo nell'app: 29 su 30 giusti sulla partita di prova |
| Palla *(con Hailo)* | rete addestrata sui fotogrammi annotati nell'app, su ogni fotogramma; visibile sul video della partita |
| Scambi dalla battuta all'ultimo colpo *(con Hailo)* | i colpi correggono gli scambi trovati dal movimento: prima finivano in media 8 s dopo l'ultimo colpo (i giocatori si muovono ancora), ora sull'ultimo colpo; uno scambio dove la palla resta ferma a lungo si divide, due pezzi con colpi continui si uniscono |
| Da dove colpisce ogni giocatore *(con Hailo)* | per ogni colpo la posizione di chi lo fa: colpi da rete, metà campo e fondo, mappa dei punti, chi chiude lo scambio (ultimo colpo) |
| Colpi per scambio e per giocatore *(con Hailo)* | dalla palla e dai giocatori, con modelli addestrati sui colpi segnati a mano (`training/hits`); su scambi esclusi dall'addestramento: trovati 77%, giusti 91%, giocatore giusto 92% |

### Non misurato — e perché

Dritti, rovesci e volée, vincenti ed errori, velocità della palla,
punteggio.

Richiedono il **tracciamento della palla**. Un Pi 5 senza
acceleratore non lo può fare a un frame rate utile: una pallina da padel a
90 km/h percorre 80 cm per frame a 30 fps e occupa pochi pixel, quindi serve
inferenza su ogni frame — ore di elaborazione per ogni partita, con modelli
che pesano più del detector di persone.

La versione precedente li riportava comunque, prodotti da euristiche
(picchi di accelerazione su una traiettoria già filtrata, un albero
decisionale a soglie fisse, "palla vicino al bordo = errore"). I numeri
avevano l'aspetto di misure e non lo erano. Sono stati rimossi: **meglio
sette metriche vere che venti plausibili.**

Con l'acceleratore (AI HAT+, Hailo) il tracciamento della palla torna alla
portata e queste metriche possono rientrare — con un modello addestrato sui
video veri e delle metriche di validazione, non con delle soglie. La palla
oggi è seguita (vedi *Con l'acceleratore*); colpi per scambio, chi colpisce
e fine dello scambio sono il passo successivo.

Dritto e rovescio dalla sola postura sono stati provati e tolti: con la
camera a 2 m il polso è coperto in metà dei colpi e la stima della postura
trema quanto un'oscillazione della racchetta. Su 20 colpi verificati, 0
erano giusti.

---

## Come funziona

### 1. Calibrazione del campo — obbligatoria

L'utente trascina quattro maniglie sugli angoli del campo in un fotogramma
del video. Mentre trascina, l'interfaccia proietta il modello del campo
(rete, linee di servizio, perimetro) sull'immagine: se la rete calcolata
coincide con quella reale, la calibrazione è corretta. Serve mezzo minuto.

Se la camera è vicina e un angolo resta fuori dall'inquadratura,
**Estendi oltre l'inquadratura** aggiunge una cornice attorno al fotogramma:
l'angolo si posiziona lì, dove si incontrerebbero le linee visibili, e
l'omografia estrapola il campo intero.

Con la camera bassa e vicina al vetro di fondo gli angoli vicini cadono
anche a più di un'inquadratura di distanza, dove metterli a mano è solo una
stima. Con **Punti di riferimento → Rete e linea di servizio** le quattro
maniglie vanno invece sulla base dei due pali della rete e sulle due
estremità della linea di servizio vicina, punti visibili o quasi; gli angoli
li calcola l'app.

La calibrazione si salva come **preset camera**, insieme al fotogramma su
cui è stata fatta. Ogni video nuovo viene confrontato con i fotogrammi dei
preset (pareti, recinzioni, tetto, pali: ciò che non si muove): se la camera
è nella stessa posizione, o spostata di poco, la calibrazione del preset viene
adattata di quei pochi pixel e applicata da sola, e la partita è subito pronta
da analizzare. Se la camera si è mossa troppo non succede nulla: il preset
resta a disposizione con un clic.

Per una posizione della camera mai vista l'app propone da sola i quattro
punti *Rete e linea di servizio*: su uno sfondo senza giocatori (mediana di
fotogrammi presi lungo il video) trova il nastro della rete, la linea di
servizio vicina, la linea centrale, la corda inferiore della rete e il bordo
del pavimento contro le pareti, e controlla che tutto sia geometricamente
coerente. Sul video reale di prova l'errore medio sul campo è 0,2 m. Resta una
*proposta*: per una camera nuova si conferma con un clic (o si correggono i
punti), perché una calibrazione sbagliata produce metri sbagliati senza alcun
segnale. Quando non trova nulla di coerente non propone niente.

### 2. Analisi — una sola decodifica del video

```
decode (una volta, con frame-skip)
   ├─▶ palla su ogni frame (solo con Hailo e la rete della palla)
   └─▶ detector persone (ONNX o Hailo, ritagliato sul campo) + postura (Hailo)
        └─▶ tracking in metri sul piano del campo
             └─▶ identità: 4 giocatori da coppia e lato in campo
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
- **Tracklet corte, identità dal posto in campo.** Le tracce muoiono e
  rinascono liberamente; a valle ogni pezzo riceve una coppia (la metà campo
  in cui gioca, seguendo i cambi di campo) e un lato (drive o revés). L'aspetto
  — re-ID o colore della divisa — corregge i pezzi che somigliano chiaramente
  al compagno, così uno scambio di lato viene seguito. L'aspetto da solo non
  distingueva i compagni su una partita reale; dove stanno sì.
- **Scambi dal movimento dei giocatori,** non dalla palla: durante il punto
  tutti e quattro si muovono, fra un punto e l'altro no. Ogni punto inizia
  con la battuta: i giocatori fermi ai loro posti (chi batte e chi riceve in
  fondo, in diagonale) e poi il gioco che riparte. Lì inizia un nuovo
  scambio, anche se la pausa dal punto precedente era troppo breve per
  separarli.

### Con l'acceleratore: postura e palla

Con l'AI HAT+ montato (`scripts/hailo_setup.sh` sul Pi, INSTALL_RASPBERRY.md
§6b) il worker lo trova da solo. Rilevatore dei giocatori e postura girano
sulla scheda, e l'analisi dura meno della metà. Se la scheda manca o si
blocca, l'analisi prosegue sul processore e lo scrive nel pannello di
affidabilità. `docker exec padel-worker python scripts/hailo_check.py`
diagnostica la scheda.

**Postura.** Per ogni giocatore i 17 punti del corpo (YOLOv8 pose su un
mosaico dei ritagli dei giocatori, così anche quelli lontani sono abbastanza
grandi). Lo **scheletro** si vede sul video della partita. Da lì vengono gli
smash e le bandeje, che si verificano colpo per colpo nell'app con
**Verifica**. Destro o mancino si indica in **Rinomina giocatori**.

**Palla.** Nessuna rete già pronta segue una pallina da padel in questi
video, quindi se ne addestra una sui propri:

1. **Annota palla** (statistiche, editor): un clic sulla palla per fotogramma,
   in spezzoni da 2 s presi dentro gli scambi. Servono 25–40 spezzoni.
2. **Scarica i dati annotati**, poi addestra su Google Colab e compila per
   Hailo in WSL2: guida passo passo in `training/ball/README.md`.
3. Copia `ball_h8l.hef` in `weights/`. Con
   `docker exec padel-worker python scripts/ball_check.py --start 60` la provi
   su 20 secondi di partita: velocità, quota di fotogrammi con la palla, e un
   video `ball_check.mp4` con la palla cerchiata.

Da lì ogni analisi la segue su **ogni** fotogramma (`BALL=auto`, `off` per
spegnerla): tre fotogrammi consecutivi in, la posizione della palla out,
17 ms sulla Hailo-8L. Le risposte grezze finiscono in `ball.jsonl`, e
`ball_track.py` le unisce in traiettorie: scarta i punti isolati, i salti e
le cose ferme come luci o palle a terra. Sul video della partita
l'interruttore **Palla** mostra la palla e la sua scia.

La rete di prova (26 spezzoni annotati) trova 3 palle visibili su 4 entro
4 px, con l'80% delle risposte giuste. Annotare spezzoni di altre partite,
con altra luce, è il modo di migliorarla.

### 3. Verifica e taratura

I numeri hanno sempre un aspetto plausibile: una distanza di 2 480 m e una
heatmap sbilanciata verso la rete sono indistinguibili da quelle giuste,
anche se due giocatori sono stati scambiati per metà partita. L'unico modo
per saperlo è **guardare i fotogrammi**.

Il modo più diretto è dalla pagina delle statistiche: **Video della partita**
riproduce il video caricato con il riquadro e il nome di ogni giocatore
sopra. Si segue a colpo d'occhio se un nome resta sulla stessa persona; un
clic su uno scambio o su un cambio di campo porta il video a quel momento.

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

# Come sono stati trovati i 4 giocatori: coppia, lato, copertura, cambi di
# campo trovati, quanto ha deciso l'aspetto; e se c'è una quinta persona
python scripts/diagnose_identity.py latest

# Rifà le foto dei giocatori dal video, senza rianalizzare la partita
python scripts/rethumb.py latest

# Con l'AI HAT+: la scheda, la sua rete e una prova su un fotogramma
python scripts/hailo_check.py [video]
# La rete della palla su 20 s di partita, con video di controllo
python scripts/ball_check.py [video o id] --start 60 --seconds 20
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
cp .env.example .env        # imposta DATA_VOLUME e PADEL_ADMIN_PASSWORD

# Esporta i modelli una volta sola, direttamente sul Pi
python3 -m venv .export-venv && source .export-venv/bin/activate
pip install -r backend/requirements.export.txt
python backend/scripts/export_yolo_onnx.py --imgsz 480 --out weights/yolov8n.onnx
python backend/scripts/export_reid_onnx.py --out weights/osnet_x0_25_msmt17.onnx
deactivate && rm -rf .export-venv
```

Il secondo modello (re-identificazione, OSNet) è facoltativo ma consigliato:
distingue i giocatori dall'aspetto e non solo dal colore della divisa, che
non basta quando i compagni sono vestiti uguali. Costa circa il 25% di tempo
di analisi in più. Senza, l'identità usa il solo colore.

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

Con Docker l'app è pubblicata su **https://shopbeautylab.it/padel/** dal
tunnel Cloudflare dello stack VORTICE (nessuna porta aperta sul Pi; regola
del tunnel in [INSTALL_RASPBERRY.md §5b](INSTALL_RASPBERRY.md#5b-pubblicazione-su-shopbeautylabit)).
Senza Docker, in rete locale, apri `http://padelpi.local:8000` (con
`BASE_PATH=` vuoto e `COOKIE_SECURE=false`).

Si entra con nome utente e password: il primo amministratore viene dal
`.env`, gli altri utenti li crea lui da **Gestione utenti**. Ognuno vede le
proprie partite, l'amministratore tutte; un account *in sola visualizzazione*
vede le partite già analizzate di tutti e non può modificare nulla
([§11b](INSTALL_RASPBERRY.md#11b-utenti-e-accesso)).

Guida completa passo per passo: [INSTALL_RASPBERRY.md](INSTALL_RASPBERRY.md).

---

## Ripresa del video

L'accuratezza dipende quasi interamente dalla ripresa.

- **Camera fissa** su treppiede. Se si muove, la calibrazione decade.
- **In alto e dietro il fondo campo**, 3-4 m di altezza, tutto il campo nel
  fotogramma compresi i quattro angoli. Più in basso la rete copre i piedi
  della coppia lontana: il riquadro del giocatore si chiude sul nastro e,
  proiettato a terra, lo collocherebbe metri più indietro. L'analisi lo
  riconosce e stima i piedi dal resto del corpo (altezza e larghezza del
  riquadro, confrontate con quelle dei giocatori della metà vicina): con la
  camera a 3-4 m l'errore resta sotto il mezzo metro, sotto i 2,5 m circa
  sale a 1-2 m. L'altezza della camera stimata è fra gli avvisi.
- **Rilevazioni incerte**: sotto `DETECTOR_CONF` e fino a
  `DETECTOR_CONF_WEAK` (0,12) servono solo a non perdere un giocatore già
  seguito, vicino a dove era atteso; non creano mai giocatori nuovi.
- **Giocatori lontani piccoli** (camera lontana dal campo o risoluzione
  bassa): il rilevatore guarda una seconda volta la sola metà lontana,
  in automatico quando i giocatori in fondo gli arriverebbero troppo
  piccoli (`DETECTOR_FAR_PASS=auto`, oppure `true`/`false` per forzarlo).
  Costa il 50-80% di tempo in più.
- **1080p a 30 fps** è sufficiente; il 4K non aggiunge nulla e rallenta il
  decode.
- **Evita zoom e stabilizzazione elettronica**: deformano la prospettiva
  fotogramma per fotogramma.
- **Grandangolo (0.6x, action cam)**: si può usare. La curvatura delle linee
  si corregge nella calibrazione (cursore *Obiettivo grandangolare*) oppure
  in automatico da *Verifica accuratezza → Correggi la calibrazione*.
- Maglie di colore diverso fra i due della stessa coppia aiutano molto la
  ricostruzione dell'identità dopo le occlusioni.

---

## Tempi sul Pi 5

A `SAMPLE_HZ=5` l'analisi procede all'incirca in tempo reale: una partita di
60 minuti richiede circa 60 minuti. È un lavoro batch — si lancia e si guarda
il risultato dopo. `SAMPLE_HZ=3` scende a ~35 minuti, al prezzo di una
sottostima della distanza percorsa di circa il 15% (segnalata nei warning).

Con l'AI HAT+ (Hailo-8L) l'analisi richiede meno della metà del tempo: una
partita è stata analizzata in circa 30 minuti, con il Pi sotto gli 80 °C con
una ventola. Seguire la palla allunga l'analisi, perché vanno letti tutti i
fotogrammi e non solo quelli campionati. Il tempo con la palla è ancora da
misurare sul Pi; `BALL=off` torna ai tempi senza palla.

---

## Sviluppo

```bash
make install      # venv, dipendenze, build della UI
make test         # 398 backend + 52 frontend
make run          # API con reload
make worker       # worker, in un altro terminale
make check        # interroga /api/health
```

Installa **sempre** le dipendenze pinnate (`make` lo fa): pacchetti non
pinnati in locale producono test verdi che falliscono in CI. Per i soli
strumenti di test aggiungi `.venv/bin/pip install -r backend/requirements.dev.txt`.

Per lavorare sul frontend con hot reload serve il dev server di Vite, che
inoltra `/api` all'API su :8000 (avviata con `BASE_PATH=` vuoto e
`COOKIE_SECURE=false`, perché il dev server è su http):

```bash
cd frontend && npm run dev
```

I test coprono calibrazione e validazione geometrica, campionamento video,
acceleratore (con una scheda finta), postura, colpi alti, palla e sue
traiettorie,
geometria del detector, tracking, ricostruzione identità, segmentazione
scambi, metriche, semantica della coda, round-trip degli artefatti e il
flusso API completo, più una prova end-to-end della pipeline su una partita
sintetica — inclusa la verifica che **il replay dagli artefatti riproduca
l'analisi originale**, senza la quale ogni soglia tarata con `retune.py`
sarebbe tarata su un sistema che non esiste.
