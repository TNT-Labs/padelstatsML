# Tracciatura della palla: addestramento

La palla da padel, piccola e veloce, non la segue nessun modello già pronto. Qui si
addestra una rete del tipo TrackNet sui fotogrammi annotati nell'app, poi la si
converte per l'acceleratore Hailo-8L del Raspberry.

## 1. Annotare (sul PC, nel browser)

1. Statistiche della partita → **Annota palla**.
2. Clicca sul centro della palla, anche quando è una scia. La rete vuole il punto esatto: usa la lente.
3. Premi **N** o la barra spaziatrice quando la palla non si vede, cioè è dietro un giocatore o fuori inquadratura. Questi fotogrammi servono quanto gli altri.
4. Usa **A / D** o le frecce per andare avanti e indietro, e **Canc** per togliere un segno.
5. Annota **spezzoni interi**. La rete guarda 3 fotogrammi consecutivi, quindi un fotogramma annotato vale solo se ha i due precedenti.

Obiettivo: **25–40 spezzoni** (1.500–2.500 fotogrammi), meglio se da 2–3 partite con luce diversa. Al ritmo di un click al secondo servono circa 30–40 minuti per partita.

Poi premi **Scarica i dati annotati**: ottieni uno zip per partita.

## 2. Addestrare (Google Colab, GPU gratuita)

1. Apri colab.research.google.com e crea un nuovo notebook. Scegli *Runtime → Cambia tipo di runtime → GPU T4*.
2. Carica `train_tracknet.py` e gli zip, dall'icona cartella a sinistra. **Aspetta che il
   cerchio di caricamento in basso a sinistra sparisca**: uno zip di 100 MB richiede qualche
   minuto, e se lanci prima lo script lo trova tronco. Per zip grandi è più sicuro metterli su
   Google Drive e collegarlo (`from google.colab import drive; drive.mount('/content/drive')`).
   Controllo: `!unzip -tq palla_*.zip` deve rispondere *No errors detected*.
3. Esegui:

```
!pip install -q opencv-python-headless onnx
!python train_tracknet.py palla_*.zip --epochs 30
```

Ad ogni epoca lo script stampa precisione e richiamo sulla validazione. Questi valori misurano gli spezzoni che la rete non ha mai visto, e la palla conta come trovata se è entro 4 px.

- Un F1 sopra 0,8 è buono.
- Se è basso, servono più spezzoni annotati.

Attenzione: di serie la validazione prende spezzoni a caso **delle stesse partite** dell'addestramento,
e gli spezzoni vicini a quelli di validazione sono in addestramento: il numero è ottimista. Con
`--holdout` la validazione è invece una partita intera, mai vista in addestramento (uno zip è una
partita):

```
!python train_tracknet.py palla_*.zip --epochs 30 --holdout palla_terzo.zip
```

È la misura che vale per una partita nuova: usala per confrontare due addestramenti, sullo stesso
zip escluso. Il modello così addestrato non ha visto quella partita: per quello da mettere sul
Raspberry si addestra di nuovo su tutto, senza `--holdout`.

Alla fine scarica `ball.onnx` e `ball.pt`. Prepara anche l'insieme di calibrazione per la conversione e scarica `calib.npy`:

```
!python make_calib.py palla_*.zip
```

### Quanto rendono i picchi deboli (senza addestrare)

L'analisi tiene, oltre alla palla, i picchi più deboli della mappa (fino a 3 per fotogramma, sopra
0,15) e li usa solo dove una traiettoria della palla stava andando. Per misurarlo sugli spezzoni
annotati, con la rete che hai già, carica anche `peaks_check.py`, `ball.py` e `ball_track.py` (da
`backend/app/ml`) ed esegui:

```
!python peaks_check.py ball.pt palla_*.zip
```

Stampa tre cose sugli spezzoni di validazione: la misura di sempre fotogramma per fotogramma; quante
volte la palla non presa è fra i picchi deboli e quanti picchi deboli ha in media un fotogramma; e
precisione e richiamo delle traiettorie (quelle che usano le statistiche) senza picchi deboli e con
ciascuna regolazione del recupero (solo nei buchi, solo oltre i capi, entrambi; soglia; tolleranza),
con i punti aggiunti divisi in giusti, nel posto sbagliato e messi dove la palla era coperta. Il
recupero si accende (`RECOVERY` in `ball_track.py`) solo con una regolazione che aggiunge richiamo
senza togliere precisione. Senza `--holdout` gli spezzoni di validazione sono delle stesse
partite dell'addestramento (misura ottimista); con `--holdout palla_X.zip` vale solo per un `ball.pt`
addestrato anch'esso con `--holdout palla_X.zip`.

### La metà lontana, ingrandita (seconda passata)

Lontano la palla è circa 2 pixel per la rete: confusa col pavimento. L'analisi può guardare una
seconda volta la sola metà lontana con l'aria sopra (pallonetti), a circa il doppio della scala:
stessa rete, addestrata anche su quei ritagli.

1. In **Annota palla**, per ogni partita annotata: **Scarica i dati annotati** e **Scarica la metà
   lontana**. Il secondo zip (`lontana_<id>.zip`) ha gli stessi fotogrammi e gli stessi segni,
   ritagliati sul Pi a piena risoluzione: nessun clic nuovo.
2. Prima la misura, su una partita esclusa (qui `3628b11c`; usa l'id di una delle tue):

```
!python train_tracknet.py palla_*.zip lontana_*.zip --epochs 30 --holdout 3628b11c --out holdout
!python far_check.py holdout.pt palla_*.zip lontana_*.zip --match 3628b11c
```

   `train_tracknet.py` stampa alla fine il punteggio zip per zip; `far_check.py` (carica anche
   `ball.py` e `ball_track.py` da `backend/app/ml`) confronta sulla partita esclusa la palla dell'immagine
   intera con quella della seconda passata, fotogramma per fotogramma e come traiettorie, metà lontana e
   resto separati, con tre regole per unire le due risposte.
3. Se la metà lontana migliora senza peggiorare il resto, la rete per il Raspberry su tutto:

```
!python train_tracknet.py palla_*.zip lontana_*.zip --epochs 30
!python make_calib.py palla_*.zip lontana_*.zip
```

   e poi la conversione come sotto. Se l'esportazione finale fallisce (`onnx` mancante: la sessione
   Colab si è riavviata e ha perso il `pip install`), non serve riaddestrare:
   `!pip install -q onnx` e poi `!python train_tracknet.py --export ball.pt --out ball`. Sul Raspberry `BALL_FAR=on` nel `.env` accende la seconda passata:
   costa una inferenza in più a fotogramma (circa +0,5 volte la durata del video).

Misura del 14/11/2026, partita `3628b11c` esclusa (rete addestrata sull'altra, 857 fotogrammi segnati),
traiettorie: nella metà lontana F1 da 0,749 (precisione 0,848, richiamo 0,671) con la sola immagine
intera a 0,845 (0,898, 0,798) con la seconda passata, tenendo nel ritaglio solo ciò che la seconda
passata conferma (la regola in uso, `FAR_MERGE` in `ball_track.py`); nel resto del campo 0,828 → 0,817.

Sul Raspberry (14/11/2026, secondo set, rete addestrata su entrambi i set: la rete ha visto 2 minuti
di quella partita, la misura è un po' ottimista; modelli dei colpi addestrati solo sul terzo set):
analisi da 1,82 a 2,10 volte la durata del video; colpi segnati trovati da 52 a 56 su 65, colpi indicati
non veri da 19 a 13, colpi persi per la palla da 4 a 1; palla vista nel mezzo secondo prima dei colpi
della coppia lontana dal 59% al 64% dei fotogrammi (dopo: dal 63% al 70%). Scambi giusti sui 24 punti
segnati, con la lettura dei punti addestrata sugli altri due set: 18 prima e dopo.

Di serie l'addestramento ora usa variazioni più forti degli esempi (`--augment strong`: anche zoom,
spostamento, sfocatura, rumore e colore); `--augment base` torna a quelle delle prime reti.

## 3. Convertire per l'Hailo-8L (Windows, WSL2 con Ubuntu 22.04)

1. Registrati su hailo.ai, nella Developer Zone.
2. Scarica il **Dataflow Compiler** per Linux e Python 3.10, cioè un file
   `hailo_dataflow_compiler-*-cp310-*.whl`. Deve essere la versione abbinata a HailoRT 4.20,
   come indicato nella tabella di compatibilità della Developer Zone: un `.hef` compilato con
   un'altra versione non si carica sul Raspberry.
3. Il compilatore vuole **Ubuntu 22.04**, che ha Python 3.10. L'Ubuntu predefinito di WSL è il
   24.04, con Python 3.12, e lì `python3.10-venv` non esiste. In PowerShell installa una seconda
   distribuzione, accanto all'altra:

```
wsl --install -d Ubuntu-22.04
```

   Poi aprila (menu Start → *Ubuntu 22.04*) e controlla con `lsb_release -rs` che risponda `22.04`.
4. Lì crea l'ambiente con il compilatore, **un comando alla volta**. Il file `.whl` scaricato in
   Windows si trova in `/mnt/c/Users/<nome>/Downloads/`.

```
sudo apt update
sudo apt install -y python3.10-venv python3.10-dev python3-tk graphviz libgraphviz-dev
python3.10 -m venv ~/hailo
source ~/hailo/bin/activate
pip install --upgrade pip
pip install /mnt/c/Users/<nome>/Downloads/hailo_dataflow_compiler-*.whl
hailo --version
```

   Il controllo iniziale chiede 16 GB di RAM, ma per questa rete piccola ne bastano meno. WSL
   usa metà della RAM di Windows: per dargliene di più, e una memoria di scambio ampia, crea in
   Windows il file `C:\Users\<nome>\.wslconfig` con:

```
[wsl2]
memory=6GB
swap=16GB
```

   Usa per `memory` la RAM del PC meno 2 GB (Gestione attività → Prestazioni → Memoria). Poi in
   PowerShell esegui `wsl --shutdown` e riapri Ubuntu 22.04. Con `free -g` vedi la nuova memoria.

5. Copia nella stessa cartella `ball.onnx`, `calib.npy`, `ball.alls` e `hailo_compile.sh`, poi lancia:

```
./hailo_compile.sh ball.onnx calib.npy
```

Senza GPU ci vogliono 10–30 minuti. Il risultato è **`ball_h8l.hef`**.

## 4. Sul Raspberry

Copia `ball_h8l.hef` nella cartella `weights/` del progetto, la stessa di `yolov8n.onnx`.

L'integrazione nell'analisi, cioè palla a 30 fotogrammi al secondo e da lì colpi, battute e fine scambio, è il passo successivo.
