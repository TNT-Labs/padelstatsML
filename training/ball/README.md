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
2. Carica `train_tracknet.py` e gli zip, dall'icona cartella a sinistra.
3. Esegui:

```
!pip install -q opencv-python-headless onnx
!python train_tracknet.py palla_*.zip --epochs 30
```

Ad ogni epoca lo script stampa precisione e richiamo sulla validazione. Questi valori misurano gli spezzoni che la rete non ha mai visto, e la palla conta come trovata se è entro 4 px.

- Un F1 sopra 0,8 è buono.
- Se è basso, servono più spezzoni annotati.

Alla fine scarica `ball.onnx` e `ball.pt`. Prepara anche l'insieme di calibrazione per la conversione e scarica `calib.npy`:

```
!python make_calib.py palla_*.zip
```

## 3. Convertire per l'Hailo-8L (Windows, WSL2 con Ubuntu 22.04)

1. Registrati su hailo.ai, nella Developer Zone.
2. Scarica il **Dataflow Compiler** per Python 3.10 (file `hailo_dataflow_compiler-*.whl`), versione compatibile con HailoRT 4.20.
3. In WSL2 crea l'ambiente con il compilatore:

```
sudo apt install python3.10-venv graphviz-dev
python3.10 -m venv ~/hailo && source ~/hailo/bin/activate
pip install hailo_dataflow_compiler-*.whl
```

4. Copia nella stessa cartella `ball.onnx`, `calib.npy`, `ball.alls` e `hailo_compile.sh`, poi lancia:

```
./hailo_compile.sh ball.onnx calib.npy
```

Senza GPU ci vogliono 10–30 minuti. Il risultato è **`ball_h8l.hef`**.

## 4. Sul Raspberry

Copia `ball_h8l.hef` nella cartella `weights/` del progetto, la stessa di `yolov8n.onnx`.

L'integrazione nell'analisi, cioè palla a 30 fotogrammi al secondo e da lì colpi, battute e fine scambio, è il passo successivo.
