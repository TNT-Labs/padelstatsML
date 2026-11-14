# Punti segnati: la verità per gli scambi

`labels/` contiene i punti segnati in app con **Segna i punti** (B alla battuta, F quando il punto è
deciso, 1/2 per la coppia vincente), copiati con "Copia i punti segnati":

| File | Set | Tratto | Punti |
|---|---|---|---|
| `primo_set.txt` | 22/10, primo set (`primo_run.tgz`) | min 6–18 | 32 |
| `secondo_set.txt` | 22/10, secondo set (`secondo_run.tgz`) | min 1–10 | 24 |
| `terzo_set.txt` | 22/10, terzo set (`palla_run.tgz`) | min 1–15 | 33 |

La misura: `backend/scripts/eval_points.py RUN PUNTI [...]` (scambi giusti uno a uno, persi,
spezzati, uniti ad altri, falsi, scarto di inizio e fine).

## Lettura dei punti dai colpi (in uso)

Ogni colpo trovato è letto come **battuta**, **in gioco** o **fra due punti** da due piccoli modelli
(`backend/app/ml/phase_model.json`, codice in `backend/app/ml/phases.py`): dai giocatori (velocità
prima e dopo, chi sta in fondo, chi colpisce e da dove, formazione di battuta vista dal movimento) e
dai colpi attorno. Uno scambio va dalla battuta all'ultimo colpo in gioco; i colpi fra due punti non
contano (neanche nelle statistiche dei colpi).

```
python training/points/train_phases.py \
    --set primo   primo_far.tgz   training/points/labels/primo_set.txt \
    --set secondo secondo_far.tgz training/points/labels/secondo_set.txt training/hits/labels/secondo_set.txt \
    --set terzo   terzo_far.tgz   training/points/labels/terzo_set.txt   training/hits/labels/terzo_set.txt
```

Ogni set misurato con modelli (e soglie) addestrati sugli altri due, e con colpi trovati da modelli
dei colpi che non l'hanno visto:

| | Giusti | Uniti ad altri | Scambi falsi | Persi | Spezzati |
|---|---|---|---|---|---|
| Senza la lettura dei punti | 62/89 | 26 | 10 | 1 | 0 |
| Con la lettura dei punti | 71/89 | 7 | 20 | 4 | 7 |

Analisi del 14/11/2026 con la seconda passata sulla metà lontana (`BALL_FAR=on`) e modelli dei colpi
riaddestrati su quella palla. Con la palla di prima: 56 → 69 giusti. Gli errori in totale restano
simili (37 senza, 38 con): la lettura scambia punti uniti con spezzoni falsi e punti spezzati, ma porta
più scambi a corrispondere esattamente a un punto, che è la misura scelta. Pesare di più i falsi nella
scelta delle soglie (`--false-cost 1` o `2`) non abbassa il totale degli errori (38 e 38).

Per set, con la palla di prima: primo 17 → 23 su 32, secondo 13 → 18 su 24, terzo 26 → 28 su 33; con la seconda passata: primo 24, secondo 21, terzo 26. Tutti i set sono della
stessa serata e dello stesso campo: punti di un'altra serata sono il modo di sapere se tiene.

## Stato prima della lettura dei punti (modelli dei colpi che non hanno visto il set)

| Set | Giusti | Persi | Spezzati | Uniti ad altri | Falsi | Battuta riconosciuta |
|---|---|---|---|---|---|---|
| Primo, 32 punti | 19 | 3 | 1 | 9 | 4 | 14 su 19 |
| Secondo, 24 punti | 13 | 1 | 0 | 10 | 3 | 10 su 13 |

L'errore principale sono i **punti uniti**: le pause fra un punto e l'altro durano 7–13 s, i
giocatori si muovono abbastanza da sembrare in gioco, e il modello dei colpi trova "colpi" nelle
pause (la palla ripassata a chi batte, i palleggi prima della battuta: 56–58 per set nel tratto
segnato), che tengono uniti i punti. Il modello dei colpi è addestrato su ±1 s attorno ai colpi
segnati: delle pause ne ha visti 4,5 s su 349.

## Provato con due set, prima del terzo

Sempre addestrato su un set e misurato sull'altro, nei due sensi.

- **Pause come esempi negativi del modello dei colpi** (`pause_exp.py`). Secondo set: giusti da
  13 a 21 su 24, uniti da 10 a 0, ma colpi segnati trovati dall'80% al 62%. Primo set: giusti da
  19 a 17. Usandone una parte (un punto della palla su 3 o su 10) il secondo resta meglio (18, 14),
  il primo peggio (17, 17). Segno opposto sui due set: non adottato.
- **Ogni colpo trovato classificato battuta / in gioco / pausa** (`phase_exp.py`), addestrato su un
  set solo: separava (AUC battuta 0,91 e 0,84, pausa 0,80) ma le soglie non tenevano sull'altro set.
  Con il terzo set (addestramento su due) è diventata la lettura dei punti in uso, sopra.

Gli script sono in `experiments/`: girano sui run estratti (secondo in `runs/b`, primo in `runs/c`) e sulla cache dei dati dei colpi di train_hits (`datas.pkl`: terzo e secondo); i percorsi sono in testa ai file.
