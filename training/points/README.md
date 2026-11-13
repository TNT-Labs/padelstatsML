# Punti segnati: la verità per gli scambi

`labels/` contiene i punti segnati in app con **Segna i punti** (B alla battuta, F quando il punto è
deciso, 1/2 per la coppia vincente), copiati con "Copia i punti segnati":

| File | Set | Tratto | Punti |
|---|---|---|---|
| `primo_set.txt` | 22/10, primo set (`primo_run.tgz`) | min 6–18 | 32 |
| `secondo_set.txt` | 22/10, secondo set (`secondo_run.tgz`) | min 1–10 | 24 |

La misura: `backend/scripts/eval_points.py RUN PUNTI [...]` (scambi giusti uno a uno, persi,
spezzati, uniti ad altri, falsi, scarto di inizio e fine).

## Stato al 13/11/2026 (modelli dei colpi che non hanno visto il set)

| Set | Giusti | Persi | Spezzati | Uniti ad altri | Falsi | Battuta riconosciuta |
|---|---|---|---|---|---|---|
| Primo, 32 punti | 19 | 3 | 1 | 9 | 4 | 14 su 19 |
| Secondo, 24 punti | 13 | 1 | 0 | 10 | 3 | 10 su 13 |

L'errore principale sono i **punti uniti**: le pause fra un punto e l'altro durano 7–13 s, i
giocatori si muovono abbastanza da sembrare in gioco, e il modello dei colpi trova "colpi" nelle
pause (la palla ripassata a chi batte, i palleggi prima della battuta: 56–58 per set nel tratto
segnato), che tengono uniti i punti. Il modello dei colpi è addestrato su ±1 s attorno ai colpi
segnati: delle pause ne ha visti 4,5 s su 349.

## Provato, senza un miglioramento che regga su entrambi i set

Sempre addestrato su un set e misurato sull'altro, nei due sensi.

- **Pause come esempi negativi del modello dei colpi** (`pause_exp.py`). Secondo set: giusti da
  13 a 21 su 24, uniti da 10 a 0, ma colpi segnati trovati dall'80% al 62%. Primo set: giusti da
  19 a 17. Usandone una parte (un punto della palla su 3 o su 10) il secondo resta meglio (18, 14),
  il primo peggio (17, 17). Segno opposto sui due set: non adottato.
- **Ogni colpo trovato classificato battuta / in gioco / pausa** (`phase_exp.py`), dai giocatori
  (velocità prima e dopo, chi è in fondo, formazione di battuta), dal tempo fra i colpi. Separa
  (AUC battuta 0,91 e 0,84, pausa 0,80), ma con soglie scelte sull'altro set: primo 19 → 23 giusti
  con 9 scambi falsi invece di 4, secondo 13 → 13. Con le etichette giuste sugli stessi colpi si
  arriverebbe a 30/32 e 22/24: i colpi trovati bastano, mancano esempi per riconoscerli (21–28
  battute per set).

Gli script sono in `experiments/`: girano sui run estratti (secondo in `runs/b`, primo in `runs/c`) e sulla cache dei dati dei colpi di train_hits (`datas.pkl`: terzo e secondo); i percorsi sono in testa ai file.
