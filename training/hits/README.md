# Colpi dalla palla: addestramento

I colpi (quanti per scambio, chi colpisce) li riconoscono tre piccoli modelli in
`backend/app/ml/hit_model.json`, addestrati sui colpi segnati a mano nell'app.

1. **Segna i colpi** (statistiche, sotto il video): pausa sul fotogramma in cui la palla
   lascia la racchetta e tasto 1–4. Vanno segnati scambi interi, battuta compresa: un
   colpo dimenticato in uno scambio segnato insegna al modello che lì non c'era.
2. **Copia i colpi segnati** e salva il testo in un file, per esempio `colpi.txt`.
3. Sul Pi, i dati dell'analisi di quella partita:
   ```
   docker exec padel-worker sh -c 'cd /data/artifacts/<id partita> && tar czf /data/run.tgz meta.json tracks.jsonl ball.jsonl result.json'
   docker cp padel-worker:/data/run.tgz .
   ```
4. Su un computer con Python e scikit-learn (`pip install scikit-learn numpy opencv-python-headless`),
   dalla cartella del progetto:
   ```
   python training/hits/train_hits.py run.tgz colpi.txt
   ```
   Lo script misura i modelli su gruppi di colpi lasciati fuori dall'addestramento a turno
   (colpi trovati, giusti, metà campo, giocatore), poi li addestra su tutto e riscrive
   `hit_model.json`. Le misure finiscono nel pannello dell'app.
5. Dopo il commit e l'aggiornamento del Pi, **Aggiorna statistiche** ricalcola i colpi.

Con più partite si passano più coppie, e lo script misura anche ogni partita con modelli
addestrati solo sulle altre: è la misura che vale per una partita nuova, ed è quella
mostrata nell'app.

```
python training/hits/train_hits.py terzo_far.tgz training/hits/labels/terzo_set.txt \
                                   secondo_far.tgz training/hits/labels/secondo_set.txt
```

Stato attuale: 180 colpi di due partite (22/10/2026, secondo e terzo set; i colpi segnati
sono in `training/hits/labels/`).

| Misura | Trovati | Giusti | Metà campo | Giocatore |
|---|---|---|---|---|
| Partita mai vista in addestramento | 87% | 84% | 92% | 86% |
| Scambi esclusi a turno, stesse partite | 92% | 90% | 98% | 91% |

Misure del 14/11/2026, con le analisi rifatte con la seconda passata sulla metà lontana
(`BALL_FAR=on`, training/ball/README.md). Con la palla di prima erano: partita mai vista 76% · 74% ·
89% · 85%. La rete della palla ha visto circa 2 minuti di spezzoni annotati di ciascuna partita: una
parte dei colpi segnati può cadere in quegli spezzoni, quindi la misura è un po' ottimista.

La prima versione, su una sola partita, dava 91% di colpi giusti sugli scambi esclusi: su
una partita nuova erano il 73%. Colpi di altre partite, con altra luce e altre posizioni
della camera, sono il modo di avvicinare le due righe.

## Dove si perdono i colpi

```
python training/hits/diagnose_hits.py palla_run.tgz training/hits/labels/terzo_set.txt \
                                      secondo_run.tgz training/hits/labels/secondo_set.txt
```

Ogni partita letta con modelli addestrati sulle altre: per ogni colpo segnato e non trovato la
causa (palla non vista, palla vista da un lato solo, giocatore senza riquadro, modello, decodifica),
per metà vicina e lontana; dei trovati quanti hanno metà campo o giocatore sbagliati; dei colpi
indicati non veri quanti sono vicini a un colpo vero (doppioni) o a uno perso (colpo giusto al tempo
sbagliato); quanta palla si vede nel mezzo secondo prima e dopo i colpi.

Sui due set del 22/10 (13/11/2026): dei 43 colpi persi su 180, 21 per la palla (17 vista da un lato
solo, 15 di questi della coppia lontana), 19 per modello e decodifica, 3 per il giocatore senza
riquadro. Nel mezzo secondo prima di un colpo la palla si vede nel 70-81% dei fotogrammi per la coppia
vicina, nel 42-59% per quella lontana.

