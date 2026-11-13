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
python training/hits/train_hits.py palla_run.tgz training/hits/labels/terzo_set.txt \
                                   secondo_run.tgz training/hits/labels/secondo_set.txt
```

Stato attuale: 180 colpi di due partite (22/10/2026, secondo e terzo set; i colpi segnati
sono in `training/hits/labels/`).

| Misura | Trovati | Giusti | Metà campo | Giocatore |
|---|---|---|---|---|
| Partita mai vista in addestramento | 76% | 74% | 89% | 85% |
| Scambi esclusi a turno, stesse partite | 77% | 83% | 97% | 92% |

La prima versione, su una sola partita, dava 91% di colpi giusti sugli scambi esclusi: su
una partita nuova erano il 73%. Colpi di altre partite, con altra luce e altre posizioni
della camera, sono il modo di avvicinare le due righe.
