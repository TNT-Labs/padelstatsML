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

Prima versione: 115 colpi di una partita (22/10/2026, terzo set). Su scambi esclusi:
trovati 77%, giusti 91%, metà campo 98%, giocatore 92%. Colpi di altre partite, con altra
luce e altre posizioni della camera, sono il modo di renderlo più generale.
