#!/usr/bin/env python3
"""Recompute a match's statistics and pictures from its stored detections.

    python scripts/refresh.py latest

The same as Aggiorna statistiche in the app (POST /api/matches/{id}/refresh).

A fix to identity, rallies or metrics (an object mistaken for a player, a
new warning) reaches an analysed match only through Rianalizza, which runs
the detector over the whole video again: about as long as the match. The
detections it would find are the ones already stored in the artifacts, so
this runs only the stages after them — the same code as the analysis —
saves the statistics in place of the old ones and cuts the players'
pictures again out of the video. Seconds, plus reading the video up to the
last picture.

Player names are kept when every player keeps their pair and side (the
ids are the same people); otherwise they are cleared, as Rianalizza does,
and the app asks for them again.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import get_settings                          # noqa: E402
from app.core.database import sync_session                         # noqa: E402
from app.core.storage import save_crop, video_path                 # noqa: E402
from app.ml.artifacts import load_artifacts, resolve_artifacts_dir  # noqa: E402
from app.ml.reprocess import pipeline_config, refresh, same_slots  # noqa: E402
from app.ml.thumbnails import choose_pictures, pictures_from_video  # noqa: E402
from app.models import Match, MatchStats, MatchStatus              # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("match_id", help="id della partita, un suo prefisso, oppure 'latest'")
    args = parser.parse_args()
    settings = get_settings()

    try:
        directory = resolve_artifacts_dir(args.match_id, settings.artifacts_path)
    except (FileNotFoundError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1
    match_id = directory.name
    artifacts = load_artifacts(directory)
    if not artifacts.complete:
        print("Questa analisi non è conclusa: riprova quando è finita.", file=sys.stderr)
        return 1
    source = video_path(match_id)
    if not source.exists():
        print(f"Video non trovato: {source}", file=sys.stderr)
        return 1

    with sync_session() as session:
        match = session.get(Match, match_id)
        if match is None:
            print("Partita non trovata nel database.", file=sys.stderr)
            return 1
        if match.status != MatchStatus.COMPLETED:
            print(f"La partita è nello stato '{match.status.value}': si aggiorna solo una "
                  "partita con l'analisi completata.", file=sys.stderr)
            return 1
        old = session.get(MatchStats, match_id)
        old_players = dict(old.per_player) if old is not None else {}
        old_quality = dict(old.data_quality or {}) if old is not None else {}
        calibration_data = match.calibration

    # Through the match's calibration, which may have been corrected after
    # the analysis (Verifica accuratezza).
    result, identity, rallies, artifacts = refresh(
        directory, calibration_data, pipeline_config(settings, artifacts.meta), old_quality)
    threshold = result["data_quality"].get("rally_speed_threshold_ms")
    print(f"Partita {match_id[:8]}: {len(identity.players)} giocatori · {len(rallies)} scambi "
          f"(in gioco sopra {threshold} m/s)")
    for warning in result["data_quality"]["warnings"]:
        print(f"  ! {warning}")

    picks = choose_pictures(identity.players)
    pictures = {}
    if picks:
        last = max(obs.timestamp_s for obs in picks.values())
        print(f"Rileggo il video fino a {int(last // 60)}:{int(last % 60):02d} per le foto…")
        pictures = pictures_from_video(source, picks)

    crops = {str(pid): save_crop(match_id, pid, image) for pid, image in pictures.items()}
    same_people = same_slots(old_players, result["per_player"])
    with sync_session() as session:
        session.query(MatchStats).filter(MatchStats.match_id == match_id).delete()
        session.add(MatchStats(
            match_id=match_id,
            per_player=result["per_player"],
            heatmaps=result["heatmaps"],
            rallies=result["rallies"],
            summary=result["summary"],
            data_quality=result["data_quality"],
            player_crops=crops or None,
        ))
        match = session.get(Match, match_id)
        if not same_people and match.player_names:
            match.player_names = None

    for player_id, stats in sorted(result["per_player"].items(), key=lambda kv: int(kv[0])):
        print(f"  G{int(player_id) + 1}: tracciato {stats['tracked_ratio'] * 100:.0f}%"
              f"{'' if int(player_id) in pictures else ' · nessuna foto utilizzabile'}")
    print("Nomi dei giocatori mantenuti." if same_people
          else "Giocatori cambiati: i nomi vanno reinseriti (te li chiede l'app).")
    print("Fatto: ricarica la pagina delle statistiche.")
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
