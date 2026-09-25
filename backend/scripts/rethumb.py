#!/usr/bin/env python3
"""Choose the players' pictures again, without analysing the match again.

    python scripts/rethumb.py latest

The pictures are chosen after identity, among each player's typical samples
(see app/ml/thumbnails.py), and cut out of the stored video. The players come
from the stored artifacts through the current identity code: they must be the
players of the statistics on screen, or the pictures would go to the wrong
names — the script checks that and stops if not (Rianalizza first).

Reading the video up to the last picture takes a few minutes on the Pi.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import get_settings                      # noqa: E402
from app.core.database import sync_session                     # noqa: E402
from app.core.storage import save_crop, video_path             # noqa: E402
from app.ml.artifacts import load_artifacts, resolve_artifacts_dir  # noqa: E402
from app.ml.identity import resolve_players                    # noqa: E402
from app.ml.player_boxes import matches_stats                 # noqa: E402
from app.ml.thumbnails import choose_pictures, pictures_from_video  # noqa: E402
from app.models import MatchStats                              # noqa: E402


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
        stats = session.get(MatchStats, match_id)
        per_player = dict(stats.per_player) if stats is not None else None
    if per_player is None:
        print("Statistiche non trovate per questa partita.", file=sys.stderr)
        return 1

    config = artifacts.meta.get("config") or {}
    speed = float(config.get("max_player_speed_ms") or settings.max_player_speed_ms)
    identity = resolve_players(artifacts.tracklets, max_speed_ms=speed)
    if not matches_stats(identity.players, per_player):
        print("I giocatori ricalcolati non sono quelli delle statistiche: sono state\n"
              "prodotte da una versione precedente del programma. Usa Rianalizza:\n"
              "l'analisi sceglie già le foto con il nuovo criterio.", file=sys.stderr)
        return 1

    picks = choose_pictures(identity.players)
    if not picks:
        print("Nessun giocatore ha un riquadro utilizzabile per una foto.", file=sys.stderr)
        return 1
    last = max(obs.timestamp_s for obs in picks.values())
    print(f"Partita {match_id[:8]}: rileggo il video fino a {int(last // 60)}:{int(last % 60):02d}…")
    pictures = pictures_from_video(source, picks)

    keys: dict[str, str] = {}
    for player_id, image in pictures.items():
        keys[str(player_id)] = save_crop(match_id, player_id, image)
        at = picks[player_id].timestamp_s
        print(f"  G{player_id + 1}: foto dal minuto {int(at // 60)}:{int(at % 60):02d}")
    with sync_session() as session:
        stats = session.get(MatchStats, match_id)
        stats.player_crops = {**(stats.player_crops or {}), **keys}
    missing = [f"G{p.player_id + 1}" for p in identity.players if p.player_id not in pictures]
    if missing:
        print(f"  nessuna foto utilizzabile per {', '.join(missing)}")
    print("Fatto: ricarica la pagina delle statistiche.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
