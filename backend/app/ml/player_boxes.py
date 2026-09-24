"""Where each player is in the video, for the web player's overlay.

Numbers can look right while two players are swapped; watching the video
with each player's box and name on it is how a person checks. The boxes
come from the stored artifacts, through the same identity code the
analysis runs, and are compared with the statistics on screen: when they
disagree — the statistics were produced by an older version of the code —
the payload says so rather than label people wrongly in silence.

Computing them takes seconds on the Pi, so the result is cached next to
the artifacts, gzip-compressed as it is sent: about 30 bytes per detection
as JSON, a few hundred kilobytes once compressed for an hour of play.
"""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

from app.ml.artifacts import load_artifacts
from app.ml.identity import IdentityResult, resolve_players
from app.ml.roles import ROLE_NAMES

# Bump when the payload or the way it is computed changes: older cached
# files are then ignored and rebuilt.
CACHE_VERSION = 1
_CACHE_GLOB = "players-*.json.gz"


class BoxesUnavailable(Exception):
    """No boxes for this match, with the reason to show the viewer."""


def player_boxes_gz(directory: Path, stats_per_player: dict | None, default_speed_ms: float) -> bytes:
    """The overlay payload as gzip-compressed JSON, from the cache when
    it is still valid."""
    tracks = directory / "tracks.jsonl"
    if not tracks.exists():
        raise BoxesUnavailable(
            "Riquadri non disponibili: l'analisi non ha salvato le rilevazioni (KEEP_ARTIFACTS)."
        )
    cache = _cache_path(directory, stats_per_player)
    if cache.exists():
        return cache.read_bytes()
    return _store(directory, build_payload(directory, stats_per_player, default_speed_ms), stats_per_player)


def save_player_boxes(
    directory: Path, identity: IdentityResult, sample_hz: float, frame: dict, stats_per_player: dict,
) -> None:
    """Write the cache at the end of an analysis, from the identity it has
    just computed: the web player then opens at once, with no replay."""
    _store(directory, payload_from(identity, sample_hz, frame, stats_per_player), stats_per_player)


def build_payload(directory: Path, stats_per_player: dict | None, default_speed_ms: float) -> dict:
    """Recompute the players from the stored artifacts — for matches
    analysed before the cache was written by the analysis itself."""
    artifacts = load_artifacts(directory)
    if not artifacts.complete:
        raise BoxesUnavailable("Riquadri non disponibili: l'analisi è ancora in corso.")
    config = artifacts.meta.get("config") or {}
    max_speed = config.get("max_player_speed_ms") or default_speed_ms
    identity = resolve_players(artifacts.tracklets, max_speed_ms=float(max_speed))
    video = artifacts.meta.get("video") or {}
    frame = {"width": video.get("width"), "height": video.get("height")}
    return payload_from(identity, artifacts.sample_hz, frame, stats_per_player)


def payload_from(identity: IdentityResult, sample_hz: float, frame: dict, stats_per_player: dict | None) -> dict:
    players = []
    for player in identity.players:
        times, boxes = [], []
        for obs in player.observations:
            times.append(round(float(obs.timestamp_s), 2))
            boxes.extend(int(round(v)) for v in obs.bbox)
        players.append({
            "id": player.player_id,
            "team": player.team,
            "role": ROLE_NAMES.get(player.role),
            "t": times,
            "box": boxes,
        })
    return {
        "sample_hz": float(sample_hz),
        "frame": frame,
        "players": players,
        "changeovers": [round(float(t), 1) for t in identity.changeovers],
        "matches_stats": _matches_stats(identity.players, stats_per_player),
    }


def _store(directory: Path, payload: dict, stats_per_player: dict | None) -> bytes:
    data = gzip.compress(json.dumps(payload, separators=(",", ":")).encode("utf-8"), compresslevel=6)
    cache = _cache_path(directory, stats_per_player)
    for stale in directory.glob(_CACHE_GLOB):
        stale.unlink(missing_ok=True)
    tmp = cache.with_suffix(".part")
    tmp.write_bytes(data)
    tmp.replace(cache)
    return data


def _matches_stats(players, stats_per_player: dict | None) -> bool:
    """Whether these are the players the statistics describe: the same ids,
    each with the same number of detections."""
    if not stats_per_player:
        return False
    ours = {str(p.player_id): len(p.observations) for p in players}
    theirs = {str(k): (v or {}).get("samples") for k, v in stats_per_player.items()}
    return ours == theirs


def _cache_path(directory: Path, stats_per_player: dict | None) -> Path:
    tracks = (directory / "tracks.jsonl").stat()
    samples = sorted((str(k), (v or {}).get("samples")) for k, v in (stats_per_player or {}).items())
    raw = json.dumps([CACHE_VERSION, tracks.st_size, tracks.st_mtime_ns, samples])
    return directory / f"players-{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:16]}.json.gz"
