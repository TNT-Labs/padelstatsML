#!/usr/bin/env python3
"""The rallies against the points marked in the app ("Segna i punti").

On the Pi, with the points saved in the app:

    docker exec padel-worker python scripts/eval_points.py latest

On a computer, from runs copied off the Pi (folder or .tgz, as for
training/hits) and the text "Copia i punti segnati" gives:

    python backend/scripts/eval_points.py RUN PUNTI [RUN PUNTI ...] [--hits COLPI COLPI ...]

Each match is measured: rallies from the players' movement alone;
corrected by the hits as before the points were read from them; with the
points read from the hits (phase_model.json), the models in use — on the
matches they were trained on the measure is optimistic; and, with --hits
(the hits marked on each match, in the same order) and two matches or
more, with hit models trained on the other matches only. That one needs
scikit-learn and the repository's training/hits. The measure of the
points read on matches never seen is training/points/train_phases.py's.

Mark *every* point of a stretch of the match, one after the other: a rally
between two marked points that is not marked counts as a false rally.
"""
from __future__ import annotations

import argparse
import sys
import tarfile
import tempfile
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

from app.ml.point_eval import Point, PointReport, Span, compare, from_labels, read_points  # noqa: E402


def _clock(t: float) -> str:
    return f"{int(t // 60)}:{t % 60:05.2f}"


def rallies_of(directory: Path, calibration: dict | None, ball: bool, hit_model=None,
               phase_model=None) -> list[Span]:
    from app.core.config import get_settings
    from app.ml.artifacts import load_artifacts
    from app.ml.reprocess import pipeline_config, recompute

    config = pipeline_config(get_settings(), load_artifacts(directory).meta)
    _, _, rallies, _ = recompute(directory, calibration, config, ball=ball, hit_model=hit_model,
                                 phase_model=phase_model)
    return [Span(r.start_s, r.end_s, bool(r.serve)) for r in rallies]


def show(name: str, how: str, report: PointReport, points: list[Point], verbose: bool) -> None:
    s = report.summary()
    st, en = s["start"], s["end"]

    def err(e: dict) -> str:
        if e["median_s"] is None:
            return "—"
        return f"mediana {e['median_s']:+.1f} s, 90% entro {e['p90_abs_s']:.1f} s, oltre 2 s: {e['far_off']}"

    print(f"\n{name} · {how}")
    print(f"  punti segnati {s['points']} · scambi nel tratto {s['rallies']}")
    print(f"  giusti (uno a uno) {s['one_to_one']} · persi {s['lost']} · spezzati {s['split']} · "
          f"uniti ad altri {s['joined']} · scambi falsi {s['false_rallies']}")
    print(f"  inizio: {err(st)}")
    print(f"  fine:   {err(en)}")
    if s["one_to_one"]:
        print(f"  battuta riconosciuta {s['serve_seen']} su {s['one_to_one']}")
    if verbose:
        for point, verdict in zip(points, report.verdicts):
            if verdict != "giusto":
                print(f"    {_clock(point.start_s)}–{_clock(point.end_s)}  {verdict}")
        for span in report.false_spans:
            print(f"    scambio falso {_clock(span.start_s)}–{_clock(span.end_s)}")


def _unpack(path: Path, tmp: Path, n: int) -> Path:
    if path.is_dir():
        return path
    folder = tmp / str(n)
    folder.mkdir()
    with tarfile.open(path) as archive:
        archive.extractall(folder, filter="data")
    return folder


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("args", nargs="+", metavar="PARTITA",
                        help="sul Pi: id della partita, un suo prefisso o 'latest'; "
                             "altrove: coppie di analisi (cartella o .tgz) e punti segnati (testo)")
    parser.add_argument("--hits", nargs="+", metavar="COLPI",
                        help="i colpi segnati di ogni partita, nello stesso ordine: misura anche "
                             "con modelli dei colpi addestrati senza quella partita")
    parser.add_argument("-v", "--verbose", action="store_true", help="elenca i punti sbagliati e gli scambi falsi")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        matches: list[tuple[str, Path, list[Point], dict | None]] = []
        if len(args.args) == 1:
            from app.core.config import get_settings
            from app.core.database import sync_session
            from app.ml.artifacts import resolve_artifacts_dir
            from app.models import Match

            try:
                directory = resolve_artifacts_dir(args.args[0], get_settings().artifacts_path)
            except (FileNotFoundError, ValueError) as exc:
                print(exc, file=sys.stderr)
                return 1
            with sync_session() as session:
                match = session.get(Match, directory.name)
                if match is None:
                    print("Partita non trovata nel database.", file=sys.stderr)
                    return 1
                points, calibration, title = from_labels(match.point_labels), match.calibration, match.title
            matches.append((title, directory, points, dict(calibration) if calibration else None))
        else:
            if len(args.args) % 2:
                print("Servono coppie: analisi e punti segnati, per ogni partita.", file=sys.stderr)
                return 1
            for n in range(0, len(args.args), 2):
                run = _unpack(Path(args.args[n]), Path(tmp), n)
                points = read_points(Path(args.args[n + 1]).read_text(encoding="utf-8"))
                matches.append((Path(args.args[n]).name, run, points, None))
        if any(not points for _, _, points, _ in matches):
            print("Nessun punto segnato per una delle partite: segnali in app con Segna i punti.", file=sys.stderr)
            return 1

        unseen = None
        if args.hits:
            if len(args.hits) != len(matches) or len(matches) < 2:
                print("--hits vuole i colpi segnati di ogni partita, e almeno due partite.", file=sys.stderr)
                return 1
            unseen = _unseen_models(matches, args.hits)

        for k, (name, directory, points, calibration) in enumerate(matches):
            print(f"\n{name}: {len(points)} punti segnati, dal minuto {_clock(points[0].start_s)} "
                  f"al {_clock(points[-1].end_s)}")
            ways = [("movimento dei giocatori", False, None, False),
                    ("con i colpi, senza la lettura dei punti (com'era prima)", True, None, False),
                    ("con i colpi e la lettura dei punti, modelli in uso (ottimista se la partita "
                     "era nell'addestramento)", True, None, None)]
            if unseen is not None:
                ways.append(("con i colpi, modelli dei colpi addestrati senza questa partita, "
                             "senza la lettura dei punti", True, unseen[k], False))
            for how, ball, model, phases in ways:
                spans = rallies_of(directory, calibration, ball, model, phases)
                show(name, how, compare(points, spans), points, args.verbose)
    return 0


def _unseen_models(matches, hits_files):
    """Per match, the hit models trained on the other matches alone."""
    sys.path.insert(0, str(BACKEND.parent / "training" / "hits"))
    try:
        import train_hits as th
    except ImportError as exc:
        raise SystemExit(f"Per --hits servono training/hits del repository e scikit-learn: {exc}") from exc
    datas = [th.Data(directory, th.read_labels(h)) for (_, directory, _, _), h in zip(matches, hits_files)]
    return [th.loaded(th.as_model(*th.fit(datas, lambda i, k, d=d: i != d))) for d in range(len(datas))]


if __name__ == "__main__":
    sys.exit(main())
