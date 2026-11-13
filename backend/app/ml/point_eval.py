"""The rallies found, against the points a person marked in the video.

"Segna i punti" in the app: for each point, the serve (start), the moment it
is decided (end) and the pair that won it. Within the stretch of the match
where points were marked — all of them, one after the other — every point
should be one rally and every rally one point. What goes wrong is counted
as such:

- a point lost: no rally overlaps it;
- a point split: two or more rallies overlap it;
- points joined: one rally overlaps two or more of them;
- a false rally: it overlaps no point (players walking, a ball passed back);
- for each point matched one to one, how far the rally's start and end are
  from the serve and the end marked, and whether the rally says it was
  opened by a serve.

Overlap is counted with OVERLAP_TOL_S of slack on either side: the marks
are made watching the video, a few tenths of a second off.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

OVERLAP_TOL_S = 0.5
# Rallies this far outside the first and last points marked are not judged:
# nobody said what happened there.
MARGIN_S = 5.0
# A start or end further off than this is counted as wrong.
FAR_OFF_S = 2.0


@dataclass(frozen=True)
class Point:
    start_s: float
    end_s: float
    winners: tuple[int, int] | None = None


@dataclass(frozen=True)
class Span:
    """A rally as found: from the analysis or recomputed."""
    start_s: float
    end_s: float
    serve: bool = False


@dataclass
class PointReport:
    points: int = 0
    rallies: int = 0
    one_to_one: int = 0
    lost: int = 0
    split: int = 0
    joined: int = 0
    false_rallies: int = 0
    serve_seen: int = 0
    start_errors: list[float] = field(default_factory=list)
    end_errors: list[float] = field(default_factory=list)
    # Per point: "giusto", "perso", "spezzato", "unito", "spezzato e unito".
    verdicts: list[str] = field(default_factory=list)
    false_spans: list[Span] = field(default_factory=list)

    def summary(self) -> dict:
        def stats(errors: list[float]) -> dict:
            if not errors:
                return {"median_s": None, "p90_abs_s": None, "far_off": 0}
            ordered = sorted(abs(e) for e in errors)
            return {"median_s": round(_median(errors), 2),
                    "p90_abs_s": round(ordered[min(len(ordered) - 1, math.ceil(0.9 * len(ordered)) - 1)], 2),
                    "far_off": sum(e > FAR_OFF_S for e in ordered)}
        return {"points": self.points, "rallies": self.rallies, "one_to_one": self.one_to_one,
                "lost": self.lost, "split": self.split, "joined": self.joined,
                "false_rallies": self.false_rallies, "serve_seen": self.serve_seen,
                "start": stats(self.start_errors), "end": stats(self.end_errors)}


def _median(values: list[float]) -> float:
    s = sorted(values)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float, tol: float = OVERLAP_TOL_S) -> bool:
    return a_start <= b_end + tol and a_end >= b_start - tol


def compare(points: list[Point], rallies: list[Span]) -> PointReport:
    """The rallies against the marked points (see the module's docstring)."""
    report = PointReport()
    points = sorted(points, key=lambda p: p.start_s)
    if not points:
        return report
    lo, hi = points[0].start_s - MARGIN_S, points[-1].end_s + MARGIN_S
    rallies = sorted((r for r in rallies if _overlap(r.start_s, r.end_s, lo, hi, 0.0)), key=lambda r: r.start_s)
    report.points, report.rallies = len(points), len(rallies)
    on_point = [[j for j, p in enumerate(points) if _overlap(r.start_s, r.end_s, p.start_s, p.end_s)]
                for r in rallies]
    for r, touched in zip(rallies, on_point):
        if not touched:
            report.false_rallies += 1
            report.false_spans.append(r)
    for j, point in enumerate(points):
        mine = [i for i, touched in enumerate(on_point) if j in touched]
        joined = any(len(on_point[i]) > 1 for i in mine)
        if not mine:
            verdict = "perso"
            report.lost += 1
        elif len(mine) > 1:
            verdict = "spezzato e unito" if joined else "spezzato"
            report.split += 1
            report.joined += joined
        elif joined:
            verdict = "unito"
            report.joined += 1
        else:
            verdict = "giusto"
            rally = rallies[mine[0]]
            report.one_to_one += 1
            report.serve_seen += rally.serve
            report.start_errors.append(rally.start_s - point.start_s)
            report.end_errors.append(rally.end_s - point.end_s)
        report.verdicts.append(verdict)
    return report


# ── The text "Copia i punti segnati" gives ──────────────────────────────────

_TIME = re.compile(r"^(\d+):(\d+(?:\.\d+)?)$")


def _seconds(text: str) -> float | None:
    m = _TIME.match(text)
    if m:
        return int(m.group(1)) * 60 + float(m.group(2))
    try:
        return float(text)
    except ValueError:
        return None


def read_points(text: str) -> list[Point]:
    """Points from the copied text: a line per point, tab separated, start
    and end (minutes:seconds), then the winners' names and their ids
    ("1,3"), or "-" when not said. Other lines (the title) are skipped."""
    out = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split("\t") if p.strip()]
        if len(parts) < 2:
            continue
        start, end = _seconds(parts[0]), _seconds(parts[1])
        if start is None or end is None or end <= start:
            continue
        winners = None
        ids = re.fullmatch(r"(\d)\s*,\s*(\d)", parts[-1]) if len(parts) > 2 else None
        if ids:
            winners = tuple(sorted((int(ids.group(1)), int(ids.group(2)))))
        out.append(Point(start, end, winners))
    return sorted(out, key=lambda p: p.start_s)


def from_labels(labels: list | None) -> list[Point]:
    """Points from Match.point_labels: [[start, end, winners or None]]."""
    return sorted((Point(float(a), float(b), tuple(sorted(int(p) for p in w)) if w else None)
                   for a, b, w in labels or []), key=lambda p: p.start_s)
