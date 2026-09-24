"""Serve a file in byte ranges, so a browser can seek in a video.

A <video> element asks for `Range: bytes=N-` whenever the viewer jumps
ahead, and needs a 206 reply to do so; without one it can only play from
the start, and some browsers refuse to play at all. Starlette's FileResponse
learnt ranges in 0.39, and FastAPI 0.115 pins an older one — so this module
answers the one form browsers send, a single range, and falls back to the
whole file for anything else, which the HTTP spec allows.
"""
from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

from fastapi import HTTPException, status
from fastapi.responses import FileResponse, StreamingResponse

CHUNK_BYTES = 1024 * 1024

_SINGLE_RANGE = re.compile(r"^\s*bytes\s*=\s*(\d*)\s*-\s*(\d*)\s*$")


def parse_range(header: str | None, size: int) -> tuple[int, int] | None:
    """The inclusive (start, end) a Range header asks for, or None to send
    the whole file. Raises 416 when the range lies past the end."""
    if not header:
        return None
    match = _SINGLE_RANGE.match(header)
    if match is None:
        return None                     # several ranges, or another unit
    first, last = match.groups()
    if not first and not last:
        return None
    if not first:                       # "bytes=-N": the last N bytes
        length = int(last)
        if length == 0:
            raise _unsatisfiable(size)
        return max(size - length, 0), size - 1
    start = int(first)
    end = int(last) if last else size - 1
    if start >= size or end < start:
        raise _unsatisfiable(size)
    return start, min(end, size - 1)


def file_range_response(path: Path, range_header: str | None, media_type: str):
    size = path.stat().st_size
    requested = parse_range(range_header, size)
    if requested is None:
        return FileResponse(str(path), media_type=media_type, headers={"Accept-Ranges": "bytes"})
    start, end = requested
    return StreamingResponse(
        _read(path, start, end),
        status_code=status.HTTP_206_PARTIAL_CONTENT,
        media_type=media_type,
        headers={
            "Accept-Ranges": "bytes",
            "Content-Range": f"bytes {start}-{end}/{size}",
            "Content-Length": str(end - start + 1),
        },
    )


def _read(path: Path, start: int, end: int) -> Iterator[bytes]:
    # A plain generator: Starlette iterates it in a worker thread, so disk
    # reads never block the event loop.
    remaining = end - start + 1
    with open(path, "rb") as handle:
        handle.seek(start)
        while remaining > 0:
            chunk = handle.read(min(CHUNK_BYTES, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
            yield chunk


def _unsatisfiable(size: int) -> HTTPException:
    return HTTPException(
        status.HTTP_416_REQUESTED_RANGE_NOT_SATISFIABLE,
        "Intervallo oltre la fine del file.",
        headers={"Content-Range": f"bytes */{size}"},
    )
