"""Byte ranges: how a browser seeks in a video."""
from __future__ import annotations

import pytest
from fastapi import HTTPException

from app.api.ranges import parse_range


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        (None, None),
        ("", None),
        ("bytes=0-", (0, 999)),                 # what a <video> asks for first
        ("bytes=100-199", (100, 199)),
        ("bytes=900-5000", (900, 999)),         # an end past the file is clipped
        ("bytes=-100", (900, 999)),             # the last 100 bytes
        ("bytes=-5000", (0, 999)),
        ("bytes=0-1,5-6", None),                # several ranges: send the whole file
        ("items=0-1", None),
    ],
)
def test_a_single_range_is_understood(header, expected):
    assert parse_range(header, 1000) == expected


@pytest.mark.parametrize("header", ["bytes=1000-", "bytes=500-100", "bytes=-0"])
def test_a_range_past_the_end_is_refused_with_the_size(header):
    with pytest.raises(HTTPException) as refused:
        parse_range(header, 1000)
    assert refused.value.status_code == 416
    assert refused.value.headers["Content-Range"] == "bytes */1000"
