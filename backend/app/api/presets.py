"""Camera presets: calibrate a camera position once, reuse it every match.

This is what makes mandatory manual calibration acceptable day to day. The
court does not move and neither does a tripod left in the same corner, so the
30 seconds spent placing four corners is paid once per setup, not per match.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.ml.court import CalibrationError, CalibrationSource, build_calibration
from app.models import CameraPreset
from app.schemas.match import PresetCreate, PresetRead

# Starlette renamed HTTP_422_UNPROCESSABLE_ENTITY; the literal is
# stable across versions.
HTTP_UNPROCESSABLE = 422

router = APIRouter(prefix="/api/presets", tags=["presets"])


@router.get("", response_model=list[PresetRead])
async def list_presets(db: AsyncSession = Depends(get_db)) -> list[CameraPreset]:
    result = await db.execute(
        select(CameraPreset).order_by(CameraPreset.times_used.desc(), CameraPreset.name)
    )
    return list(result.scalars().all())


@router.post("", response_model=PresetRead, status_code=status.HTTP_201_CREATED)
async def create_preset(
    payload: PresetCreate, db: AsyncSession = Depends(get_db)
) -> CameraPreset:
    # Validate before storing: a preset that cannot produce a homography would
    # otherwise fail later, on a different match, with a confusing message.
    try:
        build_calibration(
            corners_px=[tuple(p) for p in payload.corners_px],
            frame_size=(payload.frame_width, payload.frame_height),
            source=CalibrationSource.PRESET,
            net_px=[tuple(p) for p in payload.net_px] if payload.net_px else None,
        )
    except CalibrationError as exc:
        raise HTTPException(HTTP_UNPROCESSABLE, str(exc)) from exc

    existing = await db.scalar(select(CameraPreset).where(CameraPreset.name == payload.name))
    if existing is not None:
        raise HTTPException(
            status.HTTP_409_CONFLICT, f"Esiste già un preset chiamato '{payload.name}'."
        )

    preset = CameraPreset(
        name=payload.name,
        corners_px=[list(p) for p in payload.corners_px],
        frame_width=payload.frame_width,
        frame_height=payload.frame_height,
        net_px=[list(p) for p in payload.net_px] if payload.net_px else None,
    )
    db.add(preset)
    await db.flush()
    return preset


# A 204 route must declare `response_model=None`. These modules use
# `from __future__ import annotations`, so FastAPI resolves the `-> None`
# return annotation through `get_type_hints`, which normalises it to
# `NoneType` — a truthy value. FastAPI then treats it as a response model and
# asserts that a 204 cannot have a body, failing at import time. Saying so
# explicitly is version-proof and keeps the annotation for readers.
@router.delete(
    "/{preset_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_model=None,
)
async def delete_preset(preset_id: str, db: AsyncSession = Depends(get_db)) -> None:
    preset = await db.get(CameraPreset, preset_id)
    if preset is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Preset non trovato.")
    await db.delete(preset)
