"""Library and health endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from ..config import DISCLAIMER, load_settings
from ..engine import get_engine
from ..library import get_library_payload
from ..schemas import LibraryResponse

router = APIRouter(tags=["library"])


@router.get("/api/health")
def health() -> dict:
    settings = load_settings()
    engine = get_engine()
    payload = engine.status()
    return {"status": "ok", "service_version": "0.1.0", "disclaimer": DISCLAIMER, **payload, "settings_name": settings.model_config_name}


@router.get("/api/disclaimer")
def disclaimer() -> dict:
    return {"disclaimer": DISCLAIMER}


@router.get("/api/library", response_model=LibraryResponse)
def library() -> dict:
    return get_library_payload(DISCLAIMER)
