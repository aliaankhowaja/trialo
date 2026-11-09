"""FastAPI application entry point.

Educational screening prototype: annotates clinical trial eligibility criteria
with a fine-tuned Transformer and serves the decoding-mechanism benchmark.
NOT a clinical decision system.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import DISCLAIMER
from .routers import benchmark, library, report, screening

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("trailo_app")


@asynccontextmanager
async def lifespan(_: FastAPI):
    from .engine import get_engine

    engine = get_engine()
    logger.info(
        "TrialO screen app: preset=%s base=%s checkpoint=%s device=%s "
        "(model loads lazily on first request)",
        engine.settings.model_config_name,
        engine.settings.base_model_id,
        engine.settings.checkpoint_dir,
        engine.settings.device_name,
    )
    yield


app = FastAPI(
    title="TrialO - Trial Eligibility Screening Prototype",
    description=DISCLAIMER,
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(library.router)
app.include_router(screening.router)
app.include_router(benchmark.router)
app.include_router(report.router)
