"""Report endpoints: Markdown / HTML preview / PDF export of a finished run."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel, Field

from ..report import build_markdown, markdown_to_html, markdown_to_pdf

router = APIRouter(prefix="/api/report", tags=["report"])


class ReportRequest(BaseModel):
    patient_id: str = ""
    trial_title: str = ""
    meta: dict[str, Any]
    results: list[dict[str, Any]]
    sentences: list[str]
    benchmark_snapshot: dict[str, Any] | None = Field(default=None)


@router.post("")
def report_markdown(payload: ReportRequest) -> dict:
    md = build_markdown(
        meta=payload.meta,
        results=payload.results,
        trial_title=payload.trial_title,
        patient_id=payload.patient_id,
        sentences=payload.sentences,
        benchmark_snapshot=payload.benchmark_snapshot,
    )
    return {"markdown": md}


@router.post("/html", response_class=HTMLResponse)
def report_html(payload: ReportRequest) -> HTMLResponse:
    md = build_markdown(
        meta=payload.meta,
        results=payload.results,
        trial_title=payload.trial_title,
        patient_id=payload.patient_id,
        sentences=payload.sentences,
        benchmark_snapshot=payload.benchmark_snapshot,
    )
    return HTMLResponse(markdown_to_html(md))


@router.post("/pdf")
def report_pdf(payload: ReportRequest) -> Response:
    md = build_markdown(
        meta=payload.meta,
        results=payload.results,
        trial_title=payload.trial_title,
        patient_id=payload.patient_id,
        sentences=payload.sentences,
        benchmark_snapshot=payload.benchmark_snapshot,
    )
    try:
        pdf_bytes = markdown_to_pdf(md)
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                f"PDF rendering unavailable in this environment ({exc}). The "
                "weasyprint dependency is bundled in the backend Docker image; "
                "on a dev install use the Markdown/HTML exports, or pass this "
                "report through the browser's print-to-PDF."
            ),
        ) from exc
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": 'inline; filename="eligibility_screening_report.pdf"'},
    )
