"""Screening endpoints: note normalization, batch screening, SSE stream."""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from ..config import DISCLAIMER
from ..engine import NoteFormatError, get_engine, normalize_note as split_note_text
from ..library import find_trial
from ..schemas import (
    CriterionResult,
    EvidenceSpan,
    NormalizedNote,
    PatientRecord,
    ScreenRequest,
    ScreenResponse,
    ScreeningMeta,
)

router = APIRouter(tags=["screening"])


def _result_to_schema(result, sentences: list[str]) -> CriterionResult:
    return CriterionResult(
        index=result.index,
        criterion_type=result.criterion_type,
        criterion_text=result.criterion_text,
        label=result.label,
        evidence=[
            EvidenceSpan(sentence_id=sid, text=sentences[sid])
            for sid in result.evidence_ids
            if 0 <= sid < len(sentences)
        ],
        explanation=result.explanation,
        valid=result.valid,
        errors=result.errors,
        warnings=result.warnings,
        raw_text=result.raw_text,
        prompt_tokens=result.prompt_tokens,
        generated_tokens=result.generated_tokens,
        stopped_on_eos=result.stopped_on_eos,
        wall_seconds=round(result.wall_seconds, 3),
        decoding_timing=_redact_timing(result.decoding_timing),
    )


def _redact_timing(timing: dict[str, Any]) -> dict[str, Any]:
    """Keep the summary fields; drop bulky per-token arrays if ever present."""
    return {k: v for k, v in timing.items() if "margin" not in k or isinstance(v, (int, float))}


@router.post("/api/normalize-note", response_model=NormalizedNote)
def normalize(payload: PatientRecord) -> NormalizedNote:
    """Return the note in the numbered 'N. sentence' format the model expects."""
    try:
        note, sentences = split_note_text(payload.note)
    except NoteFormatError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return NormalizedNote(note=note, sentences=sentences, n_sentences=len(sentences))


def _screen_common(payload: ScreenRequest) -> tuple[str, list[str]]:
    if payload.trial_id and not payload.trial_title:
        trial = find_trial(payload.trial_id)
        if trial:
            payload.trial_title = trial["title"]
    try:
        note, sentences = split_note_text(payload.patient.note)
    except NoteFormatError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return note, sentences


def _screen_sync(payload: ScreenRequest) -> ScreenResponse:
    engine = get_engine()
    note, sentences = _screen_common(payload)
    results = []
    for criterion in payload.criteria:
        screened = engine.screen_criterion(
            index=len(results),
            criterion_type=criterion.criterion_type,
            criterion_text=criterion.text,
            note=note,
            sentences=sentences,
            decoding_impl=payload.decoding_impl,
            max_new_tokens=payload.max_new_tokens,
        )
        results.append(_result_to_schema(screened, sentences))
    meta = ScreeningMeta(
        model_preset=engine.settings.model_config_name,
        base_model_id=engine.settings.base_model_id,
        checkpoint_dir=engine.settings.checkpoint_dir,
        device=engine.settings.device_name,
        decoding_impl=payload.decoding_impl,
        n_sentences=len(sentences),
        note=note,
        disclaimer=DISCLAIMER,
    )
    return ScreenResponse(meta=meta, results=results)


@router.post("/api/screen", response_model=ScreenResponse)
def screen(payload: ScreenRequest) -> ScreenResponse:
    """Batch screening: one response with every criterion result."""
    return _screen_sync(payload)


@router.post("/api/screen/stream")
def screen_stream(payload: ScreenRequest) -> StreamingResponse:
    """SSE stream: meta event, one event per criterion, then a done event."""

    def generate():
        engine = get_engine()
        note, sentences = _screen_common(payload)
        meta = ScreeningMeta(
            model_preset=engine.settings.model_config_name,
            base_model_id=engine.settings.base_model_id,
            checkpoint_dir=engine.settings.checkpoint_dir,
            device=engine.settings.device_name,
            decoding_impl=payload.decoding_impl,
            n_sentences=len(sentences),
            note=note,
            disclaimer=DISCLAIMER,
        )
        yield f"data: {json.dumps({'type': 'meta', 'meta': meta.model_dump()})}\n\n"

        results: list[CriterionResult] = []
        for criterion in payload.criteria:
            try:
                screened = engine.screen_criterion(
                    index=len(results),
                    criterion_type=criterion.criterion_type,
                    criterion_text=criterion.text,
                    note=note,
                    sentences=sentences,
                    decoding_impl=payload.decoding_impl,
                    max_new_tokens=payload.max_new_tokens,
                )
            except Exception as exc:  # keep the stream alive, report the failure
                failure = {
                    "index": len(results),
                    "criterion_type": criterion.criterion_type,
                    "criterion_text": criterion.text,
                    "label": None,
                    "evidence": [],
                    "explanation": "",
                    "valid": False,
                    "errors": [f"{type(exc).__name__}: {exc}"],
                    "warnings": [],
                    "raw_text": "",
                    "prompt_tokens": 0,
                    "generated_tokens": 0,
                    "stopped_on_eos": False,
                    "wall_seconds": 0.0,
                    "decoding_timing": {},
                }
                results.append(CriterionResult(**failure))
                yield f"data: {json.dumps({'type': 'result', 'result': failure})}\n\n"
                continue
            result = _result_to_schema(screened, sentences)
            results.append(result)
            yield f"data: {json.dumps({'type': 'result', 'result': result.model_dump()})}\n\n"

        yield f"data: {json.dumps({'type': 'done', 'n_results': len(results)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
