"""Pydantic request/response schemas for the screening API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

CriterionType = Literal["inclusion", "exclusion"]
DecodingImpl = Literal["generate", "cache", "recompute"]


class PatientRecord(BaseModel):
    patient_id: str = Field(default="", description="Free-form identifier; never used in inference.")
    note: str = Field(description="Patient note. Plain text or pre-numbered '0. ...' lines.")


class CriterionIn(BaseModel):
    criterion_type: CriterionType
    text: str = Field(min_length=1)


class NormalizedNote(BaseModel):
    note: str = Field(description="Numbered note exactly as fed to the model.")
    sentences: list[str]
    n_sentences: int


class ScreenRequest(BaseModel):
    patient: PatientRecord
    criteria: list[CriterionIn] = Field(min_length=1)
    trial_id: str | None = None
    trial_title: str | None = None
    decoding_impl: DecodingImpl = "generate"
    max_new_tokens: int = Field(default=128, ge=8, le=256)


class EvidenceSpan(BaseModel):
    sentence_id: int
    text: str


class CriterionResult(BaseModel):
    index: int
    criterion_type: CriterionType
    criterion_text: str
    label: str | None
    evidence: list[EvidenceSpan]
    explanation: str
    valid: bool
    errors: list[str]
    warnings: list[str]
    raw_text: str
    prompt_tokens: int
    generated_tokens: int
    stopped_on_eos: bool
    wall_seconds: float
    decoding_timing: dict[str, Any]


class ScreeningMeta(BaseModel):
    model_preset: str
    base_model_id: str
    checkpoint_dir: str
    device: str
    decoding_impl: str
    n_sentences: int
    note: str
    disclaimer: str


class ScreenResponse(BaseModel):
    meta: ScreeningMeta
    results: list[CriterionResult]


class LibraryCriterion(BaseModel):
    index: int
    criterion_type: CriterionType
    text: str
    gold_label: str | None = None
    gold_evidence_ids: list[int] = Field(default_factory=list)


class LibraryTrial(BaseModel):
    id: str
    title: str
    n_criteria: int
    n_note_sentences: int
    note: str
    criteria: list[LibraryCriterion]


class LibraryResponse(BaseModel):
    source: str
    disclaimer: str
    label_sets: dict[str, list[str]]
    trials: list[LibraryTrial]


class BenchmarkImplRow(BaseModel):
    impl: str
    wall_seconds: float
    generated_tokens: int
    tokens_per_second: float
    peak_memory_delta_bytes: float | None
    per_criterion_seconds: list[float]
    ok: bool
    error: str | None = None


class BenchmarkAgreementEntry(BaseModel):
    index: int
    all_equal: bool
    details: dict[str, Any]


class BenchmarkResponse(BaseModel):
    meta: dict[str, Any]
    rows: list[BenchmarkImplRow]
    agreement: list[BenchmarkAgreementEntry]
    disclaimer: str
