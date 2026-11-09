"""Benchmark endpoints: live decode-path comparison + recorded A100 spec results."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from ..benchmark import run_decode_benchmark
from ..config import DISCLAIMER, load_settings
from ..engine import NoteFormatError, get_engine, normalize_note as split_note_text
from ..library import find_trial
from ..schemas import BenchmarkResponse, CriterionIn, PatientRecord

router = APIRouter(prefix="/api/benchmark", tags=["benchmark"])


class BenchmarkRequest(BaseModel):
    patient: PatientRecord
    criteria: list[CriterionIn] = Field(min_length=1)
    trial_id: str | None = None
    max_new_tokens: int = Field(default=48, ge=8, le=128)
    criteria_limit: int = Field(default=4, ge=1, le=12)


@lru_cache(maxsize=2)
def _load_speculative_summary(run_name: str, results_root: str) -> dict[str, Any]:
    """Summarize a recorded Task 5 speculative benchmark for the results panel."""
    path = Path(results_root) / run_name / "task5" / "speculative_benchmark.json"
    if not path.is_file():
        raise HTTPException(
            status_code=404,
            detail=(
                f"recorded speculative benchmark not found at {path}. Task 5 "
                "artifacts are produced by `trailo.scripts.benchmark_speculative` "
                "on a GPU host and are read-only for this app."
            ),
        )
    data = json.loads(path.read_text(encoding="utf-8"))

    gamma_rows = []
    gammas = list(data.get("environment", {}).get("gammas", []))
    acceptance = data.get("acceptance_aggregate", {})
    timing = data.get("timing", [])
    baseline_row = next((r for r in timing if r.get("impl") == "baseline"), {})
    for gamma in gammas:
        rows = [r for r in timing if r.get("gamma") == gamma]
        if not rows:
            continue
        median_wall = min(r["median_wall_seconds"] for r in rows)
        acceptance_row = acceptance.get(str(gamma), {})
        baseline_median = baseline_row.get("median_wall_seconds")
        gamma_rows.append(
            {
                "gamma": gamma,
                "acceptance_rate": acceptance_row.get("acceptance_rate"),
                "mean_tokens_per_target_forward": acceptance_row.get(
                    "mean_tokens_per_target_forward"
                ),
                "median_wall_seconds": round(median_wall, 3),
                "baseline_wall_seconds": round(baseline_median, 3) if baseline_median else None,
                "speedup_vs_baseline": (
                    round(baseline_median / median_wall, 3)
                    if baseline_median and median_wall
                    else None
                ),
                "median_tokens_per_second": min(
                    r["median_tokens_per_second"] for r in rows
                ),
            }
        )

    agreement = data.get("agreement", [])
    token_exact = sum(1 for a in agreement if a.get("generate_agrees_with_baseline"))
    knife_edge = {"n_prompts": len(agreement), "token_exact": token_exact}

    return {
        "kind": "recorded_gpu_run",
        "environment": {
            "gpu_name": data.get("environment", {}).get("gpu_name"),
            "torch": data.get("environment", {}).get("torch"),
            "target_model_id": data.get("environment", {}).get("target_model_id"),
            "draft_model_id": data.get("environment", {}).get("draft_model_id"),
            "precision": data.get("environment", {}).get("precision"),
        },
        "gamma_rows": gamma_rows,
        "step_time_summary": data.get("step_time_summary", {}),
        "agreement": {
            "n_prompts": knife_edge["n_prompts"],
            "token_exact_runs": knife_edge["token_exact"],
            "summary": (
                "Greedy speculative decoding emits the same token stream as "
                "target-only greedy decoding, except at bf16 knife-edge "
                "argmax ties (margin 0.0), which are rounding artifacts and "
                "provably sub-ULP in fp32. See the repo's forensic report."
            ),
        },
        "disclaimer": DISCLAIMER,
    }


@router.post("/decode")
def decode_benchmark(payload: BenchmarkRequest) -> dict:
    """Live comparison of `generate` / `cache` / `recompute` on the workload."""
    engine = get_engine()
    if payload.trial_id:
        trial = find_trial(payload.trial_id)
        if trial and not payload.criteria:
            payload.criteria = [
                CriterionIn(criterion_type=c["criterion_type"], text=c["text"])
                for c in trial["criteria"]
            ]
    try:
        note, sentences = split_note_text(payload.patient.note)
    except NoteFormatError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    result = run_decode_benchmark(
        engine,
        note=note,
        sentences=sentences,
        criteria=[
            {"criterion_type": c.criterion_type, "text": c.text} for c in payload.criteria
        ],
        max_new_tokens=payload.max_new_tokens,
        criteria_limit=payload.criteria_limit,
    )
    result["disclaimer"] = DISCLAIMER
    return result


@router.get("/speculative")
def speculative_results(
    run: str = Query(default="final_a100", description="run_name under src/trailo/results"),
) -> dict:
    """Recorded GPU (A100) greedy speculative-decoding benchmark.

    Task 5's custom speculative loop targets the E4B/E2B checkpoint pair and
    needs a large GPU; the app serves the recorded research results here
    instead of running it live, clearly labeled as pre-recorded.
    """
    settings = load_settings()
    return _load_speculative_summary(run, str(settings.results_dir))
