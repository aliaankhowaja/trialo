"""Trial library + sample patient records bundled with the app.

The bundle is a static snapshot of the prepared test split
(`data/processed/smoke_cpu/test.jsonl`, source: TrialGPT-Criterion-Annotations).
Each bundled trial is one (patient record, criterion set) pair; because the
source dataset provides expert reference annotations for exactly these pairs,
each criterion also carries a `gold_label` / `gold_evidence_ids` reference so
users can compare model output against the expert annotation.
"""

from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources
from typing import Any


@lru_cache(maxsize=1)
def load_library() -> dict[str, Any]:
    data_path = resources.files("trailo_app").joinpath("data/library.json")
    return json.loads(data_path.read_text(encoding="utf-8"))


def get_library_payload(disclaimer: str) -> dict[str, Any]:
    lib = load_library()
    return {
        "source": lib["source"],
        "disclaimer": disclaimer,
        "label_sets": lib["label_sets"],
        "trials": [
            {
                "id": t["id"],
                "title": t["title"],
                "n_criteria": t["n_criteria"],
                "n_note_sentences": t["n_note_sentences"],
                "note": t["note"],
                "criteria": [
                    {
                        "index": i,
                        "criterion_type": c["criterion_type"],
                        "text": c["text"],
                        "gold_label": c.get("gold_label"),
                        "gold_evidence_ids": c.get("gold_evidence_ids", []),
                    }
                    for i, c in enumerate(t["criteria"])
                ],
            }
            for t in lib["trials"]
        ],
    }


def find_trial(trial_id: str) -> dict[str, Any] | None:
    for trial in load_library()["trials"]:
        if trial["id"] == trial_id:
            return trial
    return None
