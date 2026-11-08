"""Run greedy inference over prepared examples and write metrics."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trailo.eval.inference import generate_prediction
from trailo.eval.metrics import compute_metrics


def run_records(
    model,
    tokenizer,
    records: list[dict[str, Any]],
    decoding_config: dict[str, Any],
    *,
    output_path: str | Path,
    metrics_path: str | Path,
    metadata: dict[str, Any] | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Generate predictions for records, saving raw/parsed outputs and metrics."""
    selected = records[:limit] if limit is not None else records
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    predictions: list[dict[str, Any]] = []

    with output.open("w") as handle:
        for index, example in enumerate(selected, start=1):
            result = generate_prediction(model, tokenizer, example, decoding_config)
            payload = {
                "annotation_id": example["annotation_id"],
                "patient_id": example["patient_id"],
                "criterion_type": example["criterion_type"],
                "gold_label": example["gold_label"],
                "gold_evidence_ids": example["gold_evidence_ids"],
                "n_sentences": example["n_sentences"],
                **result,
            }
            predictions.append(payload)
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            if index % 10 == 0 or index == len(selected):
                print(f"  generated {index}/{len(selected)}")

    metrics = compute_metrics(predictions)
    metrics.update(
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "decoding": {
                "max_new_tokens": int(decoding_config["max_new_tokens"]),
                "do_sample": False,
                "num_beams": 1,
            },
            "output_path": str(output),
        }
    )
    if metadata:
        metrics.update(metadata)
    metrics_output = Path(metrics_path)
    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    metrics_output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    return metrics
