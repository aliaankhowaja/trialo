"""Criterion-label and evidence metrics used by every evaluation script."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Iterable


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _prf(true_positive: int, false_positive: int, false_negative: int) -> dict[str, float]:
    precision = _safe_divide(true_positive, true_positive + false_positive)
    recall = _safe_divide(true_positive, true_positive + false_negative)
    f1 = _safe_divide(2 * precision * recall, precision + recall)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": true_positive + false_negative,
    }


def _label_metrics(
    records: list[dict[str, Any]],
    label_universe: Iterable[str],
) -> dict[str, Any]:
    counts: dict[str, Counter] = defaultdict(Counter)
    for record in records:
        gold = record["gold_label"]
        prediction = record.get("prediction")
        predicted = prediction.get("label") if record.get("valid") and prediction else None
        counts[gold]["support"] += 1
        if predicted is not None:
            counts[predicted]["predicted"] += 1
            if predicted == gold:
                counts[gold]["tp"] += 1
            else:
                counts[predicted]["fp"] += 1
                counts[gold]["fn"] += 1

    per_label: dict[str, dict[str, float]] = {}
    for label in sorted(set(label_universe)):
        true_positive = counts[label]["tp"]
        false_positive = counts[label]["fp"]
        false_negative = counts[label]["fn"]
        per_label[label] = _prf(true_positive, false_positive, false_negative)
        per_label[label]["support"] = counts[label]["support"]
        per_label[label]["predicted"] = counts[label]["predicted"]

    f1_values = [value["f1"] for value in per_label.values()]
    return {
        "labels": sorted(per_label),
        "per_label": per_label,
        "macro_f1": sum(f1_values) / len(f1_values) if f1_values else 0.0,
        "micro_support": sum(value["support"] for value in per_label.values()),
    }


def compute_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute parse, label, and evidence metrics from prediction records."""
    total = len(records)
    parsed_records = [
        record
        for record in records
        if record.get("valid") and isinstance(record.get("prediction"), dict)
    ]
    correct = sum(
        1
        for record in parsed_records
        if record["prediction"].get("label") == record["gold_label"]
    )

    gold_labels = {record["gold_label"] for record in records}
    predicted_labels = {
        record["prediction"]["label"]
        for record in parsed_records
        if isinstance(record["prediction"].get("label"), str)
    }
    overall = _label_metrics(records, gold_labels | predicted_labels)

    per_criterion: dict[str, Any] = {}
    for criterion_type in sorted({record["criterion_type"] for record in records}):
        subset = [
            record for record in records if record["criterion_type"] == criterion_type
        ]
        subset_gold = {record["gold_label"] for record in subset}
        subset_predicted = {
            record["prediction"]["label"]
            for record in subset
            if record.get("valid")
            and isinstance(record.get("prediction"), dict)
            and isinstance(record["prediction"].get("label"), str)
        }
        per_criterion[criterion_type] = _label_metrics(
            subset, subset_gold | subset_predicted
        )

    evidence_tp = evidence_fp = evidence_fn = 0
    evidence_exact = 0
    for record in records:
        gold_ids = set(record.get("gold_evidence_ids") or [])
        prediction = record.get("prediction")
        predicted_ids = (
            set(prediction.get("evidence_ids", []))
            if record.get("valid") and isinstance(prediction, dict)
            else set()
        )
        if predicted_ids == gold_ids:
            evidence_exact += 1
        evidence_tp += len(gold_ids & predicted_ids)
        evidence_fp += len(predicted_ids - gold_ids)
        evidence_fn += len(gold_ids - predicted_ids)
    evidence = _prf(evidence_tp, evidence_fp, evidence_fn)
    evidence["exact_match_rate"] = _safe_divide(evidence_exact, total)

    return {
        "n_examples": total,
        "n_parsed": len(parsed_records),
        "parse_success_rate": _safe_divide(len(parsed_records), total),
        "label_accuracy": _safe_divide(correct, total),
        "label_accuracy_parsed": _safe_divide(correct, len(parsed_records)),
        "macro_f1": overall["macro_f1"],
        "per_label": overall["per_label"],
        "label_universe": overall["labels"],
        "per_criterion": per_criterion,
        "evidence": evidence,
        "parse_failures": total - len(parsed_records),
    }


def load_prediction_records(path: str) -> list[dict[str, Any]]:
    """Load JSONL prediction records from disk."""
    import json
    from pathlib import Path

    records = []
    with Path(path).open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
