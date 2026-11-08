"""Criterion-level and evidence analysis used by the Task 2 report.

Everything here is pure analysis of prediction records and their matching
prepared examples. No model is loaded and no inference is run.

The analysis preserves the full four-way label set per criterion type
(`included`/`not applicable`/`not enough information`/`not included` for
inclusion and `excluded`/`not applicable`/`not enough information`/
`not excluded` for exclusion). Nothing is collapsed into a binary decision.

Absent-evidence policy: an empty predicted evidence list on a record with no
gold evidence is scored as `no_gold_evidence` consistency, never as proof of
eligibility or ineligibility, and records with no annotated gold evidence
cannot contribute recall credit under any view.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any, Iterable

from trailo.eval.metrics import _prf, _safe_divide

INCLUSION_LABELS = [
    "included",
    "not applicable",
    "not enough information",
    "not included",
]
EXCLUSION_LABELS = [
    "excluded",
    "not applicable",
    "not enough information",
    "not excluded",
]

#: Per criterion type, the label meaning "patient is eligible for the trial".
ELIGIBLE_LABELS = {"inclusion": "included", "exclusion": "not excluded"}
#: Per criterion type, the label meaning "patient is not eligible".
INELIGIBLE_LABELS = {"inclusion": "not included", "exclusion": "excluded"}
NEUTRAL_LABELS = ("not applicable", "not enough information")

#: Sentence text patterns that contradict an eligibility assertion. These are
#: review heuristics, not judgments; eligible predictions citing such sentences
#: are flagged for manual inspection, never auto-downgraded.
NEGATION_CUES = re.compile(
    r"\b(no|not|none|neither|nor|never|denies|denied|denial|absent|absence|"
    r"negative|without|unremarkable|lacking|lacks|lacked|unlikely|"
    r"ruled out|free of|clear of)\b",
    re.IGNORECASE,
)

_SANE_SENTENCE_ID = re.compile(r"^\s*(\d+)\.\s")

EVIDENCE_POLICY = (
    "Evidence is scored in three views. (1) Pooled: micro-aggregated "
    "true/false-positive/negative counts over every parsed record, identical "
    "to the headline metric in eval/metrics.py. Records with no annotated "
    "gold evidence cannot add recall and any cited sentence on such a record "
    "counts as a false positive. (2) Gold-present: the same scoring "
    "restricted to records whose gold evidence set is non-empty; the primary "
    "comparable retrieval number. (3) Per-record: every gold-present record "
    "contributes one P/R/F1 triple and the macro average is reported. "
    "Records with empty gold evidence are reported separately as a "
    "consistency rate (predicted evidence also empty); they never earn "
    "credit toward evidence quality. Empty predicted evidence is always "
    "measured, never treated as proof of eligibility or ineligibility, "
    "by the analysis or the report."
)


def predicted_label(record: dict[str, Any]) -> str | None:
    """Return the predicted label for a parsed record, else ``None``."""
    prediction = record.get("prediction")
    if record.get("valid") and isinstance(prediction, dict):
        label = prediction.get("label")
        if isinstance(label, str):
            return label
    return None


def predicted_evidence_ids(record: dict[str, Any]) -> list[int]:
    """Return the predicted evidence IDs for a parsed record, else ``[]``."""
    prediction = record.get("prediction")
    if record.get("valid") and isinstance(prediction, dict):
        ids = prediction.get("evidence_ids", [])
        if isinstance(ids, list) and all(isinstance(i, int) for i in ids):
            return ids
    return []


def parse_sentences(note: str) -> dict[int, str]:
    """Map sentence IDs to sentence text from a numbered note."""
    sentences: dict[int, str] = {}
    for line in note.splitlines():
        match = _SANE_SENTENCE_ID.match(line)
        if match:
            sentences[int(match.group(1))] = line[match.end() :].strip()
    return sentences


def extract_cited_evidence(
    note: str,
    evidence_ids: Iterable[int],
    *,
    n_sentences: int,
) -> list[dict[str, Any]]:
    """Resolve cited sentence IDs to text in citation order."""
    sentences = parse_sentences(note)
    extracted: list[dict[str, Any]] = []
    for sentence_id in evidence_ids:
        if sentence_id in sentences and 0 <= sentence_id < n_sentences:
            extracted.append(
                {"id": sentence_id, "text": sentences[sentence_id], "valid": True}
            )
        else:
            extracted.append({"id": sentence_id, "text": None, "valid": False})
    return extracted


def _label_universe(
    records: list[dict[str, Any]], label_sets: dict[str, list[str]]
) -> list[str]:
    """Universe of labels for a group of records, always sorted and unique.

    Mirrors eval/metrics.py: the declared label set extended with any observed
    gold or predicted label so no confusion count is dropped.
    """
    labels = set(
        label for values in label_sets.values() for label in values
    )
    for record in records:
        labels.add(record["gold_label"])
        predicted = predicted_label(record)
        if predicted is not None:
            labels.add(predicted)
    return sorted(labels)


def confusion_matrix(
    records: list[dict[str, Any]],
    labels: list[str],
) -> dict[str, Any]:
    """Counts matrix with gold rows and predicted columns, plus per-class metrics."""
    index = {label: position for position, label in enumerate(labels)}
    counts = [[0 for _ in labels] for _ in labels]
    per_class: dict[str, dict[str, float]] = {}
    unparsed = 0
    for record in records:
        gold = record["gold_label"]
        predicted = predicted_label(record)
        if predicted is None:
            unparsed += 1
            continue
        counts[index[gold]][index[predicted]] += 1
    for label in labels:
        i = index[label]
        tp = counts[i][i]
        fp = sum(counts[r][i] for r in range(len(labels))) - tp
        fn = sum(counts[i]) - tp
        per_class[label] = _prf(tp, fp, fn)
        per_class[label]["support"] = sum(counts[i])
        per_class[label]["predicted"] = sum(counts[r][i] for r in range(len(labels)))
    row_normalized = [
        [
            _safe_divide(counts[r][c], sum(counts[r]))
            for c in range(len(labels))
        ]
        for r in range(len(labels))
    ]
    total = len(records)
    correct = sum(counts[i][i] for i in range(len(labels)))
    parsed = total - unparsed
    f1_values = [value["f1"] for value in per_class.values()]
    # Two macro-F1 conventions, both reported:
    # - macro_f1: over the full declared label universe. Never-predicted
    #   classes (F1=0) are included, so a model that never outputs `excluded`
    #   cannot hide that failure.
    # - macro_f1_observed: over labels with gold support or predictions only,
    #   matching eval/metrics.py so headline numbers stay comparable.
    observed_labels = [
        label
        for label in labels
        if per_class[label]["support"] or per_class[label]["predicted"]
    ]
    macro_f1_observed = (
        sum(per_class[label]["f1"] for label in observed_labels)
        / len(observed_labels)
        if observed_labels
        else 0.0
    )
    return {
        "labels": labels,
        "counts": counts,
        "row_normalized": row_normalized,
        "per_class": per_class,
        "accuracy": _safe_divide(correct, total),
        "accuracy_parsed": _safe_divide(correct, parsed),
        "macro_f1": (
            sum(f1_values) / len(f1_values) if f1_values else 0.0
        ),
        "macro_f1_observed": macro_f1_observed,
        "n_records": total,
        "n_unparsed": unparsed,
    }


def classification_views(
    records: list[dict[str, Any]], label_sets: dict[str, list[str]]
) -> dict[str, Any]:
    """Overall confusion matrix plus one per inclusion/exclusion criterion type."""
    views: dict[str, Any] = {}
    views["overall"] = confusion_matrix(records, _label_universe(records, label_sets))
    for criterion_type in ("inclusion", "exclusion"):
        subset = [r for r in records if r["criterion_type"] == criterion_type]
        universe = list(label_sets[criterion_type])
        observed = {r["gold_label"] for r in subset} | {
            label
            for label in (predicted_label(r) for r in subset)
            if label is not None
        }
        universe = universe + sorted(set(universe) ^ observed)
        views[criterion_type] = confusion_matrix(subset, sorted(set(universe)))
    return views


def evidence_views(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Pooled, gold-present, and per-record evidence retrieval metrics."""
    gold_present = [r for r in records if r.get("gold_evidence_ids")]
    no_gold = [r for r in records if not r.get("gold_evidence_ids")]

    pooled = _aggregate_evidence(records)
    pooled["n_records"] = len(records)
    pooled["exact_match_rate"] = _safe_divide(
        sum(1 for r in records if _evidence_ids(r) == set(r.get("gold_evidence_ids") or [])),
        len(records),
    )

    gold_metrics = _aggregate_evidence(gold_present)
    gold_metrics["n_records"] = len(gold_present)
    gold_metrics["exact_match_rate"] = _safe_divide(
        sum(1 for r in gold_present if _evidence_ids(r) == set(r.get("gold_evidence_ids") or [])),
        len(gold_present),
    )

    per_record_f1 = []
    for record in gold_present:
        gold_ids = set(record.get("gold_evidence_ids") or [])
        predicted = _evidence_ids(record)
        scores = _prf(len(gold_ids & predicted), len(predicted - gold_ids), len(gold_ids - predicted))
        per_record_f1.append(scores["f1"])
    per_record = {
        "n_records": len(gold_present),
        "macro_f1": (
            sum(per_record_f1) / len(per_record_f1) if per_record_f1 else 0.0
        ),
    }

    consistency = {
        "n_records": len(no_gold),
        "empty_prediction_rate": _safe_divide(
            sum(1 for r in no_gold if not _evidence_ids(r)), len(no_gold)
        ),
        "cited_without_gold_rate": _safe_divide(
            sum(1 for r in no_gold if _evidence_ids(r)), len(no_gold)
        ),
    }
    return {
        "policy": EVIDENCE_POLICY,
        "pooled": pooled,
        "gold_present": gold_metrics,
        "per_record": per_record,
        "no_gold_evidence": consistency,
    }


def _aggregate_evidence(records: list[dict[str, Any]]) -> dict[str, float]:
    tp = fp = fn = 0
    for record in records:
        gold_ids = set(record.get("gold_evidence_ids") or [])
        predicted = _evidence_ids(record)
        tp += len(gold_ids & predicted)
        fp += len(predicted - gold_ids)
        fn += len(gold_ids - predicted)
    result = _prf(tp, fp, fn)
    result.pop("support", None)
    return result


def _evidence_ids(record: dict[str, Any]) -> set[int]:
    ids = predicted_evidence_ids(record)
    if record.get("valid"):
        return set(ids)
    return set()


def flag_contradictions(
    record: dict[str, Any],
    *, 
    note: str,
) -> list[dict[str, Any]]:
    """Heuristic review flags on one prediction. Ranked by severity later."""
    flags: list[dict[str, Any]] = []
    label = predicted_label(record)
    if label is None:
        return flags
    ids = predicted_evidence_ids(record)
    criterion_type = record["criterion_type"]
    eligible = ELIGIBLE_LABELS[criterion_type]
    ineligible = INELIGIBLE_LABELS[criterion_type]
    n_sentences = int(record.get("n_sentences", 0))

    if not ids and label in (eligible, ineligible):
        is_eligible = label == eligible
        flags.append(
            {
                "flag": f"{label}_without_evidence",
                "severity": 3 if is_eligible else 2,
                "detail": (
                    "Conclusive eligibility prediction with no cited evidence; "
                    "absent evidence is not proof either way."
                ),
            }
        )
    if ids:
        cited = extract_cited_evidence(note, ids, n_sentences=n_sentences)
        cited_text = " ".join(item["text"] or "" for item in cited if item["valid"])
        if label == eligible and NEGATION_CUES.search(cited_text):
            flags.append(
                {
                    "flag": "eligible_with_negated_evidence",
                    "severity": 2,
                    "detail": "Eligibility predicted while cited sentences contain negation cues.",
                }
            )
        if len(set(ids)) != len(ids):
            flags.append(
                {
                    "flag": "duplicate_evidence_ids",
                    "severity": 1,
                    "detail": "A sentence ID appears more than once.",
                }
            )
        for item in cited:
            if not item["valid"]:
                flags.append(
                    {
                        "flag": "evidence_id_out_of_range",
                        "severity": 2,
                        "detail": f"Sentence ID {item['id']} is outside 0..{n_sentences - 1}.",
                    }
                )
    return flags


def flag_counts(
    records: list[dict[str, Any]],
    notes: dict[Any, str],
    criteria: dict[Any, str] | None = None,
) -> dict[str, Any]:
    """Aggregate contradiction flags and the conclusive-without-evidence rate."""
    counts: Counter[str] = Counter()
    conclusive_any = conclusive_none = 0
    flagged: list[dict[str, Any]] = []
    criteria = criteria or {}
    for record in records:
        note = notes.get(_record_key(record), "")
        for flag in flag_contradictions(record, note=note):
            counts[flag["flag"]] += 1
            if flag["severity"] >= 2:
                entry = _record_entry(record)
                entry["criterion"] = criteria.get(_record_key(record))
                entry.update(
                    {"flag": flag["flag"], "severity": flag["severity"], "detail": flag["detail"]}
                )
                flagged.append(entry)
        label = predicted_label(record)
        if label in (ELIGIBLE_LABELS[record["criterion_type"]], INELIGIBLE_LABELS[record["criterion_type"]]):
            conclusive_any += 1
            if not predicted_evidence_ids(record):
                conclusive_none += 1
    flagged.sort(key=lambda item: (-item["severity"], item["patient_id"], item["annotation_id"]))
    return {
        "flag_counts": dict(counts),
        "conclusive_predictions": conclusive_any,
        "conclusive_without_evidence": conclusive_none,
        "conclusive_without_evidence_rate": _safe_divide(conclusive_none, conclusive_any),
        "flagged": flagged,
    }


def _record_key(record: dict[str, Any]) -> tuple[Any, ...]:
    return (record.get("patient_id"), record.get("annotation_id"))


def _record_entry(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "patient_id": record.get("patient_id"),
        "annotation_id": record.get("annotation_id"),
        "criterion_type": record.get("criterion_type"),
        "gold_label": record.get("gold_label"),
        "predicted_label": predicted_label(record),
        "predicted_evidence_ids": predicted_evidence_ids(record),
        "explanation": (record.get("prediction") or {}).get("explanation", "")
        if isinstance(record.get("prediction"), dict)
        else "",
    }


def confusion_pair_examples(
    records: list[dict[str, Any]],
    note_texts: dict[Any, str],
    *,
    criteria: dict[Any, str] | None = None,
    top_pairs: int,
    examples_per_pair: int,
) -> list[dict[str, Any]]:
    """Dominant gold-to-predicted confusion pairs with stock example records."""
    pairs: Counter[tuple[str, str]] = Counter()
    by_pair: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        predicted = predicted_label(record)
        if predicted is None or predicted == record["gold_label"]:
            continue
        key = (record["gold_label"], predicted)
        pairs[key] += 1
        by_pair[key].append(record)
    galleries: list[dict[str, Any]] = []
    criteria = criteria or {}
    for (gold, predicted), count in pairs.most_common(top_pairs):
        examples = []
        for record in sorted(by_pair[(gold, predicted)], key=_record_key)[:examples_per_pair]:
            entry = _record_entry(record)
            entry["note"] = note_texts.get(_record_key(record), "")
            entry["criterion"] = criteria.get(_record_key(record))
            entry["sentences"] = parse_sentences(note_texts.get(_record_key(record), ""))
            examples.append(entry)
        galleries.append({"gold": gold, "predicted": predicted, "count": count, "examples": examples})
    return galleries


def missing_information_failures(
    records: list[dict[str, Any]],
    *,
    max_examples: int,
    criteria: dict[Any, str] | None = None,
) -> dict[str, Any]:
    """Records where gold or predicted `not enough information` was mishandled."""
    under = [r for r in records if r["gold_label"] == "not enough information"
             and predicted_label(r) not in (None, "not enough information")]
    over = [r for r in records if r["gold_label"] != "not enough information"
            and predicted_label(r) == "not enough information"]
    criteria = criteria or {}

    def entry(record: dict[str, Any]) -> dict[str, Any]:
        payload = _record_entry(record)
        payload["criterion"] = criteria.get(_record_key(record))
        payload["note"] = ""
        payload["sentences"] = {}
        return payload

    return {
        "gold_nei_mispredicted_count": len(under),
        "gold_nei_total": sum(1 for r in records if r["gold_label"] == "not enough information"),
        "gold_nei_mispredicted": [entry(r) for r in under[:max_examples]],
        "predicted_nei_when_not_gold_count": len(over),
        "predicted_nei_when_not_gold": [entry(r) for r in over[:max_examples]],
    }


def not_applicable_failures(
    records: list[dict[str, Any]],
    *,
    max_examples: int,
    criteria: dict[Any, str] | None = None,
) -> dict[str, Any]:
    """Records where gold `not applicable` was predicted as anything else."""
    failed = [
        r
        for r in records
        if r["gold_label"] == "not applicable"
        and predicted_label(r) not in (None, "not applicable")
    ]
    criteria = criteria or {}
    entries = []
    for record in failed[:max_examples]:
        payload = _record_entry(record)
        payload["criterion"] = criteria.get(_record_key(record))
        payload["note"] = ""
        payload["sentences"] = {}
        entries.append(payload)
    return {
        "gold_na_total": sum(1 for r in records if r["gold_label"] == "not applicable"),
        "mispredicted_count": len(failed),
        "mispredicted": entries,
    }


def analyze_records(
    records: list[dict[str, Any]],
    split_records: list[dict[str, Any]],
    label_sets: dict[str, list[str]],
    *,
    top_pairs: int = 5,
    examples_per_pair: int = 2,
    max_examples: int = 5,
) -> dict[str, Any]:
    """Full Task 2 analysis for one prediction file against its prepared split."""
    enriched = join_with_split(records, split_records)
    parsed = [r for r in records if r.get("valid") and isinstance(r.get("prediction"), dict)]
    views = classification_views(records, label_sets)
    analysis = {
        "n_examples": len(records),
        "n_parsed": len(parsed),
        "parse_success_rate": _safe_divide(len(parsed), len(records)),
        "classification": views,
        "evidence": evidence_views(records),
        "contradictions": flag_counts(records, enriched["notes"], enriched["criteria"]),
        "errors": {
            "confusion_pairs": confusion_pair_examples(
                records,
                enriched["notes"],
                criteria=enriched["criteria"],
                top_pairs=top_pairs,
                examples_per_pair=examples_per_pair,
            ),
            "missing_information": missing_information_failures(
                records, max_examples=max_examples, criteria=enriched["criteria"]
            ),
            "not_applicable": not_applicable_failures(
                records, max_examples=max_examples, criteria=enriched["criteria"]
            ),
        },
    }
    return analysis


def join_with_split(
    predictions: list[dict[str, Any]], split_records: list[dict[str, Any]]
) -> dict[str, Any]:
    """Attach prepared-split context (criterion text, note) to predictions."""
    by_key = {
        (record["patient_id"], record["annotation_id"]): record
        for record in split_records
    }
    notes: dict[Any, str] = {}
    criteria: dict[Any, str] = {}
    enriched: list[dict[str, Any]] = []
    for record in predictions:
        split_record = by_key.get(_record_key(record))
        note = split_record["note"] if split_record else ""
        notes[_record_key(record)] = note
        criteria[_record_key(record)] = split_record["criterion_text"] if split_record else None
        ids = predicted_evidence_ids(record)
        entry = _record_entry(record)
        entry.update(
            {
                "criterion": split_record["criterion_text"] if split_record else None,
                "cited_with_text": extract_cited_evidence(
                    note, ids, n_sentences=int(record.get("n_sentences", 0))
                ),
                "valid": bool(record.get("valid")),
            }
        )
        enriched.append(entry)
    return {"enriched": enriched, "notes": notes, "criteria": criteria}


def enriched_schema(
    predictions: list[dict[str, Any]], split_records: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Task 2 per-prediction schema: criterion, category, IDs, text, explanation."""
    return join_with_split(predictions, split_records)["enriched"]
