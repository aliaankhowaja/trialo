"""Strict JSON parsing and validation for criterion predictions."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

REQUIRED_KEYS = ("label", "evidence_ids", "explanation")
EMPTY_EVIDENCE_LABELS = {"not enough information", "not applicable"}


@dataclass
class ValidationResult:
    """Outcome of parsing and validating one model output."""

    prediction: dict[str, Any] | None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.errors


class PredictionParseError(ValueError):
    """Raised when no single JSON object can be extracted."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def extract_json_object(raw: str) -> str:
    """Extract exactly one JSON object from raw model text.

    Leading prose is tolerated because base models often add a sentence before
    the JSON. Trailing non-whitespace content is rejected so the output remains
    strict and machine-parseable.
    """
    if not isinstance(raw, str):
        raise PredictionParseError("raw output is not a string")
    text = raw.strip()
    if not text:
        raise PredictionParseError("empty output")

    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline == -1:
            raise PredictionParseError("malformed fenced block")
        text = text[first_newline + 1 :].strip()
        if text.endswith("```"):
            text = text[:-3].strip()

    start = text.find("{")
    if start == -1:
        raise PredictionParseError("no JSON object found")

    in_string = False
    escaped = False
    depth = 0
    for index in range(start, len(text)):
        character = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : index + 1]
                remainder = text[index + 1 :].strip()
                if remainder and remainder != "```":
                    raise PredictionParseError(
                        f"unexpected trailing content after JSON: {remainder[:40]!r}"
                    )
                return candidate
    raise PredictionParseError("unbalanced JSON object")


def parse_json_object(raw: str) -> dict[str, Any]:
    """Parse one JSON object, rejecting duplicate keys and non-objects."""
    candidate = extract_json_object(raw)
    try:
        parsed = json.loads(candidate, object_pairs_hook=_reject_duplicate_keys)
    except ValueError as error:
        raise PredictionParseError(f"invalid JSON: {error}") from error
    if not isinstance(parsed, dict):
        raise PredictionParseError("top-level JSON value is not an object")
    return parsed


def validate_prediction(
    prediction: Any,
    *,
    criterion_type: str,
    n_sentences: int,
    allowed_labels: list[str],
) -> ValidationResult:
    """Validate one parsed prediction against the task schema."""
    errors: list[str] = []
    warnings: list[str] = []

    if not isinstance(prediction, dict):
        return ValidationResult(None, ["prediction is not a JSON object"])

    keys = set(prediction)
    missing = set(REQUIRED_KEYS) - keys
    extra = keys - set(REQUIRED_KEYS)
    if missing:
        errors.append(f"missing keys: {sorted(missing)}")
    if extra:
        errors.append(f"unexpected keys: {sorted(extra)}")

    label = prediction.get("label")
    if not isinstance(label, str):
        errors.append("label must be a string")
    elif label not in allowed_labels:
        errors.append(
            f"invalid label {label!r} for {criterion_type}; "
            f"allowed: {sorted(allowed_labels)}"
        )

    evidence_ids = prediction.get("evidence_ids")
    if not isinstance(evidence_ids, list):
        errors.append("evidence_ids must be a list")
    else:
        for value in evidence_ids:
            if isinstance(value, bool) or not isinstance(value, int):
                errors.append(f"evidence id {value!r} is not an integer")
            elif value < 0 or value >= n_sentences:
                errors.append(
                    f"evidence id {value} is outside [0, {n_sentences - 1}]"
                )
        if (
            isinstance(label, str)
            and label in EMPTY_EVIDENCE_LABELS
            and evidence_ids
            and not errors
        ):
            warnings.append(
                f"label {label!r} has evidence ids; allowed only when gold does"
            )

    explanation = prediction.get("explanation")
    if not isinstance(explanation, str):
        errors.append("explanation must be a string")

    return ValidationResult(prediction, errors, warnings)


def parse_and_validate(
    raw: str,
    *,
    criterion_type: str,
    n_sentences: int,
    allowed_labels: list[str],
) -> ValidationResult:
    """Parse a raw model output and validate the resulting prediction."""
    try:
        prediction = parse_json_object(raw)
    except PredictionParseError as error:
        return ValidationResult(None, [str(error)])
    return validate_prediction(
        prediction,
        criterion_type=criterion_type,
        n_sentences=n_sentences,
        allowed_labels=allowed_labels,
    )
