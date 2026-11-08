"""Label inventory derived from the expert annotations."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

CRITERION_TYPES = ("inclusion", "exclusion")

EXPECTED_LABEL_SETS = {
    "inclusion": {
        "included",
        "not included",
        "not enough information",
        "not applicable",
    },
    "exclusion": {
        "excluded",
        "not excluded",
        "not enough information",
        "not applicable",
    },
}


class LabelSetError(RuntimeError):
    """Raised when the dataset label inventory differs from the specification."""


def derive_label_sets(rows: Iterable[dict[str, Any]]) -> dict[str, list[str]]:
    """Read the actual label strings from expert annotations."""
    found: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        criterion_type = row["criterion_type"]
        label = row["expert_eligibility"]
        if not isinstance(label, str) or not label:
            raise LabelSetError(f"invalid expert label {label!r}")
        found[criterion_type].add(label)

    if set(found) != set(CRITERION_TYPES):
        raise LabelSetError(
            f"criterion types {sorted(found)} != {sorted(CRITERION_TYPES)}"
        )
    result = {key: sorted(values) for key, values in found.items()}
    for criterion_type, expected in EXPECTED_LABEL_SETS.items():
        actual = set(result[criterion_type])
        if actual != expected:
            raise LabelSetError(
                f"{criterion_type} labels {sorted(actual)} != expected {sorted(expected)}"
            )
    return result


def verify_label_subset(
    label_sets: dict[str, list[str]],
    rows: Iterable[dict[str, Any]],
    source_name: str,
) -> None:
    """Ensure every label in a split was present in the training inventory."""
    seen: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        seen[row["criterion_type"]].add(row["expert_eligibility"])
    for criterion_type, labels in seen.items():
        unknown = labels - set(label_sets.get(criterion_type, []))
        if unknown:
            raise LabelSetError(
                f"{source_name} contains labels absent from train: {sorted(unknown)}"
            )
