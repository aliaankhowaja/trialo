"""Verify the patient-grouped split manifest and prove there is no leakage.

The manifest is the single source of truth for the train/validation/test
assignment. This module never regenerates it: if the file is missing the
verification fails loudly.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import load_dataset

from trailo.data.datasets import (
    DATASET_CONFIG,
    DATASET_NAME,
    SOURCE_REVISION,
    SOURCE_SPLIT,
    SPLIT_SIZES,
    SPLITS,
    _annotation_sha256,
    _patient_sha256,
)
from trailo.paths import DEFAULT_MANIFEST


class SplitVerificationError(RuntimeError):
    """Raised when the manifest is missing, malformed, or leaky."""


def load_manifest(path: str | Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    """Load a split manifest, refusing to create one implicitly."""
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise SplitVerificationError(
            f"split manifest not found at {manifest_path}; refusing to regenerate it"
        )
    return json.loads(manifest_path.read_text())


def load_source_dataset(revision: str = SOURCE_REVISION):
    """Load the pinned source split used by the manifest."""
    return load_dataset(
        DATASET_NAME,
        DATASET_CONFIG,
        split=SOURCE_SPLIT,
        revision=revision,
    )


def verify_splits(
    manifest: dict[str, Any],
    dataset,
    expected_counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Run every leakage and coverage check, returning a summary."""
    expected_counts = dict(expected_counts or SPLIT_SIZES)
    errors: list[str] = []

    source = manifest.get("source", {})
    if source.get("dataset") != DATASET_NAME:
        errors.append(f"manifest dataset {source.get('dataset')!r} != {DATASET_NAME!r}")
    if source.get("config") != DATASET_CONFIG:
        errors.append(f"manifest config {source.get('config')!r} != {DATASET_CONFIG!r}")
    if source.get("split") != SOURCE_SPLIT:
        errors.append(f"manifest split {source.get('split')!r} != {SOURCE_SPLIT!r}")
    if source.get("revision") != SOURCE_REVISION:
        errors.append(
            f"manifest revision {source.get('revision')!r} != {SOURCE_REVISION!r}"
        )

    splits = manifest.get("splits", {})
    if set(splits) != set(SPLITS):
        errors.append(f"manifest split names {sorted(splits)} != {sorted(SPLITS)}")

    dataset_patients = set(dataset["patient_id"])
    dataset_annotations = {int(value) for value in dataset["annotation_id"]}
    patient_row_counts = Counter(dataset["patient_id"])
    patient_to_split: dict[str, str] = {}
    annotation_to_split: dict[int, str] = {}
    summary: dict[str, Any] = {"splits": {}, "counts": {}, "errors": errors}

    for split in SPLITS:
        info = splits.get(split)
        if info is None:
            continue
        patient_ids = list(info.get("patient_ids", []))
        annotation_ids = [int(value) for value in info.get("annotation_ids", [])]
        expected = expected_counts[split]

        if info.get("n_patients") != expected:
            errors.append(
                f"{split}: n_patients={info.get('n_patients')} != {expected}"
            )
        if len(patient_ids) != expected:
            errors.append(
                f"{split}: patient_ids length {len(patient_ids)} != {expected}"
            )
        if len(set(patient_ids)) != len(patient_ids):
            errors.append(f"{split}: duplicate patient IDs")
        if info.get("n_annotations") != len(annotation_ids):
            errors.append(f"{split}: n_annotations does not match annotation_ids")
        if len(set(annotation_ids)) != len(annotation_ids):
            errors.append(f"{split}: duplicate annotation IDs")
        if info.get("annotation_ids_sha256") != _annotation_sha256(annotation_ids):
            errors.append(f"{split}: annotation ID hash mismatch")
        if info.get("patient_ids_sha256") != _patient_sha256(patient_ids):
            errors.append(f"{split}: patient ID hash mismatch")

        for patient_id in patient_ids:
            previous = patient_to_split.setdefault(patient_id, split)
            if previous != split:
                errors.append(
                    f"patient {patient_id!r} appears in both {previous} and {split}"
                )
        for annotation_id in annotation_ids:
            previous = annotation_to_split.setdefault(annotation_id, split)
            if previous != split:
                errors.append(
                    f"annotation {annotation_id} appears in both {previous} and {split}"
                )

        expected_rows = sum(patient_row_counts[patient_id] for patient_id in patient_ids)
        if expected_rows != len(annotation_ids):
            errors.append(
                f"{split}: patient row sum {expected_rows} != "
                f"annotation count {len(annotation_ids)}"
            )
        if set(patient_ids) - dataset_patients:
            errors.append(f"{split}: manifest patients missing from the dataset")
        if set(annotation_ids) - dataset_annotations:
            errors.append(f"{split}: manifest annotations missing from the dataset")

        summary["splits"][split] = {
            "n_patients": len(patient_ids),
            "n_annotations": len(annotation_ids),
        }
        summary["counts"][split] = len(annotation_ids)

    all_patients = [patient for split in SPLITS for patient in splits.get(split, {}).get("patient_ids", [])]
    all_annotations = [
        int(value)
        for split in SPLITS
        for value in splits.get(split, {}).get("annotation_ids", [])
    ]

    if len(set(all_patients)) != len(all_patients):
        errors.append("the same patient appears in more than one split")
    if len(set(all_annotations)) != len(all_annotations):
        errors.append("the same annotation appears in more than one split")
    if set(all_patients) != dataset_patients:
        errors.append("manifest patients do not exactly cover the dataset patients")
    if set(all_annotations) != dataset_annotations:
        errors.append("manifest annotations do not exactly cover the dataset rows")

    for annotation_id, patient_id in zip(dataset["annotation_id"], dataset["patient_id"]):
        split = patient_to_split.get(patient_id)
        if split is None:
            errors.append(f"dataset patient {patient_id!r} is not assigned to a split")
            continue
        if int(annotation_id) not in set(
            int(value) for value in splits[split]["annotation_ids"]
        ):
            errors.append(
                f"annotation {annotation_id} is assigned to {split} by its patient, "
                "but is absent from that split's annotation list"
            )

    train_patients = set(splits.get("train", {}).get("patient_ids", []))
    validation_patients = set(splits.get("validation", {}).get("patient_ids", []))
    test_patients = set(splits.get("test", {}).get("patient_ids", []))
    train_annotations = set(
        int(value) for value in splits.get("train", {}).get("annotation_ids", [])
    )
    validation_annotations = set(
        int(value) for value in splits.get("validation", {}).get("annotation_ids", [])
    )
    test_annotations = set(
        int(value) for value in splits.get("test", {}).get("annotation_ids", [])
    )

    leakage = {
        "train_test_patient_overlap": sorted(train_patients & test_patients),
        "validation_test_patient_overlap": sorted(validation_patients & test_patients),
        "train_validation_patient_overlap": sorted(
            train_patients & validation_patients
        ),
        "train_test_annotation_overlap": sorted(train_annotations & test_annotations),
        "validation_test_annotation_overlap": sorted(
            validation_annotations & test_annotations
        ),
        "train_validation_annotation_overlap": sorted(
            train_annotations & validation_annotations
        ),
    }
    for name, overlap in leakage.items():
        if overlap:
            errors.append(f"leakage check failed: {name} has {len(overlap)} entries")

    summary["leakage"] = leakage
    summary["dataset"] = {
        "n_rows": len(dataset),
        "n_patients": len(dataset_patients),
        "n_annotations": len(dataset_annotations),
    }
    summary["ok"] = not errors
    if errors:
        raise SplitVerificationError("\n".join(errors))
    return summary


def format_summary(summary: dict[str, Any]) -> str:
    """Render a compact human-readable verification report."""
    lines = ["Split verification summary", "-" * 60]
    for split in SPLITS:
        info = summary["splits"][split]
        lines.append(
            f"{split:12s} patients={info['n_patients']:3d} "
            f"annotations={info['n_annotations']:4d}"
        )
    dataset = summary["dataset"]
    lines.append(
        f"dataset      patients={dataset['n_patients']:3d} "
        f"annotations={dataset['n_annotations']:4d} rows={dataset['n_rows']:4d}"
    )
    lines.append("Leakage check summary")
    for name, overlap in summary["leakage"].items():
        lines.append(f"  {name}: {'FAIL ' + str(overlap[:5]) if overlap else 'ok'}")
    lines.append(f"Result: {'PASS' if summary['ok'] else 'FAIL'}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument(
        "--revision",
        default=SOURCE_REVISION,
        help="Pinned dataset revision; defaults to the manifest revision.",
    )
    args = parser.parse_args(argv)

    manifest = load_manifest(args.manifest)
    revision = args.revision or manifest.get("source", {}).get("revision")
    dataset = load_source_dataset(revision)
    summary = verify_splits(manifest, dataset)
    print(format_summary(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
