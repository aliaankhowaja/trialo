from __future__ import annotations

import ast
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset, load_dataset

DATASET_NAME = "ncbi/TrialGPT-Criterion-Annotations"
DATASET_CONFIG = "default"
SOURCE_SPLIT = "train"
SOURCE_REVISION = "1cfcafde94a1560a33b4addc1638664fe28fc059"
SEED = 42
SPLIT_SIZES = {"train": 37, "validation": 8, "test": 8}
SPLITS = tuple(SPLIT_SIZES)
MANIFEST_VERSION = 1
DEFAULT_MANIFEST_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "splits" / "split_manifest.json"
)
LEAK_COLUMNS = (
    "gpt4_explanation",
    "explanation_correctness",
    "gpt4_sentences",
    "gpt4_eligibility",
    "training",
)
KEEP_COLUMNS = (
    "annotation_id",
    "patient_id",
    "note",
    "trial_id",
    "trial_title",
    "criterion_type",
    "criterion_text",
    "expert_eligibility",
    "expert_sentences",
)


def _annotation_sha256(annotation_ids: list[int]) -> str:
    payload = ",".join(str(i) for i in sorted(annotation_ids)).encode()
    return hashlib.sha256(payload).hexdigest()


def _patient_sha256(patient_ids: list[str]) -> str:
    payload = ",".join(sorted(patient_ids)).encode()
    return hashlib.sha256(payload).hexdigest()


class TrialGPTCriterionDataset(torch.utils.data.Dataset):
    """Criterion-level TrialGPT dataset with leakage-safe patient-grouped splits."""

    def __init__(
        self,
        split: str,
        manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
        allow_test: bool = False,
        revision: str = SOURCE_REVISION,
    ):
        """Loads one patient-grouped split and creates the manifest on first use."""
        super().__init__()

        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
        if split == "test" and not allow_test:
            raise PermissionError(
                "the test split is locked; pass allow_test=True only for final evaluation"
            )

        self.split = split
        self.manifest_path = Path(manifest_path)
        self.revision = revision

        dataset = load_dataset(
            DATASET_NAME,
            DATASET_CONFIG,
            split=SOURCE_SPLIT,
            revision=revision,
        )
        self.manifest = self._get_or_create_manifest(dataset)
        split_info = self.manifest["splits"][split]

        split_patients = set(split_info["patient_ids"])
        indices = [
            i for i, patient_id in enumerate(dataset["patient_id"])
            if patient_id in split_patients
        ]
        self.data = dataset.select(indices).select_columns(KEEP_COLUMNS)
        self._validate_selected_rows(split_info)

    @property
    def patient_ids(self) -> tuple[str, ...]:
        """Patient IDs assigned to this split."""
        return tuple(self.manifest["splits"][self.split]["patient_ids"])

    @property
    def annotation_ids(self) -> tuple[int, ...]:
        """Annotation IDs assigned to this split."""
        return tuple(self.manifest["splits"][self.split]["annotation_ids"])

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> dict:
        row = dict(self.data[index])
        row["expert_sentences"] = ast.literal_eval(row["expert_sentences"])
        return row

    def _get_or_create_manifest(self, dataset: Dataset) -> dict:
        if self.manifest_path.exists():
            manifest = json.loads(self.manifest_path.read_text())
            self._validate_manifest(manifest, dataset)
            return manifest

        manifest = self._build_manifest(dataset)
        self._write_manifest(manifest)
        return manifest

    def _build_manifest(self, dataset: Dataset) -> dict:
        patients = sorted(set(dataset["patient_id"]))
        groups = self._assign_patients(patients)
        patient_to_split = {
            patient_id: split
            for split, patient_ids in groups.items()
            for patient_id in patient_ids
        }
        annotation_ids = defaultdict(list)

        for annotation_id, patient_id in zip(
            dataset["annotation_id"], dataset["patient_id"]
        ):
            annotation_ids[patient_to_split[patient_id]].append(int(annotation_id))

        split_info = {}
        for split in SPLITS:
            ids = sorted(annotation_ids[split])
            split_info[split] = {
                "patient_ids": groups[split],
                "n_patients": len(groups[split]),
                "n_annotations": len(ids),
                "annotation_ids": ids,
                "annotation_ids_sha256": _annotation_sha256(ids),
                "patient_ids_sha256": _patient_sha256(groups[split]),
            }

        return {
            "version": MANIFEST_VERSION,
            "source": {
                "dataset": DATASET_NAME,
                "config": DATASET_CONFIG,
                "split": SOURCE_SPLIT,
                "revision": self.revision,
            },
            "split_strategy": {
                "seed": SEED,
                "rng": "numpy.random.default_rng(42)",
                "ordering": "sorted(unique(patient_id))",
                "counts": dict(SPLIT_SIZES),
            },
            "splits": split_info,
        }

    def _assign_patients(self, patients: list[str]) -> dict[str, list[str]]:
        expected = sum(SPLIT_SIZES.values())
        if len(patients) != expected:
            raise ValueError(
                f"expected {expected} unique patients, found {len(patients)}"
            )

        shuffled = list(patients)
        rng = np.random.default_rng(SEED)
        rng.shuffle(shuffled)

        groups = {}
        start = 0
        for split in SPLITS:
            end = start + SPLIT_SIZES[split]
            groups[split] = shuffled[start:end]
            start = end
        return groups

    def _write_manifest(self, manifest: dict) -> None:
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.manifest_path.with_suffix(self.manifest_path.suffix + ".tmp")
        temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        temporary.replace(self.manifest_path)

    def _validate_manifest(self, manifest: dict, dataset: Dataset) -> None:
        source = manifest.get("source", {})
        strategy = manifest.get("split_strategy", {})

        if manifest.get("version") != MANIFEST_VERSION:
            raise ValueError("manifest version mismatch")
        if source.get("dataset") != DATASET_NAME:
            raise ValueError("manifest source dataset mismatch")
        if source.get("config") != DATASET_CONFIG:
            raise ValueError("manifest source config mismatch")
        if source.get("split") != SOURCE_SPLIT:
            raise ValueError("manifest source split mismatch")
        if source.get("revision") != self.revision:
            raise ValueError("manifest source revision mismatch")
        if strategy.get("seed") != SEED:
            raise ValueError("manifest seed mismatch")
        if strategy.get("counts") != SPLIT_SIZES:
            raise ValueError("manifest split counts mismatch")

        splits = manifest.get("splits", {})
        if set(splits) != set(SPLITS):
            raise ValueError("manifest split names mismatch")

        dataset_patients = set(dataset["patient_id"])
        dataset_annotations = set(int(i) for i in dataset["annotation_id"])
        patient_counts = Counter(dataset["patient_id"])
        manifest_patients = []
        manifest_annotations = []

        for split in SPLITS:
            info = splits[split]
            patient_ids = info.get("patient_ids", [])
            annotation_ids = info.get("annotation_ids", [])

            if info.get("n_patients") != SPLIT_SIZES[split]:
                raise ValueError(f"manifest {split} patient count mismatch")
            if len(patient_ids) != SPLIT_SIZES[split]:
                raise ValueError(f"manifest {split} patient list mismatch")
            if info.get("n_annotations") != len(annotation_ids):
                raise ValueError(f"manifest {split} annotation count mismatch")
            if info.get("annotation_ids_sha256") != _annotation_sha256(annotation_ids):
                raise ValueError(f"manifest {split} annotation hash mismatch")
            if info.get("patient_ids_sha256") != _patient_sha256(patient_ids):
                raise ValueError(f"manifest {split} patient hash mismatch")
            if sum(patient_counts[p] for p in patient_ids) != len(annotation_ids):
                raise ValueError(f"manifest {split} row count mismatch")

            manifest_patients.extend(patient_ids)
            manifest_annotations.extend(annotation_ids)

        if len(set(manifest_patients)) != len(manifest_patients):
            raise ValueError("manifest contains a patient in more than one split")
        if set(manifest_patients) != dataset_patients:
            raise ValueError("manifest patients do not match the source dataset")
        if len(set(manifest_annotations)) != len(manifest_annotations):
            raise ValueError("manifest contains duplicate annotations")
        if set(manifest_annotations) != dataset_annotations:
            raise ValueError("manifest annotations do not match the source dataset")

    def _validate_selected_rows(self, split_info: dict) -> None:
        selected_annotations = sorted(int(i) for i in self.data["annotation_id"])
        selected_patients = set(self.data["patient_id"])

        if selected_annotations != sorted(split_info["annotation_ids"]):
            raise ValueError("selected annotations do not match the manifest")
        if selected_patients != set(split_info["patient_ids"]):
            raise ValueError("selected patients do not match the manifest")
        if set(LEAK_COLUMNS) & set(self.data.column_names):
            raise ValueError("leak-prone model output columns were not removed")
