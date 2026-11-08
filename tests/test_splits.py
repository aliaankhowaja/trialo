"""Tests for manifest verification and leakage detection."""

from __future__ import annotations

import copy
import unittest

from trailo.data.check_splits import (
    SplitVerificationError,
    load_manifest,
    load_source_dataset,
    verify_splits,
)
from trailo.paths import DEFAULT_MANIFEST


class SplitVerificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = load_manifest(DEFAULT_MANIFEST)
        cls.dataset = load_source_dataset()

    def test_real_manifest_passes(self) -> None:
        summary = verify_splits(self.manifest, self.dataset)
        self.assertTrue(summary["ok"])
        self.assertEqual(summary["splits"]["train"]["n_patients"], 37)
        self.assertEqual(summary["splits"]["validation"]["n_patients"], 8)
        self.assertEqual(summary["splits"]["test"]["n_patients"], 8)
        for overlap in summary["leakage"].values():
            self.assertEqual(overlap, [])

    def test_patient_in_two_splits_fails(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        moved = manifest["splits"]["test"]["patient_ids"][0]
        manifest["splits"]["train"]["patient_ids"].append(moved)
        manifest["splits"]["train"]["n_patients"] += 1
        with self.assertRaises(SplitVerificationError):
            verify_splits(manifest, self.dataset)

    def test_wrong_patient_count_fails(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        manifest["splits"]["validation"]["patient_ids"].pop()
        manifest["splits"]["validation"]["n_patients"] -= 1
        with self.assertRaises(SplitVerificationError):
            verify_splits(manifest, self.dataset)

    def test_missing_annotation_fails(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        manifest["splits"]["train"]["annotation_ids"].pop()
        with self.assertRaises(SplitVerificationError):
            verify_splits(manifest, self.dataset)

    def test_test_annotation_in_train_fails(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        moved = manifest["splits"]["test"]["annotation_ids"].pop()
        manifest["splits"]["train"]["annotation_ids"].append(moved)
        manifest["splits"]["test"]["n_annotations"] -= 1
        manifest["splits"]["train"]["n_annotations"] += 1
        with self.assertRaises(SplitVerificationError):
            verify_splits(manifest, self.dataset)


if __name__ == "__main__":
    unittest.main()
