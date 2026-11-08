"""Tests for the Task 2 analysis module."""

from __future__ import annotations

import unittest

from trailo.eval.analysis import (
    classification_views,
    confusion_matrix,
    evidence_views,
    extract_cited_evidence,
    flag_contradictions,
    missing_information_failures,
    not_applicable_failures,
    parse_sentences,
)

LABEL_SETS = {
    "inclusion": [
        "included",
        "not applicable",
        "not enough information",
        "not included",
    ],
    "exclusion": [
        "excluded",
        "not applicable",
        "not enough information",
        "not excluded",
    ],
}
UNIVERSE = sorted(set(LABEL_SETS["inclusion"]) | set(LABEL_SETS["exclusion"]))

NOTE = "0. The patient has fever.\n1. No pain reported.\n2. BMI is 22.\n"


def record(gold, predicted=None, evidence_ids=None, *, criterion_type="inclusion",
           gold_evidence=None):
    payload = {
        "patient_id": "p1",
        "annotation_id": 0,
        "criterion_type": criterion_type,
        "gold_label": gold,
        "gold_evidence_ids": [0, 2] if gold_evidence is None else gold_evidence,
        "n_sentences": 3,
        "valid": True,
        "prediction": {
            "label": gold if predicted is None else predicted,
            "evidence_ids": [0, 2] if evidence_ids is None else evidence_ids,
            "explanation": "…",
        },
    }
    if predicted is None and evidence_ids is None:
        payload["prediction"]["evidence_ids"] = []
    return payload


class SentenceTests(unittest.TestCase):
    def test_parse_sentences(self):
        sentences = parse_sentences(NOTE)
        self.assertEqual(list(sentences), [0, 1, 2])
        self.assertEqual(sentences[1], "No pain reported.")

    def test_extract_cited_evidence_orders_and_flags_invalid(self):
        cited = extract_cited_evidence(NOTE, [1, 5], n_sentences=3)
        self.assertEqual(cited[0], {"id": 1, "text": "No pain reported.", "valid": True})
        self.assertEqual(cited[1]["valid"], False)


class ConfusionTests(unittest.TestCase):
    def test_counts_and_per_class(self):
        records = [
            record("included", "included"),
            record("included", "not included"),
            record("excluded", "excluded", criterion_type="exclusion"),
            record("not applicable", "not applicable", criterion_type="exclusion"),
            record("not excluded", "not included", criterion_type="exclusion"),
            {"patient_id": "p", "annotation_id": 9, "criterion_type": "inclusion",
             "gold_label": "included", "valid": False, "prediction": None},
        ]
        view = confusion_matrix(records, UNIVERSE)
        self.assertEqual(view["counts"][UNIVERSE.index("included")][UNIVERSE.index("included")], 1)
        self.assertEqual(view["counts"][UNIVERSE.index("included")][UNIVERSE.index("not included")], 1)
        self.assertEqual(view["n_unparsed"], 1)
        self.assertEqual(view["per_class"]["included"]["support"], 2)
        self.assertAlmostEqual(view["accuracy"], 3 / 6)
        self.assertAlmostEqual(view["accuracy_parsed"], 3 / 5)
        # "excluded": 1 true positive, no false positives
        self.assertAlmostEqual(view["per_class"]["excluded"]["recall"], 1.0)
        self.assertAlmostEqual(view["per_class"]["excluded"]["precision"], 1.0)

    def test_row_normalized(self):
        records = [record("included", "included"), record("included", "not included")]
        view = confusion_matrix(records, UNIVERSE)
        row = view["row_normalized"][UNIVERSE.index("included")]
        self.assertAlmostEqual(sum(row), 1.0)

    def test_macro_f1_both_conventions(self):
        records = [record("included", "included"), record("included", "not included")]
        view = confusion_matrix(records, UNIVERSE)
        # included: tp=1, fp=0, fn=1 -> F1=2/3; not included: predicted-only, F1=0.
        # declared macro averages F1 over all 6 universe labels.
        self.assertAlmostEqual(view["macro_f1"], (2 / 3) / len(UNIVERSE))
        # observed macro averages only over included and not included.
        self.assertAlmostEqual(view["macro_f1_observed"], (2 / 3) / 2)

        # A never-predicted declared class lowers only the declared macro.
        records = [
            record("included", "included", criterion_type="inclusion"),
            record("excluded", "excluded", criterion_type="exclusion"),
        ]
        classification = classification_views(records, LABEL_SETS)
        overall = classification["overall"]
        self.assertLess(overall["macro_f1"], overall["macro_f1_observed"])
        # observed macro: included F1=1, excluded F1=1 -> 1
        self.assertAlmostEqual(overall["macro_f1_observed"], 1.0)
        # declared macro includes 4+ never-predicted classes at F1=0
        self.assertAlmostEqual(
            overall["macro_f1"], 2 / len(overall["labels"])
        )


class EvidenceTests(unittest.TestCase):
    def test_views_separate_empty_gold(self):
        records = [
            # gold [0,2], cites [0]: one true positive, one miss
            record("included", evidence_ids=[0], criterion_type="inclusion"),
            # gold empty, prediction empty -> consistency only
            record("not enough information", "not enough information",
                   evidence_ids=[], gold_evidence=[]),
            # gold empty, prediction cites -> pooled false positive
            record("not enough information", "not enough information",
                   evidence_ids=[2], gold_evidence=[]),
        ]
        evidence = evidence_views(records)
        pooled = evidence["pooled"]
        self.assertAlmostEqual(pooled["precision"], 1 / 2)  # 1 tp, 1 fp
        self.assertAlmostEqual(pooled["recall"], 1 / 2)  # 1 tp, 1 fn
        gold_present = evidence["gold_present"]
        self.assertEqual(gold_present["n_records"], 1)
        self.assertAlmostEqual(gold_present["recall"], 0.5)
        self.assertAlmostEqual(gold_present["precision"], 1.0)
        consistency = evidence["no_gold_evidence"]
        self.assertEqual(consistency["n_records"], 2)
        self.assertAlmostEqual(consistency["empty_prediction_rate"], 0.5)
        self.assertAlmostEqual(consistency["cited_without_gold_rate"], 0.5)
        self.assertEqual(evidence["per_record"]["n_records"], 1)
        self.assertAlmostEqual(evidence["per_record"]["macro_f1"], 2 / 3)

    def test_perfect_gold_present_view(self):
        evidence = evidence_views([record("included", evidence_ids=[0, 2])])
        self.assertAlmostEqual(evidence["gold_present"]["recall"], 1.0)
        self.assertAlmostEqual(evidence["gold_present"]["f1"], 1.0)
        self.assertAlmostEqual(evidence["pooled"]["f1"], 1.0)


class FlagTests(unittest.TestCase):
    def test_eligible_without_evidence_flagged(self):
        record_data = record("included", "included", evidence_ids=[],
                             criterion_type="inclusion")
        flags = flag_contradictions(record_data, note=NOTE)
        self.assertEqual(
            [flag["flag"] for flag in flags], ["included_without_evidence"]
        )
        self.assertEqual(flags[0]["severity"], 3)

    def test_ineligible_without_evidence_flagged_lower(self):
        record_data = record("not included", "not included", evidence_ids=[],
                             criterion_type="inclusion")
        flags = flag_contradictions(record_data, note=NOTE)
        self.assertEqual(flags[0]["flag"], "not included_without_evidence")
        self.assertEqual(flags[0]["severity"], 2)

    def test_negation_flag(self):
        record_data = record("included", "included", evidence_ids=[1],
                             criterion_type="inclusion")
        flags = flag_contradictions(record_data, note=NOTE)
        self.assertIn("eligible_with_negated_evidence", [f["flag"] for f in flags])

    def test_eligible_with_matching_evidence_not_flagged(self):
        record_data = record("included", "included", evidence_ids=[0],
                             criterion_type="inclusion")
        self.assertEqual(flag_contradictions(record_data, note=NOTE), [])

    def test_out_of_range_and_duplicate(self):
        record_data = record("included", "included", evidence_ids=[9, 9],
                             criterion_type="inclusion")
        flags = flag_contradictions(record_data, note=NOTE)
        names = [f["flag"] for f in flags]
        self.assertIn("evidence_id_out_of_range", names)
        self.assertIn("duplicate_evidence_ids", names)

    def test_no_flags_for_neutral_prediction(self):
        record_data = record("not enough information", "not enough information",
                             evidence_ids=[])
        self.assertEqual(flag_contradictions(record_data, note=NOTE), [])


class GalleryTests(unittest.TestCase):
    def test_missing_information_galleries(self):
        records = [
            record("not enough information", "included", evidence_ids=[0]),
            record("included", "included", evidence_ids=[0]),
        ]
        result = missing_information_failures(records, max_examples=5)
        self.assertEqual(result["gold_nei_total"], 1)
        self.assertEqual(result["gold_nei_mispredicted_count"], 1)
        self.assertEqual(result["predicted_nei_when_not_gold_count"], 0)

    def test_not_applicable_gallery(self):
        records = [
            record("not applicable", "not included"),
            record("not applicable", "not applicable"),
        ]
        result = not_applicable_failures(records, max_examples=5)
        self.assertEqual(result["gold_na_total"], 2)
        self.assertEqual(result["mispredicted_count"], 1)


if __name__ == "__main__":
    unittest.main()
