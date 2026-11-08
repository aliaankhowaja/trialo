"""Tests for the strict parser, validator, and metrics."""

from __future__ import annotations

import json
import unittest

from trailo.eval.metrics import compute_metrics
from trailo.eval.parsing import (
    PredictionParseError,
    extract_json_object,
    parse_and_validate,
    parse_json_object,
    validate_prediction,
)

INCLUSION_LABELS = [
    "included",
    "not applicable",
    "not enough information",
    "not included",
]


def valid_prediction(**overrides):
    prediction = {
        "label": "included",
        "evidence_ids": [0, 2],
        "explanation": "The note says the patient has the condition.",
    }
    prediction.update(overrides)
    return prediction


def parse(raw: str, n_sentences: int = 5):
    return parse_and_validate(
        raw,
        criterion_type="inclusion",
        n_sentences=n_sentences,
        allowed_labels=INCLUSION_LABELS,
    )


class ParserTests(unittest.TestCase):
    def test_valid_json(self) -> None:
        result = parse(json.dumps(valid_prediction()))
        self.assertTrue(result.valid, result.errors)
        self.assertEqual(result.prediction["label"], "included")

    def test_empty_evidence_is_valid(self) -> None:
        result = parse(
            json.dumps(valid_prediction(label="not enough information", evidence_ids=[]))
        )
        self.assertTrue(result.valid, result.errors)

    def test_leading_prose_is_tolerated(self) -> None:
        result = parse("Here is the answer:\n" + json.dumps(valid_prediction()))
        self.assertTrue(result.valid, result.errors)

    def test_fenced_json_is_tolerated(self) -> None:
        raw = "```json\n" + json.dumps(valid_prediction()) + "\n```"
        result = parse(raw)
        self.assertTrue(result.valid, result.errors)

    def test_malformed_json_fails(self) -> None:
        result = parse('{"label": "included", "evidence_ids": [0,}')
        self.assertFalse(result.valid)
        self.assertTrue(result.errors)

    def test_trailing_content_fails(self) -> None:
        raw = json.dumps(valid_prediction()) + "\nextra"
        with self.assertRaises(PredictionParseError):
            extract_json_object(raw)

    def test_invalid_label_fails(self) -> None:
        result = parse(json.dumps(valid_prediction(label="excluded")))
        self.assertFalse(result.valid)
        self.assertTrue(any("invalid label" in error for error in result.errors))

    def test_out_of_range_id_fails(self) -> None:
        result = parse(json.dumps(valid_prediction(evidence_ids=[5])), n_sentences=5)
        self.assertFalse(result.valid)
        self.assertTrue(any("outside" in error for error in result.errors))

    def test_negative_id_fails(self) -> None:
        result = parse(json.dumps(valid_prediction(evidence_ids=[-1])))
        self.assertFalse(result.valid)

    def test_non_integer_id_fails(self) -> None:
        result = parse(json.dumps(valid_prediction(evidence_ids=["0"])))
        self.assertFalse(result.valid)

    def test_boolean_id_fails(self) -> None:
        result = parse(json.dumps(valid_prediction(evidence_ids=[True])))
        self.assertFalse(result.valid)

    def test_extra_key_fails(self) -> None:
        result = parse(json.dumps(valid_prediction(confidence=0.9)))
        self.assertFalse(result.valid)
        self.assertTrue(any("unexpected keys" in error for error in result.errors))

    def test_missing_key_fails(self) -> None:
        prediction = valid_prediction()
        prediction.pop("explanation")
        result = parse(json.dumps(prediction))
        self.assertFalse(result.valid)

    def test_duplicate_key_fails(self) -> None:
        raw = '{"label": "included", "label": "included", "evidence_ids": [], "explanation": ""}'
        with self.assertRaises(PredictionParseError):
            parse_json_object(raw)

    def test_explanation_must_be_string(self) -> None:
        result = parse(json.dumps(valid_prediction(explanation=None)))
        self.assertFalse(result.valid)

    def test_nei_with_evidence_warns(self) -> None:
        result = validate_prediction(
            valid_prediction(label="not enough information", evidence_ids=[0]),
            criterion_type="inclusion",
            n_sentences=5,
            allowed_labels=INCLUSION_LABELS,
        )
        self.assertTrue(result.valid)
        self.assertTrue(result.warnings)


class MetricTests(unittest.TestCase):
    def test_metrics_with_mixed_parse_status(self) -> None:
        records = [
            {
                "gold_label": "included",
                "criterion_type": "inclusion",
                "prediction": valid_prediction(label="included"),
                "valid": True,
                "gold_evidence_ids": [0, 2],
            },
            {
                "gold_label": "not included",
                "criterion_type": "inclusion",
                "prediction": valid_prediction(label="included"),
                "valid": True,
                "gold_evidence_ids": [],
            },
            {
                "gold_label": "not enough information",
                "criterion_type": "inclusion",
                "prediction": None,
                "valid": False,
                "gold_evidence_ids": [1],
            },
        ]
        metrics = compute_metrics(records)
        self.assertEqual(metrics["n_examples"], 3)
        self.assertEqual(metrics["n_parsed"], 2)
        self.assertAlmostEqual(metrics["parse_success_rate"], 2 / 3)
        self.assertAlmostEqual(metrics["label_accuracy"], 1 / 3)
        self.assertAlmostEqual(metrics["label_accuracy_parsed"], 1 / 2)
        self.assertEqual(metrics["per_label"]["included"]["support"], 1)
        self.assertEqual(metrics["per_label"]["not included"]["support"], 1)
        self.assertEqual(
            metrics["per_label"]["not enough information"]["support"], 1
        )
        self.assertEqual(metrics["per_criterion"]["inclusion"]["micro_support"], 3)
        self.assertIn("inclusion", metrics["per_criterion"])
        self.assertAlmostEqual(metrics["evidence"]["exact_match_rate"], 1 / 3)


if __name__ == "__main__":
    unittest.main()
