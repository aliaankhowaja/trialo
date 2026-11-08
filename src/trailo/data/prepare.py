"""Convert TrialGPT annotations into chat-formatted prompt/target records."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from trailo.data.datasets import (
    DATASET_NAME,
    LEAK_COLUMNS,
    SOURCE_REVISION,
    TrialGPTCriterionDataset,
)
from trailo.data.labels import derive_label_sets, verify_label_subset
from trailo.data.prompting import (
    PromptFormatError,
    analyze_truncation,
    assert_no_leakage_fields,
    chat_template_hash,
    format_example,
    parse_note_sentences,
)
from trailo.paths import DEFAULT_MANIFEST, repo_path
from trailo.train.config import get_section, load_config
from trailo.train.modeling import load_tokenizer


def _percentiles(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {"min": 0, "p50": 0, "p90": 0, "p95": 0, "p99": 0, "max": 0, "mean": 0}
    array = np.asarray(values, dtype=float)
    return {
        "min": int(array.min()),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": int(array.max()),
        "mean": float(array.mean()),
    }


def _load_split_rows(
    split: str,
    manifest_path: Path,
    subset_size: int | None,
) -> list[dict[str, Any]]:
    dataset = TrialGPTCriterionDataset(
        split,
        manifest_path=manifest_path,
        allow_test=split == "test",
        revision=SOURCE_REVISION,
    )
    rows = [dataset[index] for index in range(len(dataset))]
    rows.sort(key=lambda row: int(row["annotation_id"]))
    if subset_size is not None:
        rows = rows[:subset_size]
    return rows


def prepare_split(
    split: str,
    rows: list[dict[str, Any]],
    tokenizer,
    label_sets: dict[str, list[str]],
    max_length: int,
    allow_truncated_eval: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Prepare one split and return records plus statistics."""
    records: list[dict[str, Any]] = []
    dropped = 0
    unsafe_evidence = 0
    unsafe_target = 0
    for row in rows:
        assert_no_leakage_fields(row)
        criterion_type = row["criterion_type"]
        label = row["expert_eligibility"]
        if label not in label_sets[criterion_type]:
            raise PromptFormatError(
                f"label {label!r} not in verified {criterion_type} label set"
            )
        sentences = parse_note_sentences(row["note"])
        evidence_ids = [int(value) for value in row["expert_sentences"]]
        for sentence_id in evidence_ids:
            if sentence_id < 0 or sentence_id >= len(sentences):
                raise PromptFormatError(
                    f"annotation {row['annotation_id']} evidence id {sentence_id} "
                    f"is outside [0, {len(sentences) - 1}]"
                )
        formatted = format_example(tokenizer, row, label_sets)
        analysis = analyze_truncation(
            tokenizer,
            prompt_text=formatted["prompt_text"],
            full_text=formatted["full_text"],
            note_text=row["note"].rstrip("\n"),
            sentences=sentences,
            evidence_ids=evidence_ids,
            max_length=max_length,
        )
        is_dropped = not analysis["safe"]
        if is_dropped and split == "train":
            dropped += 1
        elif is_dropped and not allow_truncated_eval:
            raise PromptFormatError(
                f"{split} annotation {row['annotation_id']} would be truncated "
                f"({analysis}); increase max_length or explicitly allow truncated eval"
            )
        if analysis["evidence_truncated"]:
            unsafe_evidence += 1
        if analysis["target_truncated"] or analysis["prompt_truncated"]:
            unsafe_target += 1
        records.append(
            {
                "annotation_id": int(row["annotation_id"]),
                "patient_id": row["patient_id"],
                "criterion_type": criterion_type,
                "criterion_text": row["criterion_text"],
                "note": row["note"].rstrip("\n"),
                "n_sentences": len(sentences),
                "gold_label": label,
                "gold_evidence_ids": evidence_ids,
                "prompt": formatted["prompt_text"],
                "target": formatted["target_text"],
                "full_text": formatted["full_text"],
                "prompt_tokens": analysis["prompt_tokens"],
                "target_tokens": analysis["target_tokens"],
                "full_tokens": analysis["full_tokens"],
                "fits": analysis["fits"],
                "evidence_preserved": analysis["evidence_preserved"],
                "target_preserved": not analysis["target_truncated"],
                "dropped": is_dropped,
            }
        )

    stats = {
        "n_examples": len(records),
        "n_dropped": dropped,
        "n_unsafe_evidence": unsafe_evidence,
        "n_unsafe_target": unsafe_target,
        "labels": dict(Counter(record["gold_label"] for record in records)),
        "criteria": dict(Counter(record["criterion_type"] for record in records)),
        "nonempty_gold_evidence": sum(
            1 for record in records if record["gold_evidence_ids"]
        ),
        "prompt_tokens": _percentiles([record["prompt_tokens"] for record in records]),
        "target_tokens": _percentiles([record["target_tokens"] for record in records]),
        "full_tokens": _percentiles([record["full_tokens"] for record in records]),
        "max_sentence_id": max(
            (record["n_sentences"] - 1 for record in records), default=-1
        ),
    }
    return records, stats


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Write one JSON object per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--splits", default="train,validation")
    parser.add_argument("--include-test", action="store_true")
    parser.add_argument("--subset-size", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--revision", default=None)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    model_config = get_section(config, "model")
    data_config = get_section(config, "data")
    model_id = args.model_id or model_config["model_id"]
    revision = args.revision or model_config.get("revision")
    max_length = args.max_length or int(data_config["max_length"])
    manifest_path = repo_path(data_config.get("manifest_path", str(DEFAULT_MANIFEST)))
    output_dir = repo_path(args.output_dir or data_config["processed_dir"])
    include_test = args.include_test or bool(data_config.get("include_test", False))
    allow_truncated_eval = bool(data_config.get("allow_truncated_eval", False))

    splits = [value.strip() for value in args.splits.split(",") if value.strip()]
    if include_test and "test" not in splits:
        splits.append("test")
    unknown = set(splits) - {"train", "validation", "test"}
    if unknown:
        raise ValueError(f"unknown splits: {sorted(unknown)}")
    if "test" in splits and not include_test:
        raise ValueError("test split requires --include-test")

    tokenizer = load_tokenizer(model_id, revision)
    all_rows: dict[str, list[dict[str, Any]]] = {}
    for split in splits:
        all_rows[split] = _load_split_rows(
            split,
            manifest_path,
            subset_size=args.subset_size,
        )

    full_train_rows = _load_split_rows(
        "train",
        manifest_path,
        subset_size=None,
    )
    label_sets = derive_label_sets(full_train_rows)
    leakage_checks: dict[str, Any] = {
        "excluded_columns": list(LEAK_COLUMNS),
        "gpt4_fields_absent": True,
        "explanations_from_expert_annotations": False,
        "target_explanation_policy": "empty string; no expert explanation field exists",
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    stats: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_dataset": DATASET_NAME,
        "source_revision": SOURCE_REVISION,
        "model_id": model_id,
        "model_revision": revision,
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_is_fast": bool(getattr(tokenizer, "is_fast", False)),
        "chat_template_sha256": chat_template_hash(tokenizer),
        "max_length": max_length,
        "label_sets": label_sets,
        "leakage_checks": leakage_checks,
        "splits": {},
    }

    for split in splits:
        rows = all_rows[split]
        verify_label_subset(label_sets, rows, split)
        records, split_stats = prepare_split(
            split,
            rows,
            tokenizer,
            label_sets,
            max_length,
            allow_truncated_eval,
        )
        write_jsonl(output_dir / f"{split}.jsonl", records)
        stats["splits"][split] = split_stats
        print(
            f"{split}: {split_stats['n_examples']} examples, "
            f"{split_stats['n_dropped']} dropped, "
            f"full tokens p99={split_stats['full_tokens']['p99']:.1f} "
            f"max={split_stats['full_tokens']['max']}"
        )

    (output_dir / "label_sets.json").write_text(
        json.dumps(label_sets, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "stats.json").write_text(
        json.dumps(stats, indent=2, sort_keys=True) + "\n"
    )
    print(f"Wrote prepared data and stats to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
