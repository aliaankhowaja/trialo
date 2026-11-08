"""Re-tokenize prepared records and verify truncation safety without a model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trailo.data.prompting import analyze_truncation, parse_note_sentences
from trailo.paths import repo_path
from trailo.train.config import get_section, load_config
from trailo.train.modeling import load_tokenizer


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--processed-dir", default=None)
    parser.add_argument("--max-length", type=int, default=None)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    model_config = get_section(config, "model")
    data_config = get_section(config, "data")
    max_length = args.max_length or int(data_config["max_length"])
    processed_dir = repo_path(args.processed_dir or data_config["processed_dir"])
    tokenizer = load_tokenizer(
        model_config["model_id"], model_config.get("revision")
    )
    records = load_jsonl(processed_dir / f"{args.split}.jsonl")

    failures: list[str] = []
    for record in records:
        sentences = parse_note_sentences(record["note"])
        analysis = analyze_truncation(
            tokenizer,
            prompt_text=record["prompt"],
            full_text=record["full_text"],
            note_text=record["note"],
            sentences=sentences,
            evidence_ids=record["gold_evidence_ids"],
            max_length=max_length,
        )
        for key in ("prompt_tokens", "target_tokens", "full_tokens"):
            if analysis[key] != record[key]:
                failures.append(
                    f"annotation {record['annotation_id']}: {key} changed "
                    f"{record[key]} -> {analysis[key]}"
                )
        if not analysis["fits"]:
            failures.append(f"annotation {record['annotation_id']}: does not fit")
        if not analysis["evidence_preserved"]:
            failures.append(
                f"annotation {record['annotation_id']}: evidence would be truncated"
            )

    if failures:
        print("\n".join(failures[:20]))
        raise SystemExit(f"tokenization check failed for {len(failures)} records")

    lengths = [record["full_tokens"] for record in records]
    print(
        f"{args.split}: {len(records)} records verified; "
        f"full tokens min={min(lengths)} max={max(lengths)} max_length={max_length}"
    )
    print("All records fit and all gold-evidence sentences are preserved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
