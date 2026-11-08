"""Prepared-example datasets and padding collator for causal LM training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from trailo.data.prompting import tokenize_ids


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load prepared JSONL records."""
    return [
        json.loads(line)
        for line in Path(path).read_text().splitlines()
        if line.strip()
    ]


class PreparedCriterionDataset(Dataset):
    """Tokenize prepared prompt/target records with prompt-token masking."""

    def __init__(
        self,
        records: list[dict[str, Any]],
        tokenizer,
        max_length: int,
        drop_truncated: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.records = [
            record
            for record in records
            if not (drop_truncated and record.get("dropped", False))
        ]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        input_ids = tokenize_ids(self.tokenizer, record["full_text"])
        prompt_ids = tokenize_ids(self.tokenizer, record["prompt"])
        if input_ids[: len(prompt_ids)] != prompt_ids:
            raise ValueError(
                f"annotation {record['annotation_id']}: prompt is not a token prefix"
            )
        if len(input_ids) > self.max_length:
            input_ids = input_ids[: self.max_length]
        labels = list(input_ids)
        labels[: min(len(prompt_ids), len(labels))] = [-100] * min(
            len(prompt_ids), len(labels)
        )
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "annotation_id": record["annotation_id"],
            "n_target_tokens": sum(1 for value in labels if value != -100),
        }


class CausalLMCollator:
    """Right-pad input IDs, attention masks, and labels for causal LM loss."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = int(pad_token_id)

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_length = max(len(feature["input_ids"]) for feature in features)
        input_ids = []
        attention_mask = []
        labels = []
        for feature in features:
            padding = max_length - len(feature["input_ids"])
            input_ids.append(
                torch.cat(
                    [
                        feature["input_ids"],
                        torch.full((padding,), self.pad_token_id, dtype=torch.long),
                    ]
                )
            )
            attention_mask.append(
                torch.cat(
                    [
                        feature["attention_mask"],
                        torch.zeros(padding, dtype=torch.long),
                    ]
                )
            )
            labels.append(
                torch.cat(
                    [
                        feature["labels"],
                        torch.full((padding,), -100, dtype=torch.long),
                    ]
                )
            )
        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_mask),
            "labels": torch.stack(labels),
        }
