"""Prompt construction, target construction, and truncation analysis."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

from trailo.data.datasets import LEAK_COLUMNS

SYSTEM_PROMPT = (
    "You are a clinical trial eligibility annotator. "
    "Return only one JSON object and no other text."
)

NOTE_NUMBER_PATTERN = re.compile(r"^(\d+)\.\s?(.*)$")


class PromptFormatError(RuntimeError):
    """Raised when the chat template or note numbering is malformed."""


@dataclass(frozen=True)
class Sentence:
    """One numbered patient-note sentence with character offsets."""

    sentence_id: int
    text: str
    start: int
    end: int


def parse_note_sentences(note: str) -> list[Sentence]:
    """Parse the numbered note and verify IDs are contiguous from zero."""
    if not isinstance(note, str) or not note:
        raise PromptFormatError("patient note is empty")
    lines = note.split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    sentences: list[Sentence] = []
    offset = 0
    for expected_id, line in enumerate(lines):
        match = NOTE_NUMBER_PATTERN.match(line)
        if match is None:
            raise PromptFormatError(
                f"note line {expected_id} does not start with a sentence ID: {line[:60]!r}"
            )
        sentence_id = int(match.group(1))
        if sentence_id != expected_id:
            raise PromptFormatError(
                f"note sentence IDs are not contiguous: expected {expected_id}, "
                f"found {sentence_id}"
            )
        sentences.append(
            Sentence(
                sentence_id=sentence_id,
                text=match.group(2),
                start=offset,
                end=offset + len(line),
            )
        )
        offset += len(line) + 1
    return sentences


def assert_no_leakage_fields(item: dict[str, Any]) -> None:
    """Fail if any GPT-4 prediction/explanation column survived data loading."""
    present = set(item) & set(LEAK_COLUMNS)
    if present:
        raise PromptFormatError(
            f"leak-prone fields are present in the model item: {sorted(present)}"
        )


def build_target(
    label: str,
    evidence_ids: list[int],
    explanation: str = "",
) -> str:
    """Build the strict, compact JSON training target."""
    payload = {
        "label": label,
        "evidence_ids": [int(value) for value in evidence_ids],
        "explanation": explanation,
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def build_user_prompt(item: dict[str, Any], label_sets: dict[str, list[str]]) -> str:
    """Build the task prompt with numbered patient sentences."""
    criterion_type = item["criterion_type"]
    if criterion_type not in label_sets:
        raise PromptFormatError(f"unknown criterion type {criterion_type!r}")
    note = item["note"].rstrip("\n")
    allowed = ", ".join(label_sets[criterion_type])
    return (
        "Patient note (sentences are numbered by their original sentence IDs):\n"
        f"{note}\n\n"
        f"Criterion type: {criterion_type}\n"
        f"Criterion: {item['criterion_text']}\n\n"
        f"Allowed labels for this criterion type: {allowed}\n"
        "Return exactly one JSON object with exactly these keys:\n"
        '  "label": one allowed label string\n'
        '  "evidence_ids": a list of integer patient sentence IDs; '
        "use an empty list when there is no supporting evidence\n"
        '  "explanation": a short explanation grounded only in the patient note\n'
        "Do not add any other keys or text."
    )


def build_messages(
    item: dict[str, Any],
    label_sets: dict[str, list[str]],
    target: str | None = None,
) -> list[dict[str, str]]:
    """Build chat messages for the prompt and optional assistant target."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(item, label_sets)},
    ]
    if target is not None:
        messages.append({"role": "assistant", "content": target})
    return messages


def format_example(
    tokenizer,
    item: dict[str, Any],
    label_sets: dict[str, list[str]],
) -> dict[str, str]:
    """Return prompt, target, and full chat-formatted text for one example."""
    assert_no_leakage_fields(item)
    evidence_ids = [int(value) for value in item["expert_sentences"]]
    target = build_target(item["expert_eligibility"], evidence_ids)
    messages = build_messages(item, label_sets, target)
    prompt_text = tokenizer.apply_chat_template(
        messages[:-1],
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    if not full_text.startswith(prompt_text):
        raise PromptFormatError(
            "chat template full text does not start with the generation prompt"
        )
    return {
        "prompt_text": prompt_text,
        "target_text": target,
        "full_text": full_text,
    }


def tokenize_ids(tokenizer, text: str) -> list[int]:
    """Tokenize without adding special tokens."""
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def tokenize_with_offsets(tokenizer, text: str) -> tuple[list[int], list[tuple[int, int]]]:
    """Tokenize with character offsets; the tokenizer must be fast."""
    if not getattr(tokenizer, "is_fast", False):
        raise PromptFormatError("a fast tokenizer is required for offset mapping")
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    return encoded["input_ids"], encoded["offset_mapping"]


def analyze_truncation(
    tokenizer,
    *,
    prompt_text: str,
    full_text: str,
    note_text: str,
    sentences: list[Sentence],
    evidence_ids: list[int],
    max_length: int,
) -> dict[str, Any]:
    """Check whether right truncation would remove evidence or target tokens."""
    prompt_ids = tokenize_ids(tokenizer, prompt_text)
    full_ids, offsets = tokenize_with_offsets(tokenizer, full_text)
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise PromptFormatError("tokenized prompt is not a prefix of tokenized full text")

    prompt_length = len(prompt_ids)
    full_length = len(full_ids)
    target_length = full_length - prompt_length
    note_start = full_text.find(note_text)
    if note_start < 0:
        raise PromptFormatError("patient note was not found verbatim in the prompt")

    evidence_truncated = False
    if full_length > max_length:
        token_end_by_sentence: dict[int, int] = {}
        for sentence in sentences:
            absolute_start = note_start + sentence.start
            absolute_end = note_start + sentence.end
            indices = [
                index
                for index, (token_start, token_end) in enumerate(offsets)
                if token_end > absolute_start and token_start < absolute_end
            ]
            token_end_by_sentence[sentence.sentence_id] = (
                max(indices) if indices else -1
            )
        for sentence_id in evidence_ids:
            token_end = token_end_by_sentence.get(sentence_id, -1)
            if token_end < 0 or token_end >= max_length:
                evidence_truncated = True

    result = {
        "prompt_tokens": prompt_length,
        "target_tokens": target_length,
        "full_tokens": full_length,
        "max_length": max_length,
        "fits": full_length <= max_length,
        "prompt_truncated": prompt_length > max_length,
        "target_truncated": prompt_length + target_length > max_length,
        "evidence_truncated": evidence_truncated,
    }
    result["evidence_preserved"] = not result["evidence_truncated"]
    result["safe"] = result["fits"] and result["evidence_preserved"]
    return result


def chat_template_hash(tokenizer) -> str:
    """Hash the tokenizer chat template for reproducibility metadata."""
    template = getattr(tokenizer, "chat_template", None)
    if template is None:
        return "none"
    return hashlib.sha256(str(template).encode()).hexdigest()
