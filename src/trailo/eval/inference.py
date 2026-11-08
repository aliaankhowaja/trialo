"""Single-example greedy inference with the shared training prompt format."""

from __future__ import annotations

from typing import Any

import torch

from trailo.eval.parsing import parse_and_validate


def _generation_eos_token_ids(model, tokenizer) -> list[int]:
    generation_config = getattr(model, "generation_config", None)
    configured = getattr(generation_config, "eos_token_id", None)
    if configured is None:
        configured = tokenizer.eos_token_id
    if isinstance(configured, int):
        configured = [configured]
    values = [int(value) for value in configured]
    end_of_turn = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    if isinstance(end_of_turn, int) and end_of_turn >= 0 and end_of_turn not in values:
        values.append(end_of_turn)
    return values


def generate_prediction(
    model,
    tokenizer,
    example: dict[str, Any],
    decoding_config: dict[str, Any],
) -> dict[str, Any]:
    """Generate one prediction using the prepared prompt and strict parsing.

    The function deliberately uses the standard ``generate`` path with greedy
    decoding. It does not implement custom KV caching or speculative decoding.
    """
    if "prompt" not in example:
        raise ValueError("example must contain a prepared 'prompt' field")
    if "criterion_type" not in example or "n_sentences" not in example:
        raise ValueError("example must contain criterion_type and n_sentences")
    if decoding_config.get("do_sample", False):
        raise ValueError("this stage only supports greedy decoding")
    if int(decoding_config.get("num_beams", 1)) != 1:
        raise ValueError("this stage only supports single-beam greedy decoding")

    device = next(model.parameters()).device
    encoded = tokenizer(
        example["prompt"],
        return_tensors="pt",
        add_special_tokens=False,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    eos_token_ids = _generation_eos_token_ids(model, tokenizer)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = eos_token_ids[0]

    with torch.inference_mode():
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=int(decoding_config["max_new_tokens"]),
            do_sample=False,
            num_beams=1,
            eos_token_id=eos_token_ids,
            pad_token_id=pad_token_id,
            use_cache=True,
        )
    new_tokens = generated[0, input_ids.shape[1] :]
    raw_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    label_sets = decoding_config["label_sets"]
    result = parse_and_validate(
        raw_text,
        criterion_type=example["criterion_type"],
        n_sentences=int(example["n_sentences"]),
        allowed_labels=label_sets[example["criterion_type"]],
    )
    return {
        "raw_text": raw_text,
        "prediction": result.prediction,
        "valid": result.valid,
        "errors": result.errors,
        "warnings": result.warnings,
        "prompt_tokens": int(input_ids.shape[1]),
        "generated_tokens": int(new_tokens.shape[0]),
        "decoding": {
            "max_new_tokens": int(decoding_config["max_new_tokens"]),
            "do_sample": False,
            "num_beams": 1,
            "eos_token_ids": eos_token_ids,
        },
    }
