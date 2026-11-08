"""Single-example greedy inference with the shared training prompt format.

Task 3 adds explicit decode-path selection via ``decoding_config["decoding_impl"]``:

- ``"generate"`` (default): the standard ``model.generate`` path, unchanged
  from before Task 3;
- ``"cache"``: explicit prefill + incremental KV-cached decode loop
  (``eval.decoding.greedy_generate_cached``);
- ``"recompute"``: reference loop with no cache reuse
  (``eval.decoding.greedy_generate_recompute``).

All three share the same greedy semantics (argmax over fp32 logits, eos
stopping on ``<eos>``/``<end_of_turn>``) and only the decoding internals
differ, so results can be compared token-for-token.
"""

from __future__ import annotations

from typing import Any

import torch

from trailo.eval.decoding import (
    generation_eos_token_ids,
    greedy_generate_cached,
    greedy_generate_recompute,
)
from trailo.eval.parsing import parse_and_validate


def _generation_eos_token_ids(model, tokenizer) -> list[int]:
    """Backwards-compatible alias; moved to trailo.eval.decoding."""
    return generation_eos_token_ids(model, tokenizer)


def _encode_prompt(model, tokenizer, example: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    if "prompt" not in example:
        raise ValueError("example must contain a prepared 'prompt' field")
    if "criterion_type" not in example or "n_sentences" not in example:
        raise ValueError("example must contain criterion_type and n_sentences")
    device = next(model.parameters()).device
    encoded = tokenizer(
        example["prompt"],
        return_tensors="pt",
        add_special_tokens=False,
    )
    return encoded["input_ids"].to(device), encoded["attention_mask"].to(device)


def _decode_with_generate(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    eos_token_ids: list[int],
    pad_token_id: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    with torch.inference_mode():
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
            num_beams=1,
            eos_token_id=eos_token_ids,
            pad_token_id=pad_token_id,
            use_cache=True,
        )
    new_tokens = generated[0, input_ids.shape[1] :]
    return {"token_ids": new_tokens.tolist(), "decoding_impl": "generate"}


def generate_prediction(
    model,
    tokenizer,
    example: dict[str, Any],
    decoding_config: dict[str, Any],
) -> dict[str, Any]:
    """Generate one prediction using the prepared prompt and strict parsing."""
    if decoding_config.get("do_sample", False):
        raise ValueError("this stage only supports greedy decoding")
    if int(decoding_config.get("num_beams", 1)) != 1:
        raise ValueError("this stage only supports single-beam greedy decoding")

    decode_impl = str(decoding_config.get("decoding_impl", "generate")).lower()
    if decode_impl not in {"generate", "cache", "recompute"}:
        raise ValueError(f"unknown decoding_impl {decode_impl!r}")

    input_ids, attention_mask = _encode_prompt(model, tokenizer, example)
    eos_token_ids = generation_eos_token_ids(model, tokenizer)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = eos_token_ids[0]
    max_new_tokens = int(decoding_config["max_new_tokens"])

    if decode_impl == "generate":
        decoded = _decode_with_generate(
            model, input_ids, attention_mask, eos_token_ids, pad_token_id, max_new_tokens
        )
        new_token_ids = decoded["token_ids"]
        impl_info: dict[str, Any] = {"decoding_impl": "generate"}
    else:
        generated_fn = greedy_generate_cached if decode_impl == "cache" else greedy_generate_recompute
        result = generated_fn(
            model,
            input_ids,
            attention_mask,
            max_new_tokens,
            eos_token_ids,
        )
        new_token_ids = result.generated_ids
        impl_info = {"decoding_impl": decode_impl, **result.to_dict()}

    raw_text = tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()
    label_sets = decoding_config["label_sets"]
    result = parse_and_validate(
        raw_text,
        criterion_type=example["criterion_type"],
        n_sentences=int(example["n_sentences"]),
        allowed_labels=label_sets[example["criterion_type"]],
    )
    stopped_on_eos = bool(new_token_ids) and new_token_ids[-1] in set(eos_token_ids)
    return {
        "raw_text": raw_text,
        "prediction": result.prediction,
        "valid": result.valid,
        "errors": result.errors,
        "warnings": result.warnings,
        "prompt_tokens": int(input_ids.shape[1]),
        "generated_tokens": len(new_token_ids),
        "stopped_on_eos": stopped_on_eos,
        "decoding_timing": impl_info,
        "decoding": {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "num_beams": 1,
            "decoding_impl": decode_impl,
            "eos_token_ids": eos_token_ids,
        },
    }
