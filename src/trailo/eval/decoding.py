"""Explicit KV-cache-aware and no-cache greedy decoding paths for Task 3.

Two decode implementations with identical stopping and tie-breaking semantics:

- ``greedy_generate_cached``: one full-prompt prefill forward with an updated
  ``DynamicCache`` (or ``DynamicSlidingWindowLayer`` layers for Gemma-style
  hybrid attention); every following step forwards only the newest token and
  reuses the key/value cache.
- ``greedy_generate_recompute``: a reference loop with ``use_cache=False`` that
  re-forwards the entire sequence so far at every step. It never reuses any
  cache, so per-step cost grows with total sequence length.

Both paths deliberately manage ``position_ids`` and the attention mask
explicitly instead of relying on library defaults, and they reproduce the
``model.generate`` conventions that matter for greedy decoding:

- eos handling: the eos token is appended to the output and generation stops
  without forwarding it back into the model (the loop exits before the next
  forward);
- token selection: ``argmax`` over last-position logits cast to ``float32``,
  matching the cast ``generate`` applies before its selection operators.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

import psutil
import torch

# Sampling interval for the RSS peak tracker; small enough to catch the
# transient cache-concat allocations of a single decode step.
_MEMORY_POLL_INTERVAL_S = 0.004


def generation_eos_token_ids(model, tokenizer) -> list[int]:
    """Return the eos token ids that stop generation (eos + <end_of_turn>)."""
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


# Backwards-compatible alias (the helper used to live in eval.inference).
_generation_eos_token_ids = generation_eos_token_ids


class PeakMemoryTracker:
    """Measure peak memory during decoding.

    On CPU this polls the process RSS in a background thread because peak
    transient allocations (per-step cache concatenation) are not otherwise
    visible. On CUDA it uses ``torch.cuda.max_memory_allocated``.
    """

    def __init__(self, device: torch.device, interval_s: float = _MEMORY_POLL_INTERVAL_S):
        self.device = device
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._peak_bytes = 0
        self._baseline_bytes = 0

    def start(self) -> "PeakMemoryTracker":
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            self._baseline_bytes = 0
            self._peak_bytes = torch.cuda.memory_allocated()
        else:
            self._baseline_bytes = self._current_cpu_bytes()
            self._peak_bytes = self._baseline_bytes
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        return self

    def _current_cpu_bytes(self) -> int:
        return int(psutil.Process().memory_info().rss)

    def _poll_loop(self) -> None:
        while not self._stop.is_set():
            self._peak_bytes = max(self._peak_bytes, self._current_cpu_bytes())
            self._stop.wait(self.interval_s)

    def stop(self) -> dict[str, float]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self.device.type == "cuda":
            peak = torch.cuda.max_memory_allocated()
        else:
            self._peak_bytes = max(self._peak_bytes, self._current_cpu_bytes())
            peak = self._peak_bytes
        return {
            "peak_bytes": float(peak),
            "baseline_bytes": float(self._baseline_bytes),
            "peak_delta_bytes": float(peak - self._baseline_bytes),
        }


def positions_from_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    """Compute position ids from a 2D mask using the ``generate`` convention.

    Positions are the cumulative count of attended tokens so far, with padded
    positions pinned to 1. For an unpadded single-example prompt this is
    ``arange(seq_len)``, and when one token is appended per step the new
    position is ``past_length``. Because rotary embeddings are applied inside
    the forward using these ids, the ids must match the cache logical length
    (``get_seq_length``) even when sliding-window cache layers have trimmed
    their stored keys.
    """
    positions = attention_mask.long().cumsum(-1) - 1
    positions.masked_fill_(attention_mask == 0, 1)
    return positions


def cache_memory_bytes(cache) -> dict[str, int]:
    """Report the stored key/value tensors of a ``Cache``.

    ``logical_bytes`` counts every stored element; ``storage_bytes`` counts the
    backing storages. Sliding-window cache layers keep a view over their last
    concatenated buffer, so storage retention can exceed the logical element
    count; both numbers are reported and their gap explained in the report.
    """
    logical = 0
    storage = 0
    for layer in getattr(cache, "layers", []):
        for name in ("keys", "values"):
            tensor = getattr(layer, name, None)
            if isinstance(tensor, torch.Tensor):
                logical += tensor.numel() * tensor.element_size()
                storage += tensor.untyped_storage().nbytes()
    return {"logical_bytes": logical, "storage_bytes": storage}


@dataclass
class DecodeResult:
    """Output of one greedy decode path invocation."""

    generated_ids: list[int] = field(default_factory=list)
    # The prefill forward is timed separately from the per-token loop. The
    # first token of ``generated_ids`` comes from the prefill argmax.
    prefill_seconds: float = 0.0
    decode_seconds: float = 0.0
    peak_memory_gb: float = 0.0
    memory_baseline_gb: float = 0.0
    memory_delta_gb: float = 0.0
    cache_logical_bytes: int = 0
    cache_storage_bytes: int = 0
    # log-probability margin between the argmax token and the runner-up at
    # every selection point (prefill included) when record_margins=True.
    logprob_margins: list[float] | None = None

    @property
    def generated_tokens(self) -> int:
        return len(self.generated_ids)

    @property
    def decode_tokens_per_second(self) -> float:
        """Loop-step rate: generated steps / loop time (excludes prefill)."""
        decode_steps = max(self.generated_tokens - 1, 0)
        if self.decode_seconds <= 0:
            return 0.0
        return decode_steps / self.decode_seconds

    @property
    def overall_tokens_per_second(self) -> float:
        total = self.prefill_seconds + self.decode_seconds
        if total <= 0:
            return 0.0
        return self.generated_tokens / total

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_tokens": self.generated_tokens,
            "prefill_seconds": round(self.prefill_seconds, 6),
            "decode_seconds": round(self.decode_seconds, 6),
            "decode_tokens_per_second": round(self.decode_tokens_per_second, 4),
            "overall_tokens_per_second": round(self.overall_tokens_per_second, 4),
            "peak_memory_gb": round(self.peak_memory_gb, 4),
            "memory_baseline_gb": round(self.memory_baseline_gb, 4),
            "memory_delta_gb": round(self.memory_delta_gb, 4),
            "cache_logical_bytes": self.cache_logical_bytes,
            "cache_storage_bytes": self.cache_storage_bytes,
        }


def _greedy_select(last_logits: torch.Tensor, record_margin: bool = False) -> tuple[int, float | None]:
    """Greedy argmax with the ``generate`` conventions.

    Logits are cast to fp32 (as ``generate`` does before selection). The full
    log-softmax (and with it the argmax/runner-up margin) is computed only when
    the caller records diagnostics, because a full-vocab softmax costs several
    milliseconds per step on a 262k vocabulary.
    """
    logits = last_logits.to(torch.float32)
    token_id = int(logits.argmax(dim=-1).item())
    if not record_margin:
        return token_id, None
    logprobs = torch.log_softmax(logits, dim=-1)
    top2 = torch.topk(logprobs, k=2, dim=-1)
    margin = float((top2.values[..., 0] - top2.values[..., 1]).item())
    return token_id, margin


def greedy_generate_cached(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    max_new_tokens: int,
    eos_token_ids: set[int] | list[int],
    *,
    record_margins: bool = False,
    track_memory: bool = True,
    track_cache: bool = True,
) -> DecodeResult:
    """Prefill on the full prompt, then decode one token at a time using a KV cache.

    Position handling: positions come from the attention mask (cumulative
    attended count); for a single unpadded example this is ``arange``. Keys and
    values are written to the cache after rotary embeddings are applied, so
    cached entries already carry their own positions and nothing cached is ever
    re-rotated; a decode step's new position equals the cache logical length
    (``get_seq_length``), which is correct even when sliding-window layers have
    trimmed their stored keys.
    """
    device = input_ids.device
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    eos_set = {int(value) for value in eos_token_ids}
    result = DecodeResult(logprob_margins=[] if record_margins else None)
    tracker = PeakMemoryTracker(device) if track_memory else None

    cache: Any = None
    if tracker is not None:
        tracker.start()
    with torch.inference_mode():
        try:
            # ---- Prefill: full prompt, cache created and filled here --------
            prefill_start = time.perf_counter()
            positions = positions_from_mask(attention_mask)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=positions,
                use_cache=True,
                logits_to_keep=1,
            )
            result.prefill_seconds = time.perf_counter() - prefill_start
            cache = outputs.past_key_values
            token_id, margin = _greedy_select(outputs.logits[:, -1], record_margin=record_margins)
            if margin is not None:
                result.logprob_margins.append(margin)
            result.generated_ids.append(token_id)
            stopped = token_id in eos_set

            # ---- Decode loop: one token at a time, cache updated in place --
            loop_start = time.perf_counter()
            while len(result.generated_ids) < int(max_new_tokens) and not stopped:
                # Forward the newest token only. The cache has length P + t,
                # the new token is at position P + t, and the mask grows by one
                # column on the right (batch of one, no padding).
                seq_mask = torch.cat(
                    [attention_mask, torch.ones((1, 1), device=device, dtype=attention_mask.dtype)], dim=1
                )
                outputs = model(
                    input_ids=torch.tensor([[result.generated_ids[-1]]], device=device),
                    attention_mask=seq_mask,
                    position_ids=positions_from_mask(seq_mask)[:, -1:],
                    past_key_values=cache,
                    use_cache=True,
                    logits_to_keep=1,
                )
                attention_mask = seq_mask
                if track_cache and cache is not None:
                    sizes = cache_memory_bytes(cache)
                    result.cache_logical_bytes = max(result.cache_logical_bytes, sizes["logical_bytes"])
                    result.cache_storage_bytes = max(result.cache_storage_bytes, sizes["storage_bytes"])
                token_id, margin = _greedy_select(outputs.logits[:, -1], record_margin=record_margins)
                if margin is not None:
                    result.logprob_margins.append(margin)
                result.generated_ids.append(token_id)
                stopped = token_id in eos_set
            result.decode_seconds = time.perf_counter() - loop_start
            if track_cache and cache is not None:
                sizes = cache_memory_bytes(cache)
                result.cache_logical_bytes = max(result.cache_logical_bytes, sizes["logical_bytes"])
                result.cache_storage_bytes = max(result.cache_storage_bytes, sizes["storage_bytes"])
        finally:
            if tracker is not None:
                memory = tracker.stop()
                result.peak_memory_gb = memory["peak_bytes"] / (1024**3)
                result.memory_baseline_gb = memory["baseline_bytes"] / (1024**3)
                result.memory_delta_gb = memory["peak_delta_bytes"] / (1024**3)
    return result


def greedy_generate_recompute(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    max_new_tokens: int,
    eos_token_ids: set[int] | list[int],
    *,
    record_margins: bool = False,
    track_memory: bool = True,
) -> DecodeResult:
    """Reference path: no cache reuse; every step re-forwards the whole prefix.

    Each step feeds ``prompt + generated tokens so far`` with ``use_cache=False``
    and reads the last-position logits. The computation per step is
    mathematically identical to the cached path (same positions, same causal
    mask) but every key/value projection is recomputed, so per-step cost grows
    with total sequence length and tiny kernel-level rounding may surface as an
    occasional argmax flip on near-tied logits.
    """
    device = input_ids.device
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    eos_set = {int(value) for value in eos_token_ids}
    result = DecodeResult(logprob_margins=[] if record_margins else None)
    tracker = PeakMemoryTracker(device) if track_memory else None

    if tracker is not None:
        tracker.start()
    with torch.inference_mode():
        try:
            # ---- Prefill: identical forward, but nothing is stored ----------
            prefill_start = time.perf_counter()
            positions = positions_from_mask(attention_mask)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=positions,
                use_cache=False,
                logits_to_keep=1,
            )
            result.prefill_seconds = time.perf_counter() - prefill_start
            token_id, margin = _greedy_select(outputs.logits[:, -1], record_margin=record_margins)
            if margin is not None:
                result.logprob_margins.append(margin)
            result.generated_ids.append(token_id)
            stopped = token_id in eos_set
            # The next forward must condition on the prompt plus every token
            # selected so far, otherwise the loop would re-select the prefill
            # continuation forever.
            sequence = torch.cat([input_ids, torch.tensor([[token_id]], device=device)], dim=1)

            # ---- Decode loop: full re-forward of everything so far ----------
            loop_start = time.perf_counter()
            while len(result.generated_ids) < int(max_new_tokens) and not stopped:
                # Sequence so far has P + t tokens; the last position logits
                # predict token t + 1. Nothing is cached or reused.
                outputs = model(
                    input_ids=sequence,
                    attention_mask=torch.ones_like(sequence),
                    position_ids=positions_from_mask(torch.ones_like(sequence)),
                    use_cache=False,
                    logits_to_keep=1,
                )
                token_id, margin = _greedy_select(outputs.logits[:, -1], record_margin=record_margins)
                if margin is not None:
                    result.logprob_margins.append(margin)
                result.generated_ids.append(token_id)
                stopped = token_id in eos_set
                if not stopped:
                    sequence = torch.cat([sequence, torch.tensor([[token_id]], device=device)], dim=1)
            result.decode_seconds = time.perf_counter() - loop_start
        finally:
            if tracker is not None:
                memory = tracker.stop()
                result.peak_memory_gb = memory["peak_bytes"] / (1024**3)
                result.memory_baseline_gb = memory["baseline_bytes"] / (1024**3)
                result.memory_delta_gb = memory["peak_delta_bytes"] / (1024**3)
    return result
