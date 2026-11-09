"""Custom greedy speculative decoding (Task 5).

Two-model loop that is mathematically equivalent to target-only greedy
decoding: a smaller draft model proposes up to ``gamma`` tokens, and ONE
target-model forward verifies the whole block at once. Tokens are accepted
left-to-right while the target's greedy argmax matches the draft; at the
first mismatch the target's own argmax token replaces the rejected draft
token and the rest of the block is discarded. Because every emitted token is
either a draft token verified equal to the target argmax or the target
argmax itself, the output equals the target's greedy sequence exactly (for
the mathematical function; bf16 kernel-order rounding can still surface as
benign near-tie argmax flips, which the agreement diagnostics in the
benchmark separate from structural bugs, exactly as in Task 3).

Cache-state bookkeeping (the hairiest part of a correct implementation):

- Both models keep one persistent KV cache and prefill the prompt once each.
- Invariant at every round start: each cache holds ``prompt + all confirmed
  output tokens except the pending token``. The pending token is the last
  target-emitted token; its KV entry does not exist yet, and it is consumed
  as the FIRST token fed to both models each round (the draft feeds it to
  propose the first candidate, the verification forward feeds it alongside
  the candidate block).
- The draft phase feeds at most ``gamma`` tokens (pending + candidates
  1..gamma-1), so the draft cache is one entry shorter than the target cache
  after a full-acceptance round - which is exactly why the pending token is
  NOT stored: the natural cache lengths after cropping match the invariant
  for both models without any disprefetch forward.
- After verification, ``DynamicCache.crop`` rolls each cache back to
  ``prompt + len(generated so far) - 1``. A mismatch round removes the KV of
  the rejected candidates; a full-acceptance round removes the KV of the
  (now pending) last accepted candidate re-fed next round.

Equality protocol note (measured on the real E4B/E2B pair, A100, bf16): the
custom loop is the exact target greedy sequence by construction, and the
tiny-model test suite verifies token-exactness end to end. On the real
checkpoint pair, occasional generated positions are bf16 knife edges - the
top-2 log-prob margin is exactly 0.0 (bitwise-equal raw logits) in the
single-token baseline forward. Forensics on the first observed case
(validation prompt 0, output slot 14, tokens 236764 vs 84123):

- cached single-token row (the baseline/generate shape): exact bf16 tie,
  28.5/28.5 (top-2 log-prob margin 0.0);
- batched verification row (this loop's shape): 28.5 vs 28.375 - one bf16
  ULP toward 236764 - and the full-recompute row agrees;
- fp32 (identical through all three shapes and candidate-independent):
  236764 wins by 1.99e-2, i.e. 0.07 percent of the logit magnitude - BELOW
  one bf16 ULP, so bf16 provably cannot resolve this pair and the baseline
  row's tie is a rounding dead heat the tie-break happens to resolve toward
  84123; the verification row lands on the fp32-accurate token;
- HF's library assisted generation (``assistant_model``), which uses the
  same batched-verification paradigm, emits THE SAME stream as this loop
  (..., 236770, 236764, 236800, 84123, ...), so the divergence belongs to
  the paradigm on this checkpoint, not to this implementation;
- the two streams re-join immediately after the flip (the flipped tokens
  are inserted, then both continuations merge again at 84123).

The benchmark therefore reports (a) token equality on all healthy-margin
positions, (b) per-divergence knife-edge classification (margins of the
verify row and of the baseline at the flip slot), and (c) custom-loop vs
HF-assisted stream equality as the paradigm control.

Synchronization: every phase boundary calls ``torch.cuda.synchronize`` so
the draft / verify / prefill wall-split is accurate rather than measuring
only kernel-launch latency.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import torch

from trailo.eval.decoding import (
    DecodeResult,
    PeakMemoryTracker,
    generation_eos_token_ids,
    greedy_generate_cached,
    positions_from_mask,
)


def argmax_token(logits_row: torch.Tensor) -> int:
    """Greedy argmax with the ``generate`` fp32-cast convention."""
    return int(logits_row.to(torch.float32).argmax(dim=-1).item())


def top2_margin(logits_row: torch.Tensor) -> float:
    """Log-prob margin between the argmax token and the runner-up (fp32)."""
    logprobs = torch.log_softmax(logits_row.to(torch.float32), dim=-1)
    values = torch.topk(logprobs, k=2, dim=-1).values
    return float((values[..., 0] - values[..., 1]).item())


@dataclass
class RoundRecord:
    """One draft+verify round."""

    drafted: int = 0  # candidates the draft proposed this round (may be < gamma)
    accepted: int = 0  # draft candidates verified equal to the target argmax
    appended: int = 0  # output tokens appended this round (accepted + replacement)
    replaced: bool = False  # True when a mismatch produced a replacement token
    draft_eos: bool = False  # a draft candidate was eos (drafting stopped early)
    draft_seconds: float = 0.0
    verify_seconds: float = 0.0
    verify_positions: int = 0  # tokens fed to the target in the verify forward
    accepted_min_margin: float | None = None  # requires record_margins=True
    replaced_margin: float | None = None

    @property
    def acceptance_ratio(self) -> float:
        return self.accepted / self.drafted if self.drafted else 0.0


@dataclass
class SpeculativeResult:
    """Full result of one speculative decoding invocation."""

    generated_ids: list[int] = field(default_factory=list)
    gamma: int = 0
    prefill_seconds: float = 0.0  # target prefill + draft prefill
    target_prefill_seconds: float = 0.0
    draft_prefill_seconds: float = 0.0
    draft_seconds: float = 0.0  # summed across rounds
    verify_seconds: float = 0.0  # summed across rounds
    total_seconds: float = 0.0
    rounds: list[RoundRecord] = field(default_factory=list)
    peak_memory_gb: float = 0.0
    memory_baseline_gb: float = 0.0
    memory_delta_gb: float = 0.0
    target_cache_logical_bytes: int = 0
    draft_cache_logical_bytes: int = 0
    stopped_on_eos: bool = False
    truncated_to_max_new: bool = False

    @property
    def drafted_total(self) -> int:
        return sum(round_.drafted for round_ in self.rounds)

    @property
    def accepted_total(self) -> int:
        return sum(round_.accepted for round_ in self.rounds)

    @property
    def acceptance_rate(self) -> float:
        """Matched draft tokens / drafted tokens."""
        return self.accepted_total / self.drafted_total if self.drafted_total else 0.0

    @property
    def round_count(self) -> int:
        return len(self.rounds)

    @property
    def tokens_per_target_forward(self) -> float:
        """Emitted round tokens / verification forwards (the speed lever)."""
        round_tokens = len(self.generated_ids) - 1  # first came from the prefill
        return round_tokens / self.round_count if self.round_count else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_tokens": len(self.generated_ids),
            "gamma": self.gamma,
            "round_count": self.round_count,
            "drafted_total": self.drafted_total,
            "accepted_total": self.accepted_total,
            "acceptance_rate": round(self.acceptance_rate, 4),
            "tokens_per_target_forward": round(self.tokens_per_target_forward, 4),
            "prefill_seconds": round(self.prefill_seconds, 6),
            "draft_seconds": round(self.draft_seconds, 6),
            "verify_seconds": round(self.verify_seconds, 6),
            "total_seconds": round(self.total_seconds, 6),
            "overall_tokens_per_second": round(
                len(self.generated_ids) / self.total_seconds, 4
            )
            if self.total_seconds > 0
            else 0.0,
            "peak_memory_gb": round(self.peak_memory_gb, 4),
            "memory_baseline_gb": round(self.memory_baseline_gb, 4),
            "memory_delta_gb": round(self.memory_delta_gb, 4),
            "target_cache_logical_bytes": self.target_cache_logical_bytes,
            "draft_cache_logical_bytes": self.draft_cache_logical_bytes,
            "stopped_on_eos": self.stopped_on_eos,
            "truncated_to_max_new": self.truncated_to_max_new,
            "rounds": [
                {
                    "drafted": r.drafted,
                    "accepted": r.accepted,
                    "appended": r.appended,
                    "replaced": r.replaced,
                    "draft_eos": r.draft_eos,
                    "draft_seconds": round(r.draft_seconds, 6),
                    "verify_seconds": round(r.verify_seconds, 6),
                    "verify_positions": r.verify_positions,
                    "acceptance_ratio": round(r.acceptance_ratio, 4),
                    "accepted_min_margin": None
                    if r.accepted_min_margin is None
                    else round(r.accepted_min_margin, 6),
                    "replaced_margin": None
                    if r.replaced_margin is None
                    else round(r.replaced_margin, 6),
                }
                for r in self.rounds
            ],
        }


def cache_logical_bytes(cache) -> int:
    """Elements * dtype size over every stored layer (Task 3 convention)."""
    total = 0
    for layer in getattr(cache, "layers", []):
        for name in ("keys", "values"):
            tensor = getattr(layer, name, None)
            if isinstance(tensor, torch.Tensor):
                total += tensor.numel() * tensor.element_size()
    return total


def activate_past_recording(cache) -> None:
    """Enable rollback on sliding-window cache layers (transformers >= 5).

    ``DynamicSlidingWindowLayer`` keeps only the last ``sliding_window - 1``
    keys once the window is reached, so a later rollback cannot reconstruct
    the window content from the trimmed buffer. ``activate_past_recording``
    makes the layer retain newly arrived (appended) keys until the next
    ``crop``, which is exactly the append-then-rollback pattern used here;
    after ``crop(-removal)`` the buffer is pruned back to the minimal
    working window plus nothing older than what the attention would need.
    Call this once per cache, right after prefill.
    """
    for layer in getattr(cache, "layers", []):
        activate = getattr(layer, "activate_past_recording", None)
        if callable(activate):
            activate()


def rollback_cache(cache, keep_len: int) -> int:
    """Roll a layer-based cache back to ``keep_len`` logical entries.

    Per-layer because layer lengths can differ once sliding-window layers
    have trimmed their buffers (logical length vs stored length). The
    negative-``crop`` convention of this transformers version removes
    ``removal`` newest entries; plain and below-window layers slice their
    stored buffer, recorded sliding-window layers prune the appended tail.
    Returns the total number of removed logical entries.
    """
    removed = 0
    layers = getattr(cache, "layers", None)
    if not layers:
        current = int(cache.get_seq_length())
        removal = current - keep_len
        if removal > 0:
            cache.crop(-removal)
            removed = removal
        return removed
    for layer in layers:
        layer_len = int(layer.get_seq_length())
        removal = layer_len - keep_len
        if removal <= 0:
            continue
        layer.crop(-removal)
        removed += removal
    return removed


def _single_token_forward(
    model,
    cache,
    token_id: int,
    seq_mask: torch.Tensor,
    seq_pos: torch.Tensor,
):
    """Feed one token through ``model`` with an explicit mask/position id."""
    outputs = model(
        input_ids=torch.tensor([[token_id]], device=seq_mask.device),
        attention_mask=seq_mask,
        position_ids=seq_pos,
        past_key_values=cache,
        use_cache=True,
        logits_to_keep=1,
    )
    return outputs


def speculative_generate_greedy(
    target,
    draft,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    max_new_tokens: int,
    eos_token_ids: set[int] | list[int],
    *,
    gamma: int = 4,
    record_margins: bool = False,
    track_memory: bool = True,
    track_cache: bool = True,
    synchronize: bool = True,
) -> SpeculativeResult:
    """Greedy speculative decoding: draft ``gamma``, verify with one target forward.

    Batch size 1, all-ones mask (no padding), matching every other decode
    path in this repository. See the module docstring for the cache-state
    invariant and why the output is the exact target greedy sequence.
    """
    if int(gamma) < 1:
        raise ValueError("gamma must be >= 1")
    gamma = int(gamma)
    device = input_ids.device
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    eos_set = {int(value) for value in eos_token_ids}

    result = SpeculativeResult(gamma=gamma)
    tracker = PeakMemoryTracker(device) if track_memory else None

    def _sync() -> None:
        if synchronize and device.type == "cuda":
            torch.cuda.synchronize()

    if tracker is not None:
        tracker.start()
    start_wall = time.perf_counter()
    with torch.inference_mode():
        try:
            prompt_len = int(input_ids.shape[1])

            # ---- Target prefill ------------------------------------------------
            _sync()
            t0 = time.perf_counter()
            positions = positions_from_mask(attention_mask)
            outputs = target(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=positions,
                use_cache=True,
                logits_to_keep=1,
            )
            target_cache = outputs.past_key_values
            _sync()
            result.target_prefill_seconds = time.perf_counter() - t0
            pending = argmax_token(outputs.logits[:, -1])
            result.generated_ids.append(pending)
            stopped = pending in eos_set

            # ---- Draft prefill: its own KV of the prompt -----------------------
            if not stopped and len(result.generated_ids) < int(max_new_tokens):
                _sync()
                t0 = time.perf_counter()
                d_outputs = draft(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=positions,
                    use_cache=True,
                    logits_to_keep=1,
                )
                draft_cache = d_outputs.past_key_values
                _sync()
                result.draft_prefill_seconds = time.perf_counter() - t0
            else:
                draft_cache = None
            result.prefill_seconds = (
                result.target_prefill_seconds + result.draft_prefill_seconds
            )

            cache_t_len = lambda: int(target_cache.get_seq_length())  # noqa: E731

            # From here on every cache must support rollback: recorded
            # sliding-window layers can reconstruct their window after a crop.
            if target_cache is not None:
                activate_past_recording(target_cache)
            if draft_cache is not None:
                activate_past_recording(draft_cache)
            # ---- Draft + verify rounds -----------------------------------------
            while (
                not stopped
                and draft_cache is not None
                and len(result.generated_ids) < int(max_new_tokens)
            ):
                cache_len = cache_t_len()  # prompt + confirmed-not-pending length

                # -- Draft phase: feed pending, then candidates, up to gamma feeds.
                # Batch of 1, no padding: the k-th feed of the round has
                # mask length cache_len + k + 1 and position cache_len + k,
                # derived from the cache length (which reflects the rollback
                # of the previous round), never from a running mask.
                _sync()
                draft_t0 = time.perf_counter()
                candidates: list[int] = []
                current = pending
                for round_step in range(gamma):
                    seq_mask = torch.ones(
                        (1, cache_len + round_step + 1),
                        device=device,
                        dtype=attention_mask.dtype,
                    )
                    # One new token at position cache_len + round_step.
                    seq_pos = torch.tensor(
                        [[cache_len + round_step]], device=device, dtype=torch.long
                    )
                    d_out = _single_token_forward(
                        draft, draft_cache, current, seq_mask, seq_pos
                    )
                    candidate = argmax_token(d_out.logits[:, -1])
                    candidates.append(candidate)
                    current = candidate
                    if candidate in eos_set:
                        break
                _sync()
                round_draft_seconds = time.perf_counter() - draft_t0
                gamma_eff = len(candidates)

                # -- Verify: ONE target forward over [pending, candidates...].
                _sync()
                verify_t0 = time.perf_counter()
                verify_len = 1 + gamma_eff
                verify_mask = torch.ones(
                    (1, cache_len + verify_len), device=device, dtype=attention_mask.dtype
                )
                verify_pos = torch.arange(
                    cache_len, cache_len + verify_len, device=device, dtype=torch.long
                ).unsqueeze(0)
                verify_ids = torch.cat(
                    [
                        torch.tensor([[pending]], device=device),
                        torch.tensor([candidates], device=device),
                    ],
                    dim=1,
                )
                t_outputs = target(
                    input_ids=verify_ids,
                    attention_mask=verify_mask,
                    position_ids=verify_pos,
                    past_key_values=target_cache,
                    use_cache=True,
                    logits_to_keep=verify_len,
                )
                _sync()
                round_verify_seconds = time.perf_counter() - verify_t0

                if track_cache:
                    result.target_cache_logical_bytes = max(
                        result.target_cache_logical_bytes,
                        cache_logical_bytes(target_cache),
                        cache_logical_bytes(t_outputs.past_key_values),
                    )
                    result.draft_cache_logical_bytes = max(
                        result.draft_cache_logical_bytes,
                        cache_logical_bytes(draft_cache),
                    )

                # -- Accept/reject scan ------------------------------------------
                record = RoundRecord(
                    drafted=gamma_eff,
                    draft_seconds=round_draft_seconds,
                    verify_seconds=round_verify_seconds,
                    verify_positions=verify_len,
                )
                appended: list[int] = []
                accepted_count = 0
                for index in range(gamma_eff):
                    row_logits = t_outputs.logits[:, index]
                    margin = top2_margin(row_logits) if record_margins else None
                    target_token = argmax_token(row_logits)
                    if target_token == candidates[index]:
                        accepted_count += 1
                        appended.append(target_token)
                        if record_margins:
                            record.accepted_min_margin = (
                                margin
                                if record.accepted_min_margin is None
                                else min(record.accepted_min_margin, margin)
                            )
                        if target_token in eos_set:
                            record.draft_eos = True
                            stopped = True
                            break
                    else:
                        record.replaced = True
                        if record_margins:
                            record.replaced_margin = margin
                        appended.append(target_token)
                        if target_token in eos_set:
                            stopped = True
                        break
                record.accepted = accepted_count
                record.appended = len(appended)
                result.rounds.append(record)
                result.draft_seconds += round_draft_seconds
                result.verify_seconds += round_verify_seconds
                result.generated_ids.extend(appended)
                pending = result.generated_ids[-1]

                # -- Cache rollback to prompt + all-outputs-except-pending -------
                keep_len = prompt_len + len(result.generated_ids) - 1
                if target_cache is not None:
                    rollback_cache(target_cache, keep_len)
                if draft_cache is not None:
                    rollback_cache(draft_cache, keep_len)

                # -- max_new_tokens cap ------------------------------------------
                if len(result.generated_ids) > int(max_new_tokens):
                    result.generated_ids = result.generated_ids[: int(max_new_tokens)]
                    result.truncated_to_max_new = True
                    stopped = True

            result.stopped_on_eos = bool(result.generated_ids) and (
                result.generated_ids[-1] in eos_set
            )
            result.total_seconds = time.perf_counter() - start_wall
        finally:
            if tracker is not None:
                memory = tracker.stop()
                result.peak_memory_gb = memory["peak_bytes"] / (1024**3)
                result.memory_baseline_gb = memory["baseline_bytes"] / (1024**3)
                result.memory_delta_gb = memory["peak_delta_bytes"] / (1024**3)
    return result


def run_greedy_baseline(
    target,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    max_new_tokens: int,
    eos_token_ids: set[int] | list[int],
    *,
    record_margins: bool = False,
) -> DecodeResult:
    """Target-only greedy decoding baseline (the Task 3 cached path)."""
    return greedy_generate_cached(
        target,
        input_ids,
        attention_mask,
        max_new_tokens,
        eos_token_ids,
        record_margins=record_margins,
    )
