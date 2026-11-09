"""Decode-mechanism benchmark across the implemented greedy paths.

The research repo implements three token-exact greedy decode mechanisms:

- ``generate``: the standard HF ``model.generate`` greedy path;
- ``cache``: explicit prefill + incremental KV-cache decode loop
  (`trailo.eval.decoding.greedy_generate_cached`);
- ``recompute``: reference loop that re-forwards the entire sequence at every
  step (`trailo.eval.decoding.greedy_generate_recompute`).

The benchmark runs a screening workload (same note + criteria) through each
mechanism and reports wall time, generated token counts, tokens/second, peak
memory delta, and pairwise token-stream agreement per criterion. Because
``recompute`` costs O(sequence^2) per token, the default workload is a small
subset of the criteria with a bounded ``max_new_tokens``.

Speculative decoding (Task 5) requires the GPU-only E4B/E2B checkpoint pair;
the app exposes the *recorded* A100 benchmark results instead of running it
live. See `routers/benchmark.py`.
"""

from __future__ import annotations

import time
from typing import Any

DEFAULT_MAX_NEW_TOKENS = 48
DEFAULT_CRITERIA_LIMIT = 4


def run_decode_benchmark(
    engine,
    *,
    note: str,
    sentences: list[str],
    criteria: list[dict[str, str]],
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    criteria_limit: int = DEFAULT_CRITERIA_LIMIT,
) -> dict[str, Any]:
    """Run the workload once per decoding implementation.

    ``criteria`` entries are ``{"criterion_type": ..., "text": ...}``. Each
    implementation runs to completion; a failure in one implementation (e.g. an
    out-of-memory custom path) is recorded on its row and never aborts the rest.
    """
    engine.ensure_loaded()
    import torch

    from trailo.eval.decoding import PeakMemoryTracker

    impls = ["generate", "cache", "recompute"]
    limited = criteria[: max(0, int(criteria_limit))]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows: list[dict[str, Any]] = []
    streams: dict[int, dict[str, list[int]]] = {}
    parsed_labels: dict[int, dict[str, str]] = {}

    for impl in impls:
        per_criterion_seconds: list[float] = []
        total_tokens = 0
        error: str | None = None
        active_seconds = 0.0

        tracker = PeakMemoryTracker(device)
        tracker.start()
        try:
            for local_index, item in enumerate(limited):
                started = time.perf_counter()
                try:
                    result = engine.screen_criterion(
                        index=local_index,
                        criterion_type=item["criterion_type"],
                        criterion_text=item["text"],
                        note=note,
                        sentences=sentences,
                        decoding_impl=impl,
                        max_new_tokens=max_new_tokens,
                    )
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    break
                took = time.perf_counter() - started
                per_criterion_seconds.append(round(took, 3))
                active_seconds += took
                total_tokens += result.generated_tokens
                streams.setdefault(local_index, {})[impl] = result.token_ids
                parsed_labels.setdefault(local_index, {})[impl] = result.label or ""
        finally:
            memory = tracker.stop()

        rows.append(
            {
                "impl": impl,
                "wall_seconds": round(active_seconds, 3),
                "generated_tokens": total_tokens,
                "tokens_per_second": (
                    round(total_tokens / active_seconds, 2) if active_seconds > 0 else 0.0
                ),
                "peak_memory_delta_bytes": memory.get("peak_delta_bytes"),
                "per_criterion_seconds": per_criterion_seconds,
                "ok": error is None,
                "error": error,
            }
        )

    # Token-stream agreement: every implemented path is documented to emit the
    # identical greedy token stream; the benchmark makes that claim checkable
    # on the actual workload instead of asserting it.
    agreement: list[dict[str, Any]] = []
    for local_index in sorted(streams):
        per_impl = streams[local_index]
        implementations = list(per_impl)
        reference = per_impl[implementations[0]]
        first_divergence: dict[str, int | None] = {}
        for impl in implementations[1:]:
            other = per_impl[impl]
            divergence = next(
                (i for i, (a, b) in enumerate(zip(reference, other)) if a != b),
                None,
            )
            if divergence is None and len(reference) != len(other):
                divergence = min(len(reference), len(other))
            first_divergence[impl] = divergence
        label_values = set(parsed_labels[local_index].values())

        agreement.append(
            {
                "index": local_index,
                "criterion_text": limited[local_index]["text"],
                "n_impls": len(implementations),
                "all_equal": all(v is None for v in first_divergence.values()),
                "first_divergence": first_divergence,
                "stream_lengths": {impl: len(stream) for impl, stream in per_impl.items()},
                "labels_equal": len(label_values) <= 1,
                "labels": parsed_labels[local_index],
            }
        )

    return {
        "meta": {
            "device": str(device),
            "n_criteria_workload": len(limited),
            "criteria_available": len(criteria),
            "criteria_limit": criteria_limit,
            "max_new_tokens": max_new_tokens,
            "note_sentences": len(sentences),
            "impls": impls,
            "models_preset": engine.settings.model_config_name,
            "note_about_recompute": (
                "The 'recompute' path intentionally re-forwards the entire sequence "
                "at every decode step (no cache reuse), so its cost grows with "
                "sequence length squared; that difference is exactly what this "
                "benchmark makes visible on the real screening workload."
            ),
        },
        "rows": rows,
        "agreement": agreement,
    }
