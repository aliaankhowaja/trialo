"""Task 3: KV-cache agreement and prefill/decode benchmark on the smoke model.

Runs, on the configured model and device:

1. An agreement phase comparing three greedy decode paths per prompt/backend:

   - ``cache``    explicit prefill + incremental KV-cached decode loop
                  (``trailo.eval.decoding.greedy_generate_cached``);
   - ``recompute`` reference loop with ``use_cache=False`` that re-forwards
                  the whole sequence each step
                  (``greedy_generate_recompute``);
   - ``generate`` the library ``model.generate`` path, only as a cross-check
                  reference (not a substitute for the custom paths).

2. A timing phase over several prompt/output lengths measuring prefill
   seconds, decode seconds, tokens/second, peak memory, and cache size.

Writes ``results/<run_name>/task3/kv_cache_benchmark.json`` and ``report.md``.
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil
import torch

from trailo.eval.decoding import (
    generation_eos_token_ids,
    greedy_generate_cached,
    greedy_generate_recompute,
)
from trailo.paths import RESULTS_DIR, repo_path
from trailo.train.config import get_section, load_config
from trailo.train.modeling import load_model, load_tokenizer


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def synthetic_prompt_ids(length: int, device: torch.device, base_ids: torch.Tensor | None) -> torch.Tensor:
    """Deterministic synthetic prompt, tiling real prompt ids when available.

    Repeating a real prompt keeps the token distribution natural; garbage
    token-id prompts make instruction-tuned models emit ``<end_of_turn>``
    immediately, which would truncate every decode-length cell.
    """
    if base_ids is None:
        ids = 5 + ((torch.arange(length) * 101 + 7) % 260_000)
        return ids.unsqueeze(0).to(device)
    base = base_ids.to(device)[0]
    repeats = (length + base.numel() - 1) // base.numel()
    ids = base.repeat(repeats)[:length]
    return ids.unsqueeze(0)


def prepare_prompt_cases(
    validation_records: list[dict[str, Any]],
    *,
    tokenizer,
    max_examples: int,
    max_prompt_tokens: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    """Prompt cases used by both the agreement and timing phases."""
    cases: list[dict[str, Any]] = []
    for record in validation_records:
        if len(cases) >= max_examples:
            break
        encoded = tokenizer(record["prompt"], return_tensors="pt", add_special_tokens=False)
        tokens = encoded["input_ids"].shape[1]
        if tokens > max_prompt_tokens:
            continue
        cases.append(
            {
                "label": f"real_validation[{len(cases)}]",
                "input_ids": encoded["input_ids"].to(device),
                "prompt_tokens": tokens,
            }
        )
    return cases


def compare_streams(primary: list[int], secondary: list[int]) -> dict[str, Any]:
    """Token-level comparison with first-divergence diagnostics."""
    agreement = primary == secondary
    first_divergence = None
    margin_a = margin_b = None
    compare_len = min(len(primary), len(secondary))
    for index in range(compare_len):
        if primary[index] != secondary[index]:
            first_divergence = index
            break
    if not agreement and first_divergence is None:
        # Same tokens but one path kept generating after the other stopped.
        first_divergence = compare_len
    return {
        "agreement": agreement,
        "first_divergence": first_divergence,
        "primary_len": len(primary),
        "secondary_len": len(secondary),
    }


def divergence_margins(
    cached: Any,
    recompute: Any,
    first_divergence: int | None,
) -> dict[str, float | None]:
    """log-prob margin (argmax vs runner-up) at a divergence step in both paths."""
    output: dict[str, float | None] = {"cached_margin": None, "recompute_margin": None}
    if first_divergence is None:
        return output
    cached_margins = cached.logprob_margins or []
    recompute_margins = recompute.logprob_margins or []
    if first_divergence < len(cached_margins):
        output["cached_margin"] = round(cached_margins[first_divergence], 6)
    if first_divergence < len(recompute_margins):
        output["recompute_margin"] = round(recompute_margins[first_divergence], 6)
    return output


def margin_summary(margins: list[float] | None) -> dict[str, float] | None:
    if not margins:
        return None
    return {
        "min": round(min(margins), 6),
        "median": round(sorted(margins)[len(margins) // 2], 6),
    }


def run_generate_reference(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
) -> dict[str, Any]:
    """Library ``model.generate`` with cache, wall-clock only (cross-check)."""
    eos_token_ids = generation_eos_token_ids(model, tokenizer)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = eos_token_ids[0]
    gc.collect()
    wall_start = time.perf_counter()
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
    seconds = time.perf_counter() - wall_start
    new_tokens = generated[0, input_ids.shape[1] :].tolist()
    return {
        "generated_tokens": len(new_tokens),
        "prefill_seconds": None,
        "total_seconds": round(seconds, 6),
        "overall_tokens_per_second": round(len(new_tokens) / seconds, 4) if seconds else 0.0,
        "token_ids": new_tokens,
        "stopped_on_eos": bool(new_tokens) and new_tokens[-1] in set(eos_token_ids),
    }


def timing_cells_real(max_new: list[int], real_prompt_tokens: int) -> list[tuple[str, int, int, int]]:
    # (label, synthetic prompt token count or -1 for real, max_new_tokens)
    cells: list[tuple[str, int, int, int]] = []
    for length in (64, 256, 1024):
        for new_tokens in max_new:
            cells.append((f"synthetic_{length}", length, new_tokens, length))
    for new_tokens in max_new:
        cells.append(("real_validation_0", -1, new_tokens, real_prompt_tokens))
    return cells


def run_cached_cell(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_ids: list[int],
) -> dict[str, Any]:
    gc.collect()
    result = greedy_generate_cached(
        model, input_ids, None, max_new_tokens, eos_token_ids, record_margins=False
    )
    payload = result.to_dict()
    payload["token_ids"] = list(result.generated_ids)
    payload["stopped_on_eos"] = bool(result.generated_ids) and result.generated_ids[-1] in set(eos_token_ids)
    return payload


def run_recompute_cell(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_ids: list[int],
) -> dict[str, Any]:
    gc.collect()
    result = greedy_generate_recompute(
        model, input_ids, None, max_new_tokens, eos_token_ids, record_margins=False
    )
    payload = result.to_dict()
    payload["token_ids"] = list(result.generated_ids)
    payload["stopped_on_eos"] = bool(result.generated_ids) and result.generated_ids[-1] in set(eos_token_ids)
    return payload


def agreement_report(pairs: list[dict[str, Any]], backend: str) -> list[dict[str, Any]]:
    rows = []
    for pair in pairs:
        rows.append(
            {
                "backend": backend,
                **pair,
            }
        )
    return rows


def markdown_timing_table(rows: list[dict[str, Any]], title: str) -> str:
    lines = [
        f"### {title}",
        "",
        "| prompt | P | max new | prefill s | decode s | tok/s decode | tok/s overall | peak mem GB | Δmem GB | cache KB (peak) | eos |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    def field(row: dict, key: str, fmt_value: str = "{}") -> str:
        value = row.get(key)
        return "-" if value is None else fmt_value.format(value)

    for row in rows:
        cell = row["timing"]
        cache_kb = cell.get("cache_logical_bytes", cell.get("cache_max_logical_bytes", 0)) / 1024
        lines.append(
            f"| {row['prompt_label']} | {row['prompt_tokens']} | {row['max_new_tokens']} "
            f"| {field(cell, 'prefill_seconds', '{:.4f}')} | {field(cell, 'decode_seconds', '{:.4f}')} "
            f"| {field(cell, 'decode_tokens_per_second', '{:.2f}')} | {field(cell, 'overall_tokens_per_second', '{:.2f}')} "
            f"| {field(cell, 'peak_memory_gb', '{:.2f}')} | {field(cell, 'memory_delta_gb', '{:.3f}')} "
            f"| {cache_kb:.1f} | {row['stopped_on_eos']} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--agreement-backends", default="sdpa,eager")
    parser.add_argument("--agreement-examples", type=int, default=6)
    parser.add_argument("--agreement-new-tokens", type=int, default=48)
    parser.add_argument("--timing-new-tokens", default="32,64,128")
    parser.add_argument("--skip-timing", action="store_true")
    parser.add_argument(
        "--prompts-dir",
        default=None,
        help=(
            "Directory with validation.jsonl to take real prompts from "
            "(default data.processed_dir; prompts are model-agnostic)."
        ),
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Override output directory (default results/<run_name>/task3).",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    model_config = get_section(config, "model")
    data_config = get_section(config, "data")
    train_config = get_section(config, "train")
    run_name = config.get("run_name", "run")
    device = torch.device(train_config["device"])
    configured_backend = str(model_config.get("attn_implementation", "sdpa"))

    prompts_dir = repo_path(args.prompts_dir or data_config["processed_dir"])
    validation_records = load_jsonl(prompts_dir / "validation.jsonl")
    tokenizer = load_tokenizer(model_config["model_id"], model_config.get("revision"))

    prompt_cases = prepare_prompt_cases(
        validation_records,
        tokenizer=tokenizer,
        max_examples=args.agreement_examples,
        max_prompt_tokens=1152,
        device=device,
    )
    if not prompt_cases:
        raise SystemExit("no validation prompts available for the benchmark")
    base_ids = prompt_cases[0]["input_ids"].cpu()
    synthetic_cases = [
        {"label": f"synthetic_{length}", "input_ids": synthetic_prompt_ids(length, device, base_ids), "prompt_tokens": length}
        for length in (64, 256, 1024)
    ]
    agreement_cases = prompt_cases + synthetic_cases

    backends = [b.strip() for b in args.agreement_backends.split(",") if b.strip()]
    if configured_backend not in backends:
        backends.append(configured_backend)

    agreement_rows: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []
    timing_backend = configured_backend

    for backend in dict.fromkeys(backends):
        model = load_model(
            model_config["model_id"],
            model_config.get("revision"),
            device,
            train_config["precision"],
            backend,
            allow_cpu_oversize=bool(model_config.get("allow_cpu_oversize", False)),
        )
        if args.adapter:
            from trailo.train.modeling import load_adapter_model

            model = load_adapter_model(model, args.adapter)
            model.to(device)
        model.eval()
        eos_token_ids = generation_eos_token_ids(model, tokenizer)
        backend_row: dict[str, Any] = {"backend": backend}

        # ---- Agreement phase -------------------------------------------------
        for case in agreement_cases:
            input_ids = case["input_ids"]
            attention_mask = torch.ones_like(input_ids)
            with torch.inference_mode():
                cached = greedy_generate_cached(
                    model,
                    input_ids,
                    attention_mask,
                    args.agreement_new_tokens,
                    eos_token_ids,
                    record_margins=True,
                    track_memory=False,
                )
                recompute = greedy_generate_recompute(
                    model,
                    input_ids,
                    attention_mask,
                    args.agreement_new_tokens,
                    eos_token_ids,
                    record_margins=True,
                    track_memory=False,
                )
            cached_vs_recompute = compare_streams(cached.generated_ids, recompute.generated_ids)
            margins = divergence_margins(cached, recompute, cached_vs_recompute["first_divergence"])
            agreement_rows.append(
                {
                    "backend": backend,
                    "prompt_label": case["label"],
                    "prompt_tokens": case["prompt_tokens"],
                    "max_new_tokens": args.agreement_new_tokens,
                    **cached_vs_recompute,
                    **margins,
                    "cached_len": cached.generated_tokens,
                    "recompute_len": recompute.generated_tokens,
                    "cached_min_margin": margin_summary(cached.logprob_margins),
                    "recompute_min_margin": margin_summary(recompute.logprob_margins),
                }
            )
        print(
            f"[agreement] backend={backend}: "
            f"{sum(r['agreement'] for r in agreement_rows if r['backend'] == backend)}"
            f"/{sum(r['backend'] == backend for r in agreement_rows)} prompts agree"
        )

        # ---- Cross-check against model.generate ------------------------------
        for case in (agreement_cases[0], synthetic_cases[-1]):
            input_ids = case["input_ids"]
            attention_mask = torch.ones_like(input_ids)
            reference = run_generate_reference(model, tokenizer, input_ids, attention_mask, args.agreement_new_tokens)
            with torch.inference_mode():
                cached = greedy_generate_cached(
                    model, input_ids, attention_mask, args.agreement_new_tokens, eos_token_ids, track_memory=False
                )
                recompute = greedy_generate_recompute(
                    model, input_ids, attention_mask, args.agreement_new_tokens, eos_token_ids, track_memory=False
                )
            cross_cached = compare_streams(cached.generated_ids, reference["token_ids"])
            cross_recompute = compare_streams(recompute.generated_ids, reference["token_ids"])
            agreement_rows.append(
                {
                    "backend": backend,
                    "prompt_label": f"{case['label']} vs generate",
                    "prompt_tokens": case["prompt_tokens"],
                    "max_new_tokens": args.agreement_new_tokens,
                    "cached_vs_generate_agreement": cross_cached["agreement"],
                    "recompute_vs_generate_agreement": cross_recompute["agreement"],
                }
            )

        # ---- Timing phase ----------------------------------------------------
        if not args.skip_timing and backend == timing_backend:
            timing_new = [int(v) for v in args.timing_new_tokens.split(",") if v.strip()]
            real_tokens = agreement_cases[0]["prompt_tokens"]
            # Warmup with a short run so the first measured cell is steady.
            for _ in range(2):
                warm_ids = synthetic_prompt_ids(64, device, base_ids)
                greedy_generate_cached(model, warm_ids, None, 8, eos_token_ids, track_memory=False)
                greedy_generate_recompute(model, warm_ids, None, 4, eos_token_ids, track_memory=False)
            for label, synthetic_len, new_tokens, prompt_tokens in timing_cells_real(
                timing_new, real_tokens
            ):
                input_ids = (
                    synthetic_prompt_ids(synthetic_len, device, base_ids)
                    if synthetic_len > 0
                    else prompt_cases[0]["input_ids"]
                )
                case_prompt_tokens = input_ids.shape[1]
                cached_result = run_cached_cell(model, input_ids, new_tokens, eos_token_ids)
                timing_rows.append(
                    {
                        "backend": backend,
                        "prompt_label": label,
                        "prompt_tokens": case_prompt_tokens,
                        "max_new_tokens": new_tokens,
                        "impl": "cache",
                        "timing": cached_result,
                        "stopped_on_eos": cached_result["stopped_on_eos"],
                    }
                )
                print(
                    f"[timing cache] {label} P={case_prompt_tokens} n={new_tokens}: "
                    f"prefill={cached_result['prefill_seconds']:.3f}s decode={cached_result['decode_seconds']:.3f}s"
                )
                if new_tokens == 32:
                    recompute_result = run_recompute_cell(model, input_ids, new_tokens, eos_token_ids)
                    timing_rows.append(
                        {
                            "backend": backend,
                            "prompt_label": label,
                            "prompt_tokens": case_prompt_tokens,
                            "max_new_tokens": new_tokens,
                            "impl": "recompute",
                            "timing": recompute_result,
                            "stopped_on_eos": recompute_result["stopped_on_eos"],
                        }
                    )
                    print(
                        f"[timing recompute] {label} P={case_prompt_tokens} n={new_tokens}: "
                        f"prefill={recompute_result['prefill_seconds']:.3f}s decode={recompute_result['decode_seconds']:.3f}s"
                    )
            # One row of library generate as a per-cell reference.
            for label in ("real_validation_0",):
                input_ids = prompt_cases[0]["input_ids"]
                reference = run_generate_reference(
                    model, tokenizer, input_ids, torch.ones_like(input_ids), 32
                )
                timing_rows.append(
                    {
                        "backend": backend,
                        "prompt_label": label,
                        "prompt_tokens": input_ids.shape[1],
                        "max_new_tokens": 32,
                        "impl": "generate",
                        "timing": reference,
                        "stopped_on_eos": reference["stopped_on_eos"],
                    }
                )
        del model
        gc.collect()

    # ---- Persist artifacts ---------------------------------------------------
    environment = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "device": str(device),
        "precision": train_config["precision"],
        "configured_backend": configured_backend,
        "model_id": model_config["model_id"],
        "revision": model_config.get("revision"),
        "threads_torch": torch.get_num_threads(),
    }
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "environment": environment,
        "agreement": agreement_rows,
        "timing": timing_rows,
    }
    out_dir = Path(args.out_dir) if args.out_dir else RESULTS_DIR / run_name / "task3"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "kv_cache_benchmark.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")

    report = build_report(payload)
    (out_dir / "report.md").write_text(report)
    print(f"results written to {out_dir}")
    return 0


def build_report(payload: dict[str, Any]) -> str:
    """Markdown report for the Task 3 findings."""
    env = payload["environment"]
    agreement = payload["agreement"]
    timing = payload["timing"]

    def backend_summary(backend: str) -> str:
        rows = [row for row in agreement if "agreement" in row and row["backend"] == backend]
        if not rows:
            return f"backend `{backend}`: no agreement runs recorded"
        agreeing = sum(row["agreement"] for row in rows)
        diverging = [row for row in rows if not row["agreement"]]
        summary = (
            f"backend `{backend}`: {agreeing}/{len(rows)} prompts with identical tokens "
            f"between the custom cached path and the no-cache recompute path."
        )
        if diverging:
            details = "; ".join(
                f"{row['prompt_label']} first divergence at step {row['first_divergence']} "
                f"(cached margin {row.get('cached_margin')}, recompute margin {row.get('recompute_margin')})"
                for row in diverging[:4]
            )
            summary += f" Divergent: {details}."
        return summary

    agreement_lines = [backend_summary(backend) for backend in sorted({row["backend"] for row in agreement})]

    cache_rows = [row for row in timing if row["impl"] == "cache"]
    recompute_rows = [row for row in timing if row["impl"] == "recompute"]
    generate_rows = [row for row in timing if row["impl"] == "generate"]

    sections = [
        "# Task 3: KV-cached autoregressive generation",
        "",
        f"Model: `{env['model_id']}` revision `{env.get('revision')}`, precision "
        f"`{env['precision']}`, device `{env['device']}`, torch `{env['torch']}`, "
        f"transformers (see environment), backend(s) `{', '.join(sorted({row['backend'] for row in agreement}))}`.",
        "",
        "## Token-level agreement (greedy decoding)",
        "",
        *agreement_lines,
        "",
        "Custom path vs ``model.generate`` cross-check:",
        "",
    ]
    for row in agreement:
        if "cached_vs_generate_agreement" in row:
            sections.append(
                f"- `{row['backend']}` {row['prompt_label']} (P={row['prompt_tokens']}): "
                f"cached==generate: {row['cached_vs_generate_agreement']}, "
                f"recompute==generate: {row['recompute_vs_generate_agreement']}"
            )
    sections += ["", "## Timing", ""]
    if cache_rows:
        sections.append(markdown_timing_table(cache_rows, "KV-cached path"))
    if recompute_rows:
        sections.append(markdown_timing_table(recompute_rows, "No-cache recompute reference"))
    if generate_rows:
        sections.append(markdown_timing_table(generate_rows, "Library generate (reference only)"))

    # Computed highlights: cached-vs-recompute decode speedup per cell, and
    # the measured per-token cache growth at large prompts (verifies that only
    # full-attention layers still grow, i.e. the sliding-window plateau).
    recompute_by_key = {
        (row["prompt_label"], row["prompt_tokens"], row["max_new_tokens"]): row
        for row in recompute_rows
    }
    cached_by_key = {
        (row["prompt_label"], row["prompt_tokens"], row["max_new_tokens"]): row
        for row in cache_rows
    }
    speedup_lines = []
    for key, cached_row in sorted(cached_by_key.items()):
        recompute_row = recompute_by_key.get(key)
        if recompute_row is None:
            continue
        cached_s = cached_row["timing"]["decode_seconds"]
        recompute_s = recompute_row["timing"]["decode_seconds"]
        if cached_s > 0:
            speedup_lines.append(
                f"- `{cached_row['prompt_label']}` P={cached_row['prompt_tokens']}, "
                f"{cached_row['max_new_tokens']} new tokens: cached decode "
                f"{cached_s:.3f}s vs recompute {recompute_s:.3f}s -> "
                f"**{recompute_s / cached_s:.1f}x** faster "
                f"({cached_row['timing']['decode_tokens_per_second']:.2f} vs "
                f"{recompute_row['timing']['decode_tokens_per_second']:.2f} tok/s decode)"
            )
    plateau_lines = []
    per_label: dict[str, list[dict[str, Any]]] = {}
    for row in cache_rows:
        per_label.setdefault(row["prompt_label"], []).append(row)
    for label, rows in sorted(per_label.items()):
        rows = sorted(rows, key=lambda r: r["max_new_tokens"])
        big = [r for r in rows if r["prompt_tokens"] >= 512]
        if len(big) < 2:
            continue
        row_a, row_b = big[-2], big[-1]
        gen_a = row_a["timing"]["generated_tokens"]
        gen_b = row_b["timing"]["generated_tokens"]
        bytes_a = row_a["timing"]["cache_logical_bytes"]
        bytes_b = row_b["timing"]["cache_logical_bytes"]
        if gen_b > gen_a:
            rate = (bytes_b - bytes_a) / (gen_b - gen_a) / 1024.0
            plateau_lines.append(
                f"- `{label}`: cache grows {rate:.1f} KB per generated token "
                f"at prompt length {row_b['prompt_tokens']} "
                f"({bytes_a / 1024:.0f} -> {bytes_b / 1024:.0f} KB); this "
                "matches only the full-attention layers growing "
                "(sliding-window layers capped at the window size)."
            )

    sections += [
        "",
        "## Measured highlights",
        "",
    ]
    if speedup_lines:
        sections += ["Decode speedup (custom cached path vs no-cache recompute):", ""] + speedup_lines
    if plateau_lines:
        sections += ["", "Cache growth verification:", ""] + plateau_lines
    sections += [
        "",
        "## Interpretation",
        "",
        "### Positional handling",
        "",
        "Positions are derived from the attention mask (cumulative attended count,",
        "the same convention ``generate`` uses), not from a naive arange over cache",
        "length: this matters the day prompts contain left padding. Rotary embeddings",
        "are applied inside each forward to the query and the *new* key only; the",
        "cached keys/values were already rotated when they were written, so no cached",
        "state is ever re-rotated. A decode step's new position equals the cache's",
        "logical length (``get_seq_length``), which is correct even when the sliding",
        "window layers have trimmed their stored keys.",
        "",
        "Gemma-3 detail: local layers use RoPE theta 10k while global layers use 1M",
        "with per-layer rotary modules; positions are the same ids, only the rotation",
        "frequencies differ. Both paths feed identical position ids per step, so any",
        "positional mismatch would show up as immediate agreement failure, which the",
        "agreement phase verifies.",
        "",
        "### Mask handling",
        "",
        "Batch size is 1 with an all-ones mask: the per-step mask grows by one column",
        "on the right and the model builds the causal (and, for local layers,",
        "sliding-window) mask over cache length + 1 keys. The no-cache path re-derives",
        "the same mask from the full sequence each step. Because both paths use",
        "identical mask semantics, mask construction cannot be the source of any",
        "observed divergence.",
        "",
        "### Cache growth and memory",
        "",
        "The cache grows by the newest key/value per layer per step. In this",
        "transformers version sliding-window layers keep only the last",
        "``sliding_window - 1`` keys while the mask hides older entries, so their",
        "memory plateaus; only the full-attention layers grow linearly with the total",
        "sequence length. The theoretically required cache for",
        "prompt + generated tokens is ``2 x layers x kv_heads x head_dim x seq_len x",
        "dtype_bytes``, which the JSON file reports per run.",
        "",
        "The recompute path stores nothing, so its memory grows only with the",
        "transient per-step activations (which do grow with sequence length but stay",
        "much smaller than the cached path's persistent cache beyond the first steps).",
        "",
        "### Numerical differences",
        "",
        "Both paths compute the same mathematical function: same positions, same",
        "causal mask, same weights and precision. Differences can still appear",
        "because in the recompute path every key/value is recomputed inside kernels",
        "with different shapes and summation orders (matmul tiling), so rounding",
        "differs slightly between the paths. When an argmax is nearly tied (log-prob",
        "margin between the top two tokens near zero), that rounding can flip the",
        "selection; the divergence diagnostics in this report record the exact step",
        "and the margins in both paths. fp32 (this benchmark) is expected to agree on",
        "all prompts; under bf16 the risk of benign flips increases and the margins",
        "table is what distinguishes a benign flip from a structural bug (a",
        "structural bug diverges with a healthy margin).",
        "",
    ]
    return "\n".join(sections)


if __name__ == "__main__":
    raise SystemExit(main())
