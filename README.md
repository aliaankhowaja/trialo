# Trial eligibility fine-tuning

LoRA fine-tuning for the TrialGPT criterion-annotation task. Each example maps a
patient note and one inclusion/exclusion criterion to a strict JSON object:

```json
{"label": "...", "evidence_ids": [0, 2], "explanation": ""}
```

This stage covers supervised fine-tuning and evaluation. Retrieval is out of
scope. The dataset has no expert explanation field, so `explanation` is always
the empty string in targets.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The Gemma checkpoints are gated. Accept the Google Gemma license and set
`HF_TOKEN` if the model is not already cached. The local CPU smoke run used a
pre-existing cache and `HF_HUB_OFFLINE=1`.

The CPU smoke configuration uses `google/gemma-3-270m-it` because the 15.74 GB
E4B checkpoint does not fit in 7.1 GiB of RAM. Smoke results only prove that the
pipeline runs; they are not scientific results.

## Data and splits

The source is `ncbi/TrialGPT-Criterion-Annotations` at revision
`1cfcafde94a1560a33b4addc1638664fe28fc059`. The provided manifest is
`data/splits/split_manifest.json` and is used unchanged:

| Split | Patients | Annotations |
| --- | ---: | ---: |
| train | 37 | 603 |
| validation | 8 | 206 |
| test | 8 | 206 |

The splits are patient-disjoint and annotation-disjoint. The test split is
locked: it is never used for training, early stopping, prompt or threshold
selection, or hyperparameter selection. It is evaluated exactly once by
`evaluate_test.py` after model selection is complete.

The following columns are never read: `gpt4_explanation`,
`explanation_correctness`, `gpt4_sentences`, `gpt4_eligibility`, `training`.

Prepare data in a single invocation so `stats.json` covers every split:

```bash
export PYTHONPATH=src
python -m trailo.data.check_splits
python -m trailo.data.prepare \
  --config src/trailo/configs/final_a100.yaml \
  --splits train,validation,test --include-test
python -m trailo.data.check_tokenization \
  --config src/trailo/configs/final_a100.yaml --split train
```

All prepared examples fit `max_length=1024`, and no gold-evidence sentence is
truncated. `check_tokenization --split validation` and `--split test` can be
run for the same check.

## CPU smoke test

```bash
export PYTHONPATH=src HF_HUB_OFFLINE=1
python -m trailo.data.check_splits
python -m trailo.data.prepare \
  --config src/trailo/configs/smoke_cpu.yaml \
  --splits train,validation,test --include-test
for split in train validation test; do
  python -m trailo.data.check_tokenization \
    --config src/trailo/configs/smoke_cpu.yaml --split "$split"
done
python -m trailo.train.finetune \
  --config src/trailo/configs/smoke_cpu.yaml \
  --max-steps 3 --subset-size 8 --verify-reload
python -m trailo.scripts.run_baseline_val \
  --config src/trailo/configs/smoke_cpu.yaml --split validation --subset-size 8
python -m trailo.scripts.run_eval_val \
  --config src/trailo/configs/smoke_cpu.yaml --subset-size 8
python -m trailo.scripts.check_inference \
  --config src/trailo/configs/smoke_cpu.yaml
python -m trailo.scripts.generate_report \
  --config src/trailo/configs/smoke_cpu.yaml
```

The smoke config uses fp32. This CPU reports bf16 matmul support, but bf16
autograd was measured to be unusably slow; fp32 takes about 5 seconds per
270m forward/backward micro-batch on this machine. The A100 configuration uses bf16.

## Loss tracking with Weights & Biases

Training logs train loss, learning rate, gradient norm, throughput, and
validation loss to Weights & Biases when `wandb.enabled` is true in the config.

- The smoke config uses `mode: offline`; runs are stored under
  `src/trailo/logs/wandb/` and can be uploaded later with
  `wandb sync src/trailo/logs/wandb/offline-run-*`.
- The final A100 config uses `mode: online`; run `wandb login` or set
  `WANDB_API_KEY` before training. To avoid any external service, switch the
  config to `mode: offline` or run with `--no-wandb`.
- CLI overrides: `--wandb/--no-wandb`, `--wandb-project`, `--wandb-entity`,
  `--wandb-mode`.

Tests:

```bash
```bash
python -m unittest tests.test_splits tests.test_parsing tests.test_analysis -v
```
```

## A100 / Colab run

The A100 configuration is `src/trailo/configs/final_a100.yaml`: E4B, bf16, batch
2 with gradient accumulation 4 (effective batch 8), 3 epochs, learning rate
2e-4, rank 16, `max_length=1024`, seed 42. Expect 76 optimizer steps per epoch
(228 total) and use a 40 GB A100. If memory is tight, enable
`gradient_checkpointing` in the config; do not change the data or test splits.

```bash
export PYTHONPATH=src
python -m trailo.data.check_splits
python -m trailo.data.prepare \
  --config src/trailo/configs/final_a100.yaml \
  --splits train,validation,test --include-test
python -m trailo.data.check_tokenization \
  --config src/trailo/configs/final_a100.yaml --split train
python -m trailo.train.finetune \
  --config src/trailo/configs/final_a100.yaml --verify-reload
python -m trailo.scripts.run_baseline_val \
  --config src/trailo/configs/final_a100.yaml --split validation
python -m trailo.scripts.run_eval_val \
  --config src/trailo/configs/final_a100.yaml
```

Model selection uses validation only. When it is final:

```bash
python -m trailo.scripts.evaluate_test \
  --config src/trailo/configs/final_a100.yaml --confirm-test-once
python -m trailo.scripts.generate_report \
  --config src/trailo/configs/final_a100.yaml
```

`evaluate_test.py` writes `results/final_a100/TEST_EVALUATED.json` and refuses a
second run. Use `--force` only to repair a corrupted test artifact; never use it
to try another model or setting.

## Task 2: criterion-level eligibility and evidence analysis

Task 2 is pure analysis of the saved prediction files; no model is loaded and
no inference is run, so it never touches the locked test split.

```bash
PYTHONPATH=src
python -m trailo.scripts.generate_task2_report \
  --config src/trailo/configs/final_a100.yaml
python -m unittest tests.test_analysis -v
```
Outputs under `src/trailo/results/final_a100/task2/`:

- `task2_analysis.json` — full numbers per model (`baseline`, `finetuned`) and
  split (`validation`, `test`).
- `report.md` — the human-readable report.
- `evidence_<model>_<split>.jsonl` — every prediction in the Task 2 schema:
  criterion, criterion type, predicted category, cited sentence IDs, resolved
  evidence text, and the model explanation.

Design rules implemented by the analysis:

- The four-way label set is preserved per criterion type
  (`included`/`not applicable`/`not enough information`/`not included` for
  inclusion, `excluded`/... for exclusion); nothing is collapsed to binary.
- Confusion matrices (counts and row-normalized) are reported overall and
  separately for inclusion and exclusion criteria, with per-class
  precision/recall/F1.
- Two macro-F1 conventions are reported: **declared** (full label universe,
  so a never-predicted class penalizes the score) and **observed** (gold or
  predicted labels only, matching `eval/metrics.py` headline numbers).
- Evidence retrieval is scored in three views: pooled, gold-present records
  only (the primary number), and per-record macro F1. 116/206 validation and
  145/206 test records have no gold evidence; those are reported separately as
  an empty-prediction consistency rate and never earn retrieval credit.
- Empty predicted evidence is always measured, never treated as proof of
  eligibility or ineligibility.
- Contradiction cases and error galleries (top confusion pairs with extracted
  criterion and cited-sentence text, missing-information failures, and
  not-applicable mishandling) are selected by deterministic severity-ranked
  heuristics, so the report is reproducible.

## Task 3: KV-cached autoregressive generation

Task 3 implements and validates explicit KV-cache decoding instead of relying
on a library switch. `src/trailo/eval/decoding.py` contains the custom paths:

- `greedy_generate_cached`: one full-prompt prefill forward with
  `use_cache=True`, then a loop that forwards only the newest token each step
  and reuses/updates the `DynamicCache`;
- `greedy_generate_recompute`: the reference path with no cache reuse — every
  step re-forwards the entire sequence so far with `use_cache=False`.

Both manage `position_ids` and the attention mask explicitly and reproduce
`generate` greedy conventions (fp32 argmax, eos appended then generation
stops without feeding the eos token back). The eval pipeline selects a path
with `eval.decoding_impl: generate|cache|recompute` (default `generate`, so
existing eval artifacts are unchanged).

Validation and benchmarks:

```bash
PYTHONPATH=src python -m unittest tests.test_decoding -v
PYTHONPATH=src HF_HUB_OFFLINE=1 python -m trailo.scripts.benchmark_kv_cache \
  --config src/trailo/configs/smoke_cpu.yaml \
  --prompts-dir data/processed/final_a100
python -m unittest tests.test_splits tests.test_parsing tests.test_analysis tests.test_decoding -v
```

Outputs under `src/trailo/results/<run_name>/task3/`:
`kv_cache_benchmark.json` (all raw timings, agreement diagnostics, margins)
and `report.md` (tables and the positional-handling / masking / cache-growth /
numerical-differences discussion). The local benchmark runs on the 270m smoke
checkpoint; the same script runs unchanged on the A100 config for E4B.

Details that the implementation deliberately gets right (and that the report
documents empirically):

- Positions come from the attention mask (cumulative attended count), matching
  the `generate` convention rather than a naive arange over cache length.
- Cached keys/values are stored after RoPE is applied, so nothing cached is
  ever re-rotated; a decode step's position equals the cache logical length
  even when sliding-window cache layers have trimmed stored keys.
- In this transformers version, sliding-window cache layers cap their stored
  K/V at `sliding_window - 1` tokens (memory plateaus) while full-attention
  layers grow linearly; both facts are verified against the measured cache
  sizes in the report.

## Task 5: greedy speculative decoding (E4B target, E2B draft)

Task 5 implements greedy speculative decoding from scratch. The target is the
finetuned E4B (the LoRA adapter merged into the base, with a merged-vs-attached
greedy check before merging so the merged model is proven to reproduce the
adapter eval model); the draft is `google/gemma-3n-E2B-it`, which shares the
262k Gemma-3n tokenizer (compatibility is verified per-prompt in the script,
not assumed). `src/trailo/eval/speculative.py` contains the custom loop:

- both models prefill the prompt once and keep one persistent KV cache;
- per round the draft proposes up to `gamma` tokens with its own cache, one
  target forward verifies the whole block (`logits_to_keep=1+gamma`), tokens
  are accepted left-to-right while the target greedy argmax matches, the
  first mismatch is replaced by the target argmax, and nothing is emitted
  past an accepted eos;
- after verification both caches roll back per-layer with `crop(-removal)` to
  `prompt + len(outputs) - 1`; sliding-window cache layers need
  `activate_past_recording()` once per cache for rollback to reconstruct the
  attention window (documented gotcha of the transformers version in use);
- the loop is therefore the exact target greedy sequence by construction;
  bf16 near-tie flips, if any, get margin forensics as in Task 3.

**Measured correctness result (A100, bf16).** `tests/test_speculative.py`
(8 CPU tests on a tiny same-vocab Gemma3 pair, including forced rollback,
forced eos, and replacement rounds) proves stream-exactness. On the real
E4B/E2B pair, the final agreement run emitted the **exact** target-only
greedy stream in 16/24 (prompt, gamma) runs - 4 prompts and the
1024-token synthetic prompt, which crosses the sliding-window boundary,
are token-exact at every gamma in {2, 3, 4, 6}. The remaining 8 runs (the
two prompts with a tie, all gammas) all diverge at one bf16
**knife-edge rounding tie**, not a bug: the baseline's own next-token row
at the flip slot carries a top-2 margin of exactly **0.0** (bitwise tie,
raw logits 28.5 vs 28.5 for tokens 236764/84123), while the batched
verification row differs by one bf16 ULP. fp32 probes (all three query
shapes, candidate-independent) put the true gap at 0.0199 = 0.07% of the
logit magnitude - below one bf16 ULP, so bf16 provably cannot resolve the
pair; the verification row lands on the fp32-accurate token. Critically,
**HF's own assisted generation (the same batched-verify paradigm) emits the
same stream as the custom loop** (`spec == HF-assisted` holds on the cases
where the library's tie resolution matches; at the exact-tie slot of prompt
0 the library's own resolution also varies run-to-run). See
`src/trailo/scripts/forensic_knife_edge.py` and
`results/final_a100/task5/knife_edge_forensics.{json,md}`.

**Measured performance result.** On this checkpoint pair speculative
decoding is (correctly) a measured slowdown: the E2B draft step costs ~88%
of the E4B target step because Gemma-3n as implemented here recomputes the
KV states of its KV-shared layers over the whole sequence on every forward,
charging every draft step with O(sequence length) work. A round of γ drafts
plus one verify-equivalent forward emits γα+1 tokens against
γ·0.885+1 target-step units, so the loop only wins when the acceptance rate
α exceeds the draft share (0.885); the measured α tops out at 0.69 (γ=2)
and falls to 0.41 (γ=6) as the finetuned target drifts from the base draft,
so the sweep records 0.33-0.59x across all cells. The report carries the
full gamma sweep, acceptance histograms, the phase split, and the cost
model.

Benchmark + report:

```bash
HF_HUB_CACHE=/dev/shm/hf_hub hf download google/gemma-3n-E2B-it   # once
PYTHONPATH=src python -m trailo.scripts.benchmark_speculative \
  --config src/trailo/configs/final_a100.yaml \
  --draft-cache-dir /dev/shm/hf_hub
PYTHONPATH=src python -m trailo.scripts.forensic_knife_edge \
  --config src/trailo/configs/final_a100.yaml \
  --draft-cache-dir /dev/shm/hf_hub
PYTHONPATH=src python -m unittest tests.test_speculative tests.test_decoding -v
```

(A100 required: both bf16 checkpoints plus caches. `--draft-cache-dir /dev/shm/hf_hub`
keeps the 10.9 GB draft checkpoint in tmpfs when the system disk is full;
re-download it after a reboot.)

Outputs under `src/trailo/results/<run_name>/task5/`:
`speculative_benchmark.json` (compat check, per-round acceptance records,
timings, memory) and `report.md` (equality evidence, acceptance statistics,
gamma sweep latency table, step-time ratio cost model, memory accounting,
and the measured explanation of when speculative decoding slows down).

## Outputs

- Prepared data: `data/processed/<run>/`.
- Adapters, tokenizer, resolved config, environment, metadata, checksums:
  `src/trailo/checkpoints/<run_id>/{best,epoch-N,final}`.
- Predictions and metrics: `src/trailo/results/<run_name>/`.
- Training logs and curves: `src/trailo/logs/`.
- Generated report: `src/trailo/results/<run_name>/report.md`.
- Issue log: `src/trailo/logs/issues.md`.

The saved adapter reload is verified against the in-memory model on one fixed
example; the check fails the run if the logits differ.

## Design choices and known issues

- LoRA is attached only to the text-model projections
  (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`),
  not to multimodal or audio towers. Base weights stay frozen.
- Training uses the Gemma chat template; loss is computed only on the assistant
  target, with prompt tokens masked to `-100`.
- Decoding is strict greedy decoding with `do_sample=false`, `num_beams=1`, and
  both `<eos>` and `<end_of_turn>` as stop tokens.
- The parser accepts fenced or leading prose around the JSON object but rejects
  invalid labels, duplicate keys, extra keys, non-integer or out-of-range
  evidence IDs, and missing fields.
- The local machine cannot run the E4B checkpoint. The first real E4B run must
  happen on an A100; the code path and artifact handling are shared with the
  270m smoke run.
- See `src/trailo/logs/issues.md` for the full problem/fix history.

## Report

`python -m trailo.scripts.generate_report --config <config>` builds a Markdown
report from the checkpoints, training curves, and validation/test metrics. Smoke
reports are explicitly marked as smoke runs and must not be presented as final
experimental results.
