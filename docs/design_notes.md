# Design notes

## Scope

This stage fine-tunes a causal language model to produce a criterion decision
from a patient note and one criterion. Inputs are already prepared; retrieval,
candidate generation, and cohort construction are separate concerns. The model
must emit one JSON object with a label, evidence sentence IDs, and an
explanation string.

## Data integrity

- The dataset revision is pinned in code and checked against the split manifest.
  The manifest's dataset revision is not interchangeable with the model
  revision.
- Splits are patient-disjoint. This matters because criteria and note style are
  correlated within a patient; an annotation-level split would leak patient
  context into validation and test.
- The test split is loaded only by the one-time final evaluation script.
  Training has an explicit guard that rejects a processed directory whose path
  contains `test`.
- GPT-4 annotation fields are excluded at the data-loader boundary and checked
  again after prompt/target construction.
- No expert explanation field exists. Targets therefore use `explanation: ""`.
  The field is retained so the output schema is stable and so a future dataset
  with rationales could be used without changing the parser.

## Prompt and tokenization

- Notes are parsed into sentence units with stable IDs. Evidence IDs refer to
  those units, not to raw character offsets or token indices.
- The prompt states the criterion type and criterion text, the numbered
  sentences, the allowed labels, and the JSON contract. The same builder is used
  for training, baseline inference, fine-tuned inference, and the test script.
- The Gemma chat template is applied once. The full sequence is
  `system + user + assistant prefix + target`. The training collator masks the
  prompt and padding to `-100`, so the loss is the token-level negative
  log-likelihood of the JSON target only.
- `max_length=1024` was chosen after measuring full sequences: the training
  maximum is 585 tokens, validation 454, and test 478. Zero examples are
  truncated and no gold-evidence sentence is lost. The check is implemented in
  `check_tokenization`.
- The tokenizer identifies `<end_of_turn>` as a stop token in addition to
  `<eos>`. Greedy decoding uses both IDs.

## Model and parameter-efficient tuning

- The final model is `google/gemma-3n-E4B-it`, which is a
  `Gemma3nForConditionalGeneration`; `AutoModelForCausalLM` maps to that class.
  LoRA target names are resolved by matching projection suffixes inside the
  text-model layers, not by hard-coding a full module path. This keeps the code
  usable for both `Gemma3ForCausalLM` (270m smoke model) and the E4B class.
- LoRA is applied to seven projections per text layer. The base model is frozen.
  LoRA parameters are kept in fp32 for optimizer stability; autocast supplies
  bf16 compute on the A100.
- Multimodal and audio towers are not fine-tuned. This task uses text only.

## Precision and hardware

- The A100 configuration uses bf16 autocast with SDPA attention. If SDPA fails
  at load time, the loader retries with eager attention and logs a warning.
- The CPU smoke configuration uses fp32. The local CPU advertises bf16 matmul,
  but a 270m bf16 backward pass did not finish within five minutes while the
  same fp32 step took about five seconds. Precision is a configuration value,
  not a hard-coded assumption.
- The memory preflight refuses an obviously oversized checkpoint on CPU unless
  overridden. It skips the Hub size query when offline.

## Optimization and model selection

- AdamW, linear warmup then linear decay, gradient clipping, and gradient
  accumulation are implemented directly in the training loop. The effective
  batch on A100 is 8 examples, with 228 planned optimizer steps.
- Validation loss is token-weighted. A checkpoint is saved when validation loss
  improves; the final adapter is also saved. Early stopping uses the number of
  non-improving evaluations, not raw steps.
- Model selection uses validation only. The learning rate, LoRA rank, epochs,
  and maximum length are fixed before the final run; there is no test-driven
  search.
- The adapter save/reload check compares final-position logits on one fixed
  validation example. It catches silent adapter-loading or dtype/device
  regressions before evaluation.

## Evaluation

- Decoding is greedy with a fixed maximum number of new tokens. The same
  decoding settings are used for baseline and fine-tuned models.
- The parser is strict about the schema but tolerant of Markdown fences and
  short leading prose, because instruction-tuned models commonly emit those.
  It rejects invalid labels, extra or missing keys, duplicate keys, non-integer
  evidence IDs, and out-of-range IDs.
- Metrics are reported overall and per criterion type: parse success rate,
  label accuracy (all examples), label accuracy on parsed examples, macro-F1,
  evidence precision/recall/F1, and evidence exact-match rate. Macro-F1 is
  computed over labels observed in gold or predictions; support is the gold
  count, not the number of correct predictions.
- Invalid outputs count as wrong for accuracy and contribute no evidence IDs.
  Parse success rate is reported separately so a model cannot hide syntax
  failures.

## Test protocol

The test split is evaluated once, after validation-based model selection, by
`evaluate_test.py`. The script writes a marker and refuses a second run unless
`--force` is used for artifact repair. The baseline test result is produced in
the same invocation so both models see identical data and decoding. Test
metrics must not be fed back into prompt, threshold, checkpoint, or
hyperparameter choices.

## Task 3: KV-cache decoding protocol

`src/trailo/eval/decoding.py` implements two greedy decode paths with shared
stopping and selection semantics: an explicit prefill + incremental KV-cache
loop (`greedy_generate_cached`) and a no-cache reference that re-forwards the
whole prefix each step (`greedy_generate_recompute`). `model.generate` is used
only as a cross-check. Token-level agreement between the custom paths, and
against `generate`, is asserted on every fixed benchmark prompt. The eval
pipeline exposes `decoding_impl: generate|cache|recompute` and keeps
`generate` as the default so recorded results are unchanged.

- Greedy selection casts last-position logits to fp32 (as `generate` does)
  before argmax; the decode loop records the log-probability margin between
  the argmax token and the runner-up, which is what makes an occasional
  numerical flip distinguishable from a structural bug.
- Positions derive from the attention mask via cumulative attended count
  (`generate`'s convention, not a bare arange over cache length), so left
  padding is handled correctly; RoPE is applied to Q and the new K inside each
  forward, and cached K/V are already rotated, so nothing cached is
  re-rotated. Gemma-3's hybrid attention (local layers theta 10k, global theta
  1M, sliding window 512 on the 270m and E4B text stacks) is handled by the
  model itself; the loop feeds identical position ids to both paths.
- Masks: the loop keeps one 2D mask and appends a column per step; the model
  builds the causal/sliding-window masks over cache length + 1 keys. Since
  both paths use the same mask semantics, mask construction cannot cause
  divergence.
- Cache growth: per layer the cache appends one K/V write per step. In
  transformers >= 5.20 sliding-window cache layers keep only the last
  `sliding_window - 1` stored keys (memory plateaus once the window is full)
  while full-attention layers grow linearly with total sequence length; the
  benchmark reports measured (logical element count and retained storage)
  against the theoretical `2 x layers x kv_heads x head_dim x seq x dtype`
  size.
- Numerical differences: the recompute path reproduces the same function but
  inside different kernel shapes (matmul tiling over longer sequences), so
  rounding differs slightly. Given fp32 this is too small to flip an argmax
  between the paths, and the benchmark measures agreement and margins rather
  than assuming it; under bf16, benign near-tie flips become possible and the
  margin-at-divergence diagnostics distinguish rounding from a structural bug
  (which would diverge with a healthy margin).

## Reproducibility

- Seeds cover Python, NumPy, torch, and CUDA. Deterministic algorithms are
  requested with `warn_only=True` so unsupported operations warn rather than
  abort.
- Each checkpoint stores the source config, resolved config, environment,
  training metadata, tokenizer files, adapter checksum, and a `SHA256SUMS` file.
- Training writes a CSV curve file, a JSONL event stream, and a hardware
  summary. The report generator consumes these artifacts rather than hard-coded
  numbers.
- Loss curves are also logged to Weights & Biases when enabled. The smoke
  configuration uses offline mode so no network or account is required; the
  final configuration uses online mode. Tracking can be disabled with
  `--no-wandb` without changing any training behavior.
- The manifest filename in the specification (`split_manifests`) differs from
  the repository file (`split_manifest.json`). The repository file is used and
  is not regenerated; this is recorded in the issue log.
