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
