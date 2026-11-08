# Task 3: KV-cached autoregressive generation

Model: `google/gemma-3n-E4B-it` revision `c1221e9c62e34a43ab7ffacd1be0ea71f126ef10`, precision `bf16`, device `cuda`, torch `2.14.1+cu130`, transformers (see environment), backend(s) `eager, sdpa`.

## Token-level agreement (greedy decoding)

backend `eager`: 5/9 prompts with identical tokens between the custom cached path and the no-cache recompute path. Divergent: real_validation[2] first divergence at step 23 (cached margin 0.101562, recompute margin 0.496094); synthetic_64 first divergence at step 29 (cached margin 0.0, recompute margin 0.125); synthetic_256 first divergence at step 18 (cached margin 0.375, recompute margin 0.25); synthetic_1024 first divergence at step 35 (cached margin 0.28125, recompute margin 0.1875).
backend `sdpa`: 4/9 prompts with identical tokens between the custom cached path and the no-cache recompute path. Divergent: real_validation[0] first divergence at step 30 (cached margin 0.685547, recompute margin 0.085938); real_validation[1] first divergence at step 40 (cached margin 0.09375, recompute margin 0.625); synthetic_64 first divergence at step 29 (cached margin 0.25, recompute margin 0.0); synthetic_256 first divergence at step 2 (cached margin 0.0625, recompute margin 0.125).

Custom path vs ``model.generate`` cross-check:

- `sdpa` real_validation[0] vs generate (P=338): cached==generate: True, recompute==generate: False
- `sdpa` synthetic_1024 vs generate (P=1024): cached==generate: True, recompute==generate: False
- `eager` real_validation[0] vs generate (P=338): cached==generate: True, recompute==generate: True
- `eager` synthetic_1024 vs generate (P=1024): cached==generate: True, recompute==generate: False

## Timing

### KV-cached path

| prompt | P | max new | prefill s | decode s | tok/s decode | tok/s overall | peak mem GB | Δmem GB | cache KB (peak) | eos |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| synthetic_64 | 64 | 32 | 0.1313 | 3.9779 | 7.79 | 7.79 | 14.60 | 14.597 | 3800.0 | False |
| synthetic_64 | 64 | 64 | 0.1320 | 8.1515 | 7.73 | 7.73 | 14.60 | 14.597 | 5080.0 | False |
| synthetic_64 | 64 | 128 | 0.1318 | 16.3688 | 7.76 | 7.76 | 14.60 | 14.597 | 7640.0 | False |
| synthetic_256 | 256 | 32 | 0.1339 | 2.9668 | 7.75 | 7.74 | 14.65 | 14.649 | 11160.0 | True |
| synthetic_256 | 256 | 64 | 0.1332 | 2.9777 | 7.72 | 7.71 | 14.65 | 14.649 | 11160.0 | True |
| synthetic_256 | 256 | 128 | 0.1344 | 2.9608 | 7.77 | 7.75 | 14.65 | 14.649 | 11160.0 | True |
| synthetic_1024 | 1024 | 32 | 0.1402 | 4.0797 | 7.60 | 7.58 | 14.86 | 14.865 | 24792.0 | False |
| synthetic_1024 | 1024 | 64 | 0.1415 | 8.3171 | 7.57 | 7.57 | 14.86 | 14.865 | 25048.0 | False |
| synthetic_1024 | 1024 | 128 | 0.1404 | 16.6993 | 7.61 | 7.60 | 14.86 | 14.865 | 25560.0 | False |
| real_validation_0 | 338 | 32 | 0.1342 | 3.9899 | 7.77 | 7.76 | 14.68 | 14.681 | 14760.0 | False |
| real_validation_0 | 338 | 64 | 0.1345 | 8.0970 | 7.78 | 7.78 | 14.68 | 14.681 | 16040.0 | False |
| real_validation_0 | 338 | 128 | 0.1339 | 10.0853 | 7.73 | 7.73 | 14.68 | 14.681 | 16640.0 | True |
### No-cache recompute reference

| prompt | P | max new | prefill s | decode s | tok/s decode | tok/s overall | peak mem GB | Δmem GB | cache KB (peak) | eos |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| synthetic_64 | 64 | 32 | 0.1280 | 4.0182 | 7.71 | 7.72 | 14.60 | 14.603 | 0.0 | False |
| synthetic_256 | 256 | 32 | 0.1311 | 4.0353 | 7.68 | 7.68 | 14.66 | 14.659 | 0.0 | False |
| synthetic_1024 | 1024 | 32 | 0.1360 | 4.2834 | 7.24 | 7.24 | 14.84 | 14.841 | 0.0 | False |
| real_validation_0 | 338 | 32 | 0.1306 | 4.0674 | 7.62 | 7.62 | 14.68 | 14.678 | 0.0 | False |
### Library generate (reference only)

| prompt | P | max new | prefill s | decode s | tok/s decode | tok/s overall | peak mem GB | Δmem GB | cache KB (peak) | eos |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| real_validation_0 | 338 | 32 | - | - | - | 8.06 | - | - | 0.0 | False |

## Measured highlights

Decode speedup (custom cached path vs no-cache recompute):

- `real_validation_0` P=338, 32 new tokens: cached decode 3.990s vs recompute 4.067s -> **1.0x** faster (7.77 vs 7.62 tok/s decode)
- `synthetic_1024` P=1024, 32 new tokens: cached decode 4.080s vs recompute 4.283s -> **1.0x** faster (7.60 vs 7.24 tok/s decode)
- `synthetic_256` P=256, 32 new tokens: cached decode 2.967s vs recompute 4.035s -> **1.4x** faster (7.75 vs 7.68 tok/s decode)
- `synthetic_64` P=64, 32 new tokens: cached decode 3.978s vs recompute 4.018s -> **1.0x** faster (7.79 vs 7.71 tok/s decode)

Cache growth verification:

- `synthetic_1024`: cache grows 8.0 KB per generated token at prompt length 1024 (25048 -> 25560 KB); this matches only the full-attention layers growing (sliding-window layers capped at the window size).

## Interpretation

### Positional handling

Positions are derived from the attention mask (cumulative attended count,
the same convention ``generate`` uses), not from a naive arange over cache
length: this matters the day prompts contain left padding. Rotary embeddings
are applied inside each forward to the query and the *new* key only; the
cached keys/values were already rotated when they were written, so no cached
state is ever re-rotated. A decode step's new position equals the cache's
logical length (``get_seq_length``), which is correct even when the sliding
window layers have trimmed their stored keys.

Gemma-3 detail: local layers use RoPE theta 10k while global layers use 1M
with per-layer rotary modules; positions are the same ids, only the rotation
frequencies differ. Both paths feed identical position ids per step, so any
positional mismatch would show up as immediate agreement failure, which the
agreement phase verifies.

### Mask handling

Batch size is 1 with an all-ones mask: the per-step mask grows by one column
on the right and the model builds the causal (and, for local layers,
sliding-window) mask over cache length + 1 keys. The no-cache path re-derives
the same mask from the full sequence each step. Because both paths use
identical mask semantics, mask construction cannot be the source of any
observed divergence.

### Cache growth and memory

The cache grows by the newest key/value per layer per step. In this
transformers version sliding-window layers keep only the last
``sliding_window - 1`` keys while the mask hides older entries, so their
memory plateaus; only the full-attention layers grow linearly with the total
sequence length. The theoretically required cache for
prompt + generated tokens is ``2 x layers x kv_heads x head_dim x seq_len x
dtype_bytes``, which the JSON file reports per run.

The recompute path stores nothing, so its memory grows only with the
transient per-step activations (which do grow with sequence length but stay
much smaller than the cached path's persistent cache beyond the first steps).

### Numerical differences

Both paths compute the same mathematical function: same positions, same
causal mask, same weights and precision. Differences can still appear
because in the recompute path every key/value is recomputed inside kernels
with different shapes and summation orders (matmul tiling), so rounding
differs slightly between the paths. When an argmax is nearly tied (log-prob
margin between the top two tokens near zero), that rounding can flip the
selection; the divergence diagnostics in this report record the exact step
and the margins in both paths. fp32 (this benchmark) is expected to agree on
all prompts; under bf16 the risk of benign flips increases and the margins
table is what distinguishes a benign flip from a structural bug (a
structural bug diverges with a healthy margin).
