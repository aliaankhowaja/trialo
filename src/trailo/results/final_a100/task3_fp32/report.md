# Task 3: KV-cached autoregressive generation

Model: `google/gemma-3n-E4B-it` revision `c1221e9c62e34a43ab7ffacd1be0ea71f126ef10`, precision `fp32`, device `cuda`, torch `2.14.1+cu130`, transformers (see environment), backend(s) `eager, sdpa`.

## Token-level agreement (greedy decoding)

backend `eager`: 9/9 prompts with identical tokens between the custom cached path and the no-cache recompute path.
backend `sdpa`: 9/9 prompts with identical tokens between the custom cached path and the no-cache recompute path.

Custom path vs ``model.generate`` cross-check:

- `sdpa` real_validation[0] vs generate (P=338): cached==generate: True, recompute==generate: True
- `sdpa` synthetic_1024 vs generate (P=1024): cached==generate: True, recompute==generate: True
- `eager` real_validation[0] vs generate (P=338): cached==generate: True, recompute==generate: True
- `eager` synthetic_1024 vs generate (P=1024): cached==generate: True, recompute==generate: True

## Timing


## Measured highlights


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
