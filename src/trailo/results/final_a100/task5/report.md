# Task 5: greedy speculative decoding (finetuned E4B target, E2B draft)

device: `NVIDIA A100-SXM4-40GB` (39.49 GB), torch `2.14.1+cu130`, transformers `5.20.2`, precision `bf16`, backend `sdpa`.
target: `google/gemma-3n-E4B-it` (adapter `/home/ali/trialo/src/trailo/checkpoints/final_a100_seed42/best`, 430 LoRA tensors/141 MiB, torch.float32) merged into the base; draft: `google/gemma-3n-E2B-it` (revision `None`, cache /dev/shm/hf_hub).
weights resident on the GPU: target 14.563 GB + draft 10.093 GB; gammas 2, 3, 4, 6.

## Target preparation

- adapter-attached vs merged greedy agreement on a validation prompt: **True** (first divergence: None) - the merged model used everywhere below reproduces the finetuned (adapter) eval model exactly on the check prompt.
- Merging also removes the per-token LoRA overhead, so the target-only greedy baseline is a fair reference implementation.

## Draft model choice and tokenizer compatibility

- E2B shares the Gemma-3n tokenizer with E4B: vocab 262400 == 262400, tokenizer_len_equal **True**, special-token ids equal **True**; 4 prompts checked, 4 encodings identical token-for-token.
- Same-tokenizer property is what makes drafted token ids directly verifiable against target argmax ids with no conversion layer.
- E2B is the only same-family draft for this E4B target: it is generation-compatible, roughly half the activated size, and shares the chat template and eos conventions.

## Exact output equality (speculative vs target-only greedy)

Across the agreement phase, **16/24 (prompt, gamma) runs** emit exactly the target-only greedy token stream, including the eos stop point (compare_streams equality).
- `real_validation[0]` gamma=2: first divergence at slot 14 - verification-row margin 0.125, baseline margin at the same slot 0.0, knife-edge class: **True** (threshold 0.25).
- `real_validation[0]` gamma=3: first divergence at slot 14 - verification-row margin 0.125, baseline margin at the same slot 0.0, knife-edge class: **True** (threshold 0.25).
- `real_validation[0]` gamma=4: first divergence at slot 14 - verification-row margin 0.125, baseline margin at the same slot 0.0, knife-edge class: **True** (threshold 0.25).
- `real_validation[0]` gamma=6: first divergence at slot 14 - verification-row margin 0.125, baseline margin at the same slot 0.0, knife-edge class: **True** (threshold 0.25).
- `synthetic_256` gamma=2: first divergence at slot 2 - verification-row margin 0.125, baseline margin at the same slot 0.0, knife-edge class: **True** (threshold 0.25).
- `synthetic_256` gamma=3: first divergence at slot 2 - verification-row margin 0.125, baseline margin at the same slot 0.0, knife-edge class: **True** (threshold 0.25).
- `synthetic_256` gamma=4: first divergence at slot 2 - verification-row margin 0.125, baseline margin at the same slot 0.0, knife-edge class: **True** (threshold 0.25).
- `synthetic_256` gamma=6: first divergence at slot 2 - verification-row margin 0.125, baseline margin at the same slot 0.0, knife-edge class: **True** (threshold 0.25).
**All divergences classify as knife-edge rounding artifacts**: both the batched verification row that decided the flipped slot and the target-only row at the same slot carry sub-threshold margins (top-2 log-prob difference below 0.25), so different-but-equally-valid bf16 continuations were selected by the two computation shapes. A structural bug (wrong position ids, masks, or cache state) flips rows with healthy margins instead.

Library cross-references (diagnostic only):

- `real_validation[0]` (P=338): generate==baseline: True; spec==generate: g2=False g3=False g4=False g6=False; spec==HF-assisted: g2=False g3=False g4=False g6=False; hf assisted==baseline: True
- `real_validation[1]` (P=319): generate==baseline: True; spec==generate: g2=True g3=True g4=True g6=True; spec==HF-assisted: g2=True g3=True g4=True g6=True; hf assisted==baseline: True

## Draft acceptance statistics (agreement phase)

| gamma | rounds | drafted | accepted | rate | drafted/round | tokens per target fwd | full-accept | replaced |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 90 | 180 | 124 | 0.6889 | 2.00 | 1.789 | 53 | 37 |
| 3 | 73 | 215 | 125 | 0.5814 | 2.95 | 2.219 | 36 | 37 |
| 4 | 63 | 244 | 126 | 0.5164 | 3.87 | 2.587 | 26 | 37 |
| 6 | 55 | 314 | 128 | 0.4076 | 5.71 | 3.000 | 18 | 37 |

## Step-time ratio (cost-model inputs)

- per decode step (same cell, single-token loop): draft 0.113775 s, target 0.128605 s -> the E2B draft costs 88.47% of an E4B step (1.13x ratio), measured on real_validation_0, cap=64 decode loop steps.
- cost model per round: gamma draft steps + one verify forward of 1+gamma target positions; predicted per-round emitted tokens = gamma*alpha (accepted) + 1 (replacement or next-round argmax).

## Latency, throughput, and memory

| prompt | P | cap | impl | gamma | median s | tok/s | x baseline | accept | tok/verify | rounds | peak GB | eos | stable |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| natural_0 | 338 | 128 | baseline | - | 2.5896 | 7.723 | - | - | - | - | 24.759 | True | True |
| natural_0 | 338 | 128 | spec | 2 | 4.6236 | 4.758 | 0.56 | 0.5 | 1.75 | 12 | 24.759 | True | True |
| natural_0 | 338 | 128 | spec | 3 | 5.3765 | 4.092 | 0.482 | 0.375 | 1.9091 | 11 | 24.759 | True | True |
| natural_0 | 338 | 128 | spec | 4 | 5.9151 | 3.719 | 0.438 | 0.3158 | 2.1 | 10 | 24.759 | True | True |
| natural_0 | 338 | 128 | spec | 6 | 7.9098 | 2.781 | 0.327 | 0.2143 | 2.1 | 10 | 24.759 | True | True |
| natural_1 | 319 | 128 | baseline | - | 2.0778 | 7.7 | - | - | - | - | 24.753 | True | True |
| natural_1 | 319 | 128 | spec | 2 | 3.5204 | 4.545 | 0.59 | 0.4444 | 1.6667 | 9 | 24.753 | True | True |
| natural_1 | 319 | 128 | spec | 3 | 4.4395 | 3.604 | 0.468 | 0.3077 | 1.6667 | 9 | 24.753 | True | True |
| natural_1 | 319 | 128 | spec | 4 | 4.7308 | 3.382 | 0.439 | 0.2667 | 1.875 | 8 | 24.753 | True | True |
| natural_1 | 319 | 128 | spec | 6 | 6.2773 | 2.549 | 0.331 | 0.1818 | 1.875 | 8 | 24.753 | True | True |
| natural_2 | 325 | 128 | baseline | - | 2.0622 | 7.759 | - | - | - | - | 24.755 | True | True |
| natural_2 | 325 | 128 | spec | 2 | 3.5098 | 4.559 | 0.588 | 0.5 | 1.6667 | 9 | 24.755 | True | True |
| natural_2 | 325 | 128 | spec | 3 | 3.9349 | 4.066 | 0.524 | 0.3913 | 1.875 | 8 | 24.755 | True | True |
| natural_2 | 325 | 128 | spec | 4 | 4.72 | 3.39 | 0.437 | 0.3 | 1.875 | 8 | 24.755 | True | True |
| natural_2 | 325 | 128 | spec | 6 | 5.4821 | 2.919 | 0.376 | 0.2368 | 2.1429 | 7 | 24.755 | True | True |
| real_validation_0 | 338 | 64 | baseline | - | 2.5782 | 7.757 | - | - | - | - | 24.759 | True | True |
| real_validation_0 | 338 | 64 | spec | 2 | 4.5905 | 4.793 | 0.562 | 0.5 | 1.75 | 12 | 24.759 | True | True |
| real_validation_0 | 338 | 64 | spec | 3 | 5.4083 | 4.068 | 0.477 | 0.375 | 1.9091 | 11 | 24.759 | True | True |
| real_validation_0 | 338 | 64 | spec | 4 | 6.0087 | 3.661 | 0.429 | 0.3158 | 2.1 | 10 | 24.759 | True | True |
| real_validation_0 | 338 | 64 | spec | 6 | 8.1016 | 2.716 | 0.318 | 0.2143 | 2.1 | 10 | 24.759 | True | True |
| real_validation_0 | 338 | 128 | baseline | - | 2.7305 | 7.325 | - | - | - | - | 24.759 | True | True |
| real_validation_0 | 338 | 128 | spec | 2 | 4.733 | 4.648 | 0.577 | 0.5 | 1.75 | 12 | 24.759 | True | True |
| real_validation_0 | 338 | 128 | spec | 3 | 5.4738 | 4.019 | 0.499 | 0.375 | 1.9091 | 11 | 24.759 | True | True |
| real_validation_0 | 338 | 128 | spec | 4 | 6.035 | 3.645 | 0.452 | 0.3158 | 2.1 | 10 | 24.759 | True | True |
| real_validation_0 | 338 | 128 | spec | 6 | 8.3042 | 2.649 | 0.329 | 0.2143 | 2.1 | 10 | 24.759 | True | True |
| draft_only | 338 | 64 | draft_only | - | 7.2864 | 8.783 | - | - | - | - | - | False | True |

### Speculative phase split (medians over repeats)

| prompt | cap | gamma | prefill s | draft s | verify s | rounds | acceptance |
|---|---:|---:|---:|---:|---:|---:|---:|
| natural_0 | 128 | 2 | 0.2501 | 2.6982 | 1.6507 | 12 | 0.5 |
| natural_0 | 128 | 3 | 0.2483 | 3.596 | 1.5091 | 11 | 0.375 |
| natural_0 | 128 | 4 | 0.2524 | 4.2596 | 1.3806 | 10 | 0.3158 |
| natural_0 | 128 | 6 | 0.2494 | 6.2667 | 1.3716 | 10 | 0.2143 |
| natural_1 | 128 | 2 | 0.2484 | 2.021 | 1.2347 | 9 | 0.4444 |
| natural_1 | 128 | 3 | 0.2488 | 2.9341 | 1.2422 | 9 | 0.3077 |
| natural_1 | 128 | 4 | 0.2488 | 3.3645 | 1.1022 | 8 | 0.2667 |
| natural_1 | 128 | 6 | 0.2491 | 4.9129 | 1.0994 | 8 | 0.1818 |
| natural_2 | 128 | 2 | 0.2489 | 2.0136 | 1.2325 | 9 | 0.5 |
| natural_2 | 128 | 3 | 0.2476 | 2.5715 | 1.0966 | 8 | 0.3913 |
| natural_2 | 128 | 4 | 0.248 | 3.3586 | 1.0967 | 8 | 0.3 |
| natural_2 | 128 | 6 | 0.2484 | 4.2591 | 0.9643 | 7 | 0.2368 |
| real_validation_0 | 64 | 2 | 0.2471 | 2.6803 | 1.6425 | 12 | 0.5 |
| real_validation_0 | 64 | 3 | 0.249 | 3.6231 | 1.5166 | 11 | 0.375 |
| real_validation_0 | 64 | 4 | 0.2528 | 4.3398 | 1.3962 | 10 | 0.3158 |
| real_validation_0 | 64 | 6 | 0.2516 | 6.4232 | 1.4061 | 10 | 0.2143 |
| real_validation_0 | 128 | 2 | 0.253 | 2.7677 | 1.692 | 12 | 0.5 |
| real_validation_0 | 128 | 3 | 0.2549 | 3.6599 | 1.5391 | 11 | 0.375 |
| real_validation_0 | 128 | 4 | 0.2549 | 4.3543 | 1.4023 | 10 | 0.3158 |
| real_validation_0 | 128 | 6 | 0.2554 | 6.5822 | 1.4379 | 10 | 0.2143 |

Note on the prefill split: the speculative prefill includes an extra draft prefill over the same prompt (the draft needs its own KV of the prompt to draft from), so its fixed cost per generation is one E2B prompt forward on top of the target's - amortized by eos-stopped generations of practical length.

### Memory accounting

- baseline: decode-phase peak 24.759 GB (target cache 14280 KB) - `natural_0`, cap 128
- gamma=2: decode-phase peak 24.759 GB (target cache 14400 KB + draft cache 14360 KB) - `natural_0`, cap 128
- gamma=3: decode-phase peak 24.759 GB (target cache 14400 KB + draft cache 14360 KB) - `natural_0`, cap 128
- gamma=4: decode-phase peak 24.759 GB (target cache 14440 KB + draft cache 14400 KB) - `natural_0`, cap 128
- gamma=6: decode-phase peak 24.759 GB (target cache 14520 KB + draft cache 14480 KB) - `natural_0`, cap 128
- baseline: decode-phase peak 24.753 GB (target cache 13360 KB) - `natural_1`, cap 128
- gamma=2: decode-phase peak 24.753 GB (target cache 13400 KB + draft cache 13360 KB) - `natural_1`, cap 128
- gamma=3: decode-phase peak 24.753 GB (target cache 13400 KB + draft cache 13360 KB) - `natural_1`, cap 128
- gamma=4: decode-phase peak 24.753 GB (target cache 13440 KB + draft cache 13400 KB) - `natural_1`, cap 128
- gamma=6: decode-phase peak 24.753 GB (target cache 13520 KB + draft cache 13480 KB) - `natural_1`, cap 128
- baseline: decode-phase peak 24.755 GB (target cache 13600 KB) - `natural_2`, cap 128
- gamma=2: decode-phase peak 24.755 GB (target cache 13640 KB + draft cache 13600 KB) - `natural_2`, cap 128
- gamma=3: decode-phase peak 24.755 GB (target cache 13640 KB + draft cache 13600 KB) - `natural_2`, cap 128
- gamma=4: decode-phase peak 24.755 GB (target cache 13680 KB + draft cache 13640 KB) - `natural_2`, cap 128
- gamma=6: decode-phase peak 24.755 GB (target cache 13760 KB + draft cache 13720 KB) - `natural_2`, cap 128
- baseline: decode-phase peak 24.759 GB (target cache 14280 KB) - `real_validation_0`, cap 64
- gamma=2: decode-phase peak 24.759 GB (target cache 14400 KB + draft cache 14360 KB) - `real_validation_0`, cap 64
- gamma=3: decode-phase peak 24.759 GB (target cache 14400 KB + draft cache 14360 KB) - `real_validation_0`, cap 64
- gamma=4: decode-phase peak 24.759 GB (target cache 14440 KB + draft cache 14400 KB) - `real_validation_0`, cap 64
- gamma=6: decode-phase peak 24.759 GB (target cache 14520 KB + draft cache 14480 KB) - `real_validation_0`, cap 64
- baseline: decode-phase peak 24.759 GB (target cache 14280 KB) - `real_validation_0`, cap 128
- gamma=2: decode-phase peak 24.759 GB (target cache 14400 KB + draft cache 14360 KB) - `real_validation_0`, cap 128
- gamma=3: decode-phase peak 24.759 GB (target cache 14400 KB + draft cache 14360 KB) - `real_validation_0`, cap 128
- gamma=4: decode-phase peak 24.759 GB (target cache 14440 KB + draft cache 14400 KB) - `real_validation_0`, cap 128
- gamma=6: decode-phase peak 24.759 GB (target cache 14520 KB + draft cache 14480 KB) - `real_validation_0`, cap 128

The speculative loop always pays the draft checkpoint plus a second KV cache on top of the target-only baseline; the recorded sliding-window recording mode additionally retains the post-prompt appended keys between rollbacks, still bounded by prompt + generated tokens rather than unbounded.

## Interpretation

### Correctness

Greedy speculative decoding is exact for the mathematical function: every emitted token is either a draft token whose position's target argmax - computed in the same verification forward that extends the target cache - equals the draft token, or the target argmax at the first rejected position, and nothing is generated past an accepted eos. The agreement phase verifies this token-for-token against the target-only Task 3 path and model.generate, and the tiny-model test suite (tests/test_speculative.py) proves stream-exactness including eos stops and forced rollbacks. Every divergence this benchmark ever recorded on the real checkpoint pair classifies as a knife-edge rounding artifact: at the flipped slot the baseline's own single-token row carries a top-2 log-prob margin of exactly 0.0 (a bitwise bf16 tie) and the verification row differs by at most one bf16 ULP (threshold 0.25). Companion artifact `knife_edge_forensics.{json,md}` in the same directory dumps the exact logits of the first such position through all three query shapes in bf16 and fp32: the true (fp32, shape-independent, candidate-independent) logit gap is 0.0199 - 0.07 percent of the logit magnitude, below one bf16 ULP - so bf16 provably cannot resolve the pair, and the batched-verification shape lands on the fp32-accurate token while the baseline's tie-break picks the other. HF's own assisted generation (same paradigm, same verify rows) reproduces the speculative stream token-for-token on the case where its tie-resolution matches this loop's; at the exact-tie slot of prompt 0 the library's own resolution also varies run-to-run, which is further evidence that such slots are dead heats rather than well-defined argmaxes. A structural bug (bad position ids, wrong masks, stale cache) flips rows with healthy margins instead - none was observed; margins are recorded per row to keep that check auditable.

### Cache-state handling

Both models prefill once and keep a persistent cache each. Invariant at every round start: each cache holds prompt + all confirmed outputs except the pending token; the pending token is re-fed as the first token of both the draft phase and the verification forward. After verification the per-layer rollback (`crop(-removal)`) restores that invariant exactly - on this transformers version sliding-window layers trim their buffer once the window is reached, so `activate_past_recording` is called once per cache to keep the appended tail recoverable; recorded-layer `crop(-removal)` then reconstructs exactly the post-rollback attention window, and the agreement phase over hundreds of rounds is the empirical proof.

### When it speeds up, and why it can slow down

With draft share 88.47% of a target step, a round of gamma drafts costs gamma*0.88 target-step units plus one verification forward - the loop only wins while the mean emitted tokens per round (gamma*alpha+1) exceeds that mean round cost. The acceptance histogram shows where blocks break: after the first rejection inside a block, the remaining drafted tokens are pure cost, so gamma beyond the tail of the histogram wastes verify width.
Why the draft share is so high on THIS checkpoint pair: Gemma-3n (as implemented in this transformers version) recomputes the key/value states of its KV-shared layers fresh over the whole sequence on EVERY forward, so the per-step cost grows with the sequence length for both models and E2B ends up costing 88.47% of an E4B step rather than the ~50% its parameter count suggests. The break-even condition follows directly: a round of gamma drafts plus one verify-equivalent forward emits gamma*alpha + 1 tokens against gamma*0.88 + 1 target-step units, so the loop wins exactly when acceptance alpha EXCEEDS the draft share (0.88 here). The measured acceptance tops out at 0.69 (gamma=2) and falls with gamma (0.41 at gamma=6) because the finetuned target's output distribution drifts from the base E2B draft after a handful of tokens, so the speculative loop lands strictly below 1x - a slowdown, measured and explained. It would still pay off if the same target were paired with a genuinely cheaper draft (draft share well below the acceptance rate); with this E4B/E2B pair on this implementation there is no gamma that breaks even.