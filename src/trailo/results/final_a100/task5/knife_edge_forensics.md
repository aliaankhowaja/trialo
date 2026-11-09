# Knife-edge forensics (validation prompt 0, output slot 14)

The speculative loop diverges from the target-only greedy baseline
first at output slot 14 of prompt 0. The probes below dump
the exact logits that decided that slot through the three
computation shapes involved, in bf16 and with the target cast to
fp32 (draft unloaded for the fp32 pass). P1 is the query shape of
the target-only baseline (and `model.generate`), P2 is the batched
verification row used by both the custom loop and HF assisted
generation, P3 is the full recompute row (Task 3 reference).

| shape | precision | argmax | runner-up | top-2 margin | exact tie |
|---|---|---:|---:|---:|---|
| P1 cached 1-token (baseline/generate) | bf16 | 84123 | 236764 | 0 | True |
| P2 cached verify row (speculative) | bf16 | 236764 | 84123 | 0.125 | False |
| P3 full recompute | bf16 | 236764 | 84123 | 0.125 | False |
| P1 cached 1-token (baseline/generate) | fp32 | 236764 | 84123 | 0.0199 | False |
| P2 cached verify row (speculative) | fp32 | 236764 | 84123 | 0.0199 | False |
| P3 full recompute | fp32 | 236764 | 84123 | 0.0199 | False |

Streams on this prompt:

- baseline: `[14937, 2491, 12375, 2217, 3487, 1938, 4337, 128010, 236779, 2643, 89045, 236771, 236764, 236770, 84123, 146879, 1083, 8612, 236783, 106]`
- speculative (gamma=2): `[14937, 2491, 12375, 2217, 3487, 1938, 4337, 128010, 236779, 2643, 89045, 236771, 236764, 236770, 236764, 236800, 84123, 146879, 1083, 8612, 236783, 106]`
- HF assisted: `[14937, 2491, 12375, 2217, 3487, 1938, 4337, 128010, 236779, 2643, 89045, 236771, 236764, 236770, 236764, 236800, 84123, 146879, 1083, 8612, 236783, 106]` (== custom loop: True, == baseline: False)

Reading of the bf16 rows: the baseline's own decision row carries an
**exact top-2 bitwise tie** (both 28.5), while the
verification row and the recompute row separate the same two tokens
by one bf16 ULP. The fp32 rows settle it: all three shapes agree
(candidate-independently, hence not an artifact of the batched-row
content) on 236764 winning by
0.0199 - about 0.07 percent of the
logit magnitude, below one bf16 ULP. bf16 therefore cannot resolve
this pair: the baseline row rounds to a dead heat and its tie-break
picks the runner-up, while the batched verification row lands one
ULP above the fp32-accurate token. The flip is a rounding dead heat,
exactly the documented benign class; the library's own assisted
generation reproduces the speculative stream token-for-token, so the
divergence belongs to the speculative-verification paradigm on this
checkpoint, not to this implementation.
