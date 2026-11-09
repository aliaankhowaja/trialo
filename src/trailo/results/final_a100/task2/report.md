# Task 2: Criterion-Level Eligibility and Evidence Verification

Generated `2026-11-08T15:15:58.424497+00:00` for run `final_a100` from saved prediction files; no model was loaded and no inference was run for this report.

## Label policy

The full four-way expert label set is preserved per criterion type: `included`/`not applicable`/`not enough information`/`not included` for inclusion and `excluded`/`not applicable`/`not enough information`/`not excluded` for exclusion. `not applicable` and `not enough information` are never collapsed into yes/no outcomes.

### Macro-F1 convention

Two macro-F1 numbers are reported everywhere. **Declared** averages over the full expert label universe, so a class the model never predicts (F1 = 0, e.g. `excluded` in the fine-tuned run) still penalizes the score. **Observed** averages only over labels with gold support or predictions, exactly matching the headline `macro_f1` in eval/metrics.py so the numbers here stay comparable with the validation/test metric files.

## Evidence policy

Evidence is scored in three views. (1) Pooled: micro-aggregated true/false-positive/negative counts over every parsed record, identical to the headline metric in eval/metrics.py. Records with no annotated gold evidence cannot add recall and any cited sentence on such a record counts as a false positive. (2) Gold-present: the same scoring restricted to records whose gold evidence set is non-empty; the primary comparable retrieval number. (3) Per-record: every gold-present record contributes one P/R/F1 triple and the macro average is reported. Records with empty gold evidence are reported separately as a consistency rate (predicted evidence also empty); they never earn credit toward evidence quality. Empty predicted evidence is always measured, never treated as proof of eligibility or ineligibility, by the analysis or the report.

## 1. Criterion classification (overall)

| model | split | n | parse | accuracy | macro-F1 (declared) | macro-F1 (observed) |
|---|---|---:|---:|---:|---:|---:|
| baseline | test | 206 | 1.000 | 0.335 | 0.310 | 0.310 |
| baseline | validation | 206 | 1.000 | 0.325 | 0.294 | 0.294 |
| finetuned | test | 206 | 1.000 | 0.748 | 0.518 | 0.621 |
| finetuned | validation | 206 | 1.000 | 0.733 | 0.420 | 0.420 |

## 2. Confusion matrices

### baseline - test - overall

`accuracy=0.335` `macro-F1 (declared)=0.310` `macro-F1 (observed)=0.310` `n=206`

| gold \ pred | excluded | included | not applicable | not enough information | not excluded | not included |
|---|---|---|---|---|---|---|
| **excluded** | 0.000 (0) | 0.000 (0) | 0.000 (0) | 0.000 (0) | 0.000 (0) | 0.000 (0) |
| **included** | 0.000 (0) | 0.926 (25) | 0.000 (0) | 0.074 (2) | 0.000 (0) | 0.000 (0) |
| **not applicable** | 0.000 (0) | 0.000 (0) | 0.800 (12) | 0.133 (2) | 0.000 (0) | 0.067 (1) |
| **not enough information** | 0.030 (2) | 0.106 (7) | 0.303 (20) | 0.439 (29) | 0.015 (1) | 0.106 (7) |
| **not excluded** | 0.077 (7) | 0.000 (0) | 0.681 (62) | 0.242 (22) | 0.000 (0) | 0.000 (0) |
| **not included** | 0.000 (0) | 0.143 (1) | 0.286 (2) | 0.143 (1) | 0.000 (0) | 0.429 (3) |

Per-class precision/recall/F1:

| label | support | predicted | precision | recall | F1 |
|---|---:|---:|---:|---:|---:|
| excluded | 0 | 9 | 0.000 | 0.000 | 0.000 |
| included | 27 | 33 | 0.758 | 0.926 | 0.833 |
| not applicable | 15 | 96 | 0.125 | 0.800 | 0.216 |
| not enough information | 66 | 56 | 0.518 | 0.439 | 0.475 |
| not excluded | 91 | 1 | 0.000 | 0.000 | 0.000 |
| not included | 7 | 11 | 0.273 | 0.429 | 0.333 |

### baseline - test - inclusion criteria

`accuracy=0.693` `macro-F1 (declared)=0.577` `macro-F1 (observed)=0.577` `n=75`

| gold \ pred | included | not applicable | not enough information | not included |
|---|---|---|---|---|
| **included** | 0.926 (25) | 0.000 (0) | 0.074 (2) | 0.000 (0) |
| **not applicable** | 0.000 (0) | 0.667 (2) | 0.000 (0) | 0.333 (1) |
| **not enough information** | 0.184 (7) | 0.053 (2) | 0.579 (22) | 0.184 (7) |
| **not included** | 0.143 (1) | 0.286 (2) | 0.143 (1) | 0.429 (3) |

Per-class precision/recall/F1:

| label | support | predicted | precision | recall | F1 |
|---|---:|---:|---:|---:|---:|
| included | 27 | 33 | 0.758 | 0.926 | 0.833 |
| not applicable | 3 | 6 | 0.333 | 0.667 | 0.444 |
| not enough information | 38 | 25 | 0.880 | 0.579 | 0.698 |
| not included | 7 | 11 | 0.273 | 0.429 | 0.333 |

### baseline - test - exclusion criteria

`accuracy=0.130` `macro-F1 (declared)=0.108` `macro-F1 (observed)=0.108` `n=131`

| gold \ pred | excluded | not applicable | not enough information | not excluded |
|---|---|---|---|---|
| **excluded** | 0.000 (0) | 0.000 (0) | 0.000 (0) | 0.000 (0) |
| **not applicable** | 0.000 (0) | 0.833 (10) | 0.167 (2) | 0.000 (0) |
| **not enough information** | 0.071 (2) | 0.643 (18) | 0.250 (7) | 0.036 (1) |
| **not excluded** | 0.077 (7) | 0.681 (62) | 0.242 (22) | 0.000 (0) |

Per-class precision/recall/F1:

| label | support | predicted | precision | recall | F1 |
|---|---:|---:|---:|---:|---:|
| excluded | 0 | 9 | 0.000 | 0.000 | 0.000 |
| not applicable | 12 | 90 | 0.111 | 0.833 | 0.196 |
| not enough information | 28 | 31 | 0.226 | 0.250 | 0.237 |
| not excluded | 91 | 1 | 0.000 | 0.000 | 0.000 |

##### baseline - test - evidence retrieval

| view | n | precision | recall | F1 | exact match |
|---|---:|---:|---:|---:|---:|
| pooled | 206 | 0.476 | 0.415 | 0.443 | 0.714 |
| gold_present | 61 | 0.696 | 0.415 | 0.520 | 0.377 |
| no-gold consistency | 145 | - | - | - | empty-pred rate 0.855 |
| per-record macro F1 (gold-present) | 61 | - | - | 0.469 | - |


### baseline - validation - overall

`accuracy=0.325` `macro-F1 (declared)=0.294` `macro-F1 (observed)=0.294` `n=206`

| gold \ pred | excluded | included | not applicable | not enough information | not excluded | not included |
|---|---|---|---|---|---|---|
| **excluded** | 1.000 (2) | 0.000 (0) | 0.000 (0) | 0.000 (0) | 0.000 (0) | 0.000 (0) |
| **included** | 0.000 (0) | 0.743 (26) | 0.057 (2) | 0.086 (3) | 0.000 (0) | 0.114 (4) |
| **not applicable** | 0.167 (2) | 0.000 (0) | 0.667 (8) | 0.083 (1) | 0.083 (1) | 0.000 (0) |
| **not enough information** | 0.075 (4) | 0.226 (12) | 0.151 (8) | 0.415 (22) | 0.019 (1) | 0.113 (6) |
| **not excluded** | 0.119 (12) | 0.000 (0) | 0.525 (53) | 0.277 (28) | 0.079 (8) | 0.000 (0) |
| **not included** | 0.000 (0) | 0.667 (2) | 0.000 (0) | 0.000 (0) | 0.000 (0) | 0.333 (1) |

Per-class precision/recall/F1:

| label | support | predicted | precision | recall | F1 |
|---|---:|---:|---:|---:|---:|
| excluded | 2 | 20 | 0.100 | 1.000 | 0.182 |
| included | 35 | 40 | 0.650 | 0.743 | 0.693 |
| not applicable | 12 | 71 | 0.113 | 0.667 | 0.193 |
| not enough information | 53 | 54 | 0.407 | 0.415 | 0.411 |
| not excluded | 101 | 10 | 0.800 | 0.079 | 0.144 |
| not included | 3 | 11 | 0.091 | 0.333 | 0.143 |

### baseline - validation - inclusion criteria

`accuracy=0.545` `macro-F1 (declared)=0.322` `macro-F1 (observed)=0.322` `n=66`

| gold \ pred | included | not applicable | not enough information | not included |
|---|---|---|---|---|
| **included** | 0.743 (26) | 0.057 (2) | 0.086 (3) | 0.114 (4) |
| **not applicable** | 0.000 (0) | 0.000 (0) | 1.000 (1) | 0.000 (0) |
| **not enough information** | 0.444 (12) | 0.000 (0) | 0.333 (9) | 0.222 (6) |
| **not included** | 0.667 (2) | 0.000 (0) | 0.000 (0) | 0.333 (1) |

Per-class precision/recall/F1:

| label | support | predicted | precision | recall | F1 |
|---|---:|---:|---:|---:|---:|
| included | 35 | 40 | 0.650 | 0.743 | 0.693 |
| not applicable | 1 | 2 | 0.000 | 0.000 | 0.000 |
| not enough information | 27 | 13 | 0.692 | 0.333 | 0.450 |
| not included | 3 | 11 | 0.091 | 0.333 | 0.143 |

### baseline - validation - exclusion criteria

`accuracy=0.221` `macro-F1 (declared)=0.229` `macro-F1 (observed)=0.229` `n=140`

| gold \ pred | excluded | not applicable | not enough information | not excluded |
|---|---|---|---|---|
| **excluded** | 1.000 (2) | 0.000 (0) | 0.000 (0) | 0.000 (0) |
| **not applicable** | 0.182 (2) | 0.727 (8) | 0.000 (0) | 0.091 (1) |
| **not enough information** | 0.154 (4) | 0.308 (8) | 0.500 (13) | 0.038 (1) |
| **not excluded** | 0.119 (12) | 0.525 (53) | 0.277 (28) | 0.079 (8) |

Per-class precision/recall/F1:

| label | support | predicted | precision | recall | F1 |
|---|---:|---:|---:|---:|---:|
| excluded | 2 | 20 | 0.100 | 1.000 | 0.182 |
| not applicable | 11 | 69 | 0.116 | 0.727 | 0.200 |
| not enough information | 26 | 41 | 0.317 | 0.500 | 0.388 |
| not excluded | 101 | 10 | 0.800 | 0.079 | 0.144 |

##### baseline - validation - evidence retrieval

| view | n | precision | recall | F1 | exact match |
|---|---:|---:|---:|---:|---:|
| pooled | 206 | 0.369 | 0.417 | 0.392 | 0.558 |
| gold_present | 90 | 0.585 | 0.417 | 0.487 | 0.367 |
| no-gold consistency | 116 | - | - | - | empty-pred rate 0.707 |
| per-record macro F1 (gold-present) | 90 | - | - | 0.465 | - |


### finetuned - test - overall

`accuracy=0.748` `macro-F1 (declared)=0.518` `macro-F1 (observed)=0.621` `n=206`

| gold \ pred | excluded | included | not applicable | not enough information | not excluded | not included |
|---|---|---|---|---|---|---|
| **excluded** | 0.000 (0) | 0.000 (0) | 0.000 (0) | 0.000 (0) | 0.000 (0) | 0.000 (0) |
| **included** | 0.000 (0) | 0.704 (19) | 0.000 (0) | 0.296 (8) | 0.000 (0) | 0.000 (0) |
| **not applicable** | 0.000 (0) | 0.000 (0) | 0.267 (4) | 0.067 (1) | 0.667 (10) | 0.000 (0) |
| **not enough information** | 0.000 (0) | 0.000 (0) | 0.000 (0) | 0.576 (38) | 0.409 (27) | 0.015 (1) |
| **not excluded** | 0.000 (0) | 0.000 (0) | 0.000 (0) | 0.000 (0) | 1.000 (91) | 0.000 (0) |
| **not included** | 0.000 (0) | 0.000 (0) | 0.143 (1) | 0.571 (4) | 0.000 (0) | 0.286 (2) |

Per-class precision/recall/F1:

| label | support | predicted | precision | recall | F1 |
|---|---:|---:|---:|---:|---:|
| excluded | 0 | 0 | 0.000 | 0.000 | 0.000 |
| included | 27 | 19 | 1.000 | 0.704 | 0.826 |
| not applicable | 15 | 5 | 0.800 | 0.267 | 0.400 |
| not enough information | 66 | 51 | 0.745 | 0.576 | 0.650 |
| not excluded | 91 | 128 | 0.711 | 1.000 | 0.831 |
| not included | 7 | 3 | 0.667 | 0.286 | 0.400 |

### finetuned - test - inclusion criteria

`accuracy=0.800` `macro-F1 (declared)=0.683` `macro-F1 (observed)=0.683` `n=75`

| gold \ pred | included | not applicable | not enough information | not included |
|---|---|---|---|---|
| **included** | 0.704 (19) | 0.000 (0) | 0.296 (8) | 0.000 (0) |
| **not applicable** | 0.000 (0) | 0.667 (2) | 0.333 (1) | 0.000 (0) |
| **not enough information** | 0.000 (0) | 0.000 (0) | 0.974 (37) | 0.026 (1) |
| **not included** | 0.000 (0) | 0.143 (1) | 0.571 (4) | 0.286 (2) |

Per-class precision/recall/F1:

| label | support | predicted | precision | recall | F1 |
|---|---:|---:|---:|---:|---:|
| included | 27 | 19 | 1.000 | 0.704 | 0.826 |
| not applicable | 3 | 3 | 0.667 | 0.667 | 0.667 |
| not enough information | 38 | 50 | 0.740 | 0.974 | 0.841 |
| not included | 7 | 3 | 0.667 | 0.286 | 0.400 |

### finetuned - test - exclusion criteria

`accuracy=0.718` `macro-F1 (declared)=0.296` `macro-F1 (observed)=0.395` `n=131`

| gold \ pred | excluded | not applicable | not enough information | not excluded |
|---|---|---|---|---|
| **excluded** | 0.000 (0) | 0.000 (0) | 0.000 (0) | 0.000 (0) |
| **not applicable** | 0.000 (0) | 0.167 (2) | 0.000 (0) | 0.833 (10) |
| **not enough information** | 0.000 (0) | 0.000 (0) | 0.036 (1) | 0.964 (27) |
| **not excluded** | 0.000 (0) | 0.000 (0) | 0.000 (0) | 1.000 (91) |

Per-class precision/recall/F1:

| label | support | predicted | precision | recall | F1 |
|---|---:|---:|---:|---:|---:|
| excluded | 0 | 0 | 0.000 | 0.000 | 0.000 |
| not applicable | 12 | 2 | 1.000 | 0.167 | 0.286 |
| not enough information | 28 | 1 | 1.000 | 0.036 | 0.069 |
| not excluded | 91 | 128 | 0.711 | 1.000 | 0.831 |

##### finetuned - test - evidence retrieval

| view | n | precision | recall | F1 | exact match |
|---|---:|---:|---:|---:|---:|
| pooled | 206 | 0.496 | 0.745 | 0.596 | 0.743 |
| gold_present | 61 | 0.729 | 0.745 | 0.737 | 0.607 |
| no-gold consistency | 145 | - | - | - | empty-pred rate 0.800 |
| per-record macro F1 (gold-present) | 61 | - | - | 0.748 | - |


### finetuned - validation - overall

`accuracy=0.733` `macro-F1 (declared)=0.420` `macro-F1 (observed)=0.420` `n=206`

| gold \ pred | excluded | included | not applicable | not enough information | not excluded | not included |
|---|---|---|---|---|---|---|
| **excluded** | 0.000 (0) | 0.000 (0) | 0.000 (0) | 0.000 (0) | 1.000 (2) | 0.000 (0) |
| **included** | 0.000 (0) | 0.600 (21) | 0.000 (0) | 0.400 (14) | 0.000 (0) | 0.000 (0) |
| **not applicable** | 0.000 (0) | 0.000 (0) | 0.250 (3) | 0.083 (1) | 0.667 (8) | 0.000 (0) |
| **not enough information** | 0.000 (0) | 0.019 (1) | 0.000 (0) | 0.491 (26) | 0.491 (26) | 0.000 (0) |
| **not excluded** | 0.000 (0) | 0.000 (0) | 0.000 (0) | 0.000 (0) | 1.000 (101) | 0.000 (0) |
| **not included** | 0.000 (0) | 0.000 (0) | 0.000 (0) | 1.000 (3) | 0.000 (0) | 0.000 (0) |

Per-class precision/recall/F1:

| label | support | predicted | precision | recall | F1 |
|---|---:|---:|---:|---:|---:|
| excluded | 2 | 0 | 0.000 | 0.000 | 0.000 |
| included | 35 | 22 | 0.955 | 0.600 | 0.737 |
| not applicable | 12 | 3 | 1.000 | 0.250 | 0.400 |
| not enough information | 53 | 44 | 0.591 | 0.491 | 0.536 |
| not excluded | 101 | 137 | 0.737 | 1.000 | 0.849 |
| not included | 3 | 0 | 0.000 | 0.000 | 0.000 |

### finetuned - validation - inclusion criteria

`accuracy=0.712` `macro-F1 (declared)=0.367` `macro-F1 (observed)=0.367` `n=66`

| gold \ pred | included | not applicable | not enough information | not included |
|---|---|---|---|---|
| **included** | 0.600 (21) | 0.000 (0) | 0.400 (14) | 0.000 (0) |
| **not applicable** | 0.000 (0) | 0.000 (0) | 1.000 (1) | 0.000 (0) |
| **not enough information** | 0.037 (1) | 0.000 (0) | 0.963 (26) | 0.000 (0) |
| **not included** | 0.000 (0) | 0.000 (0) | 1.000 (3) | 0.000 (0) |

Per-class precision/recall/F1:

| label | support | predicted | precision | recall | F1 |
|---|---:|---:|---:|---:|---:|
| included | 35 | 22 | 0.955 | 0.600 | 0.737 |
| not applicable | 1 | 0 | 0.000 | 0.000 | 0.000 |
| not enough information | 27 | 44 | 0.591 | 0.963 | 0.732 |
| not included | 3 | 0 | 0.000 | 0.000 | 0.000 |

### finetuned - validation - exclusion criteria

`accuracy=0.743` `macro-F1 (declared)=0.319` `macro-F1 (observed)=0.319` `n=140`

| gold \ pred | excluded | not applicable | not enough information | not excluded |
|---|---|---|---|---|
| **excluded** | 0.000 (0) | 0.000 (0) | 0.000 (0) | 1.000 (2) |
| **not applicable** | 0.000 (0) | 0.273 (3) | 0.000 (0) | 0.727 (8) |
| **not enough information** | 0.000 (0) | 0.000 (0) | 0.000 (0) | 1.000 (26) |
| **not excluded** | 0.000 (0) | 0.000 (0) | 0.000 (0) | 1.000 (101) |

Per-class precision/recall/F1:

| label | support | predicted | precision | recall | F1 |
|---|---:|---:|---:|---:|---:|
| excluded | 2 | 0 | 0.000 | 0.000 | 0.000 |
| not applicable | 11 | 3 | 1.000 | 0.273 | 0.429 |
| not enough information | 26 | 0 | 0.000 | 0.000 | 0.000 |
| not excluded | 101 | 137 | 0.737 | 1.000 | 0.849 |

##### finetuned - validation - evidence retrieval

| view | n | precision | recall | F1 | exact match |
|---|---:|---:|---:|---:|---:|
| pooled | 206 | 0.593 | 0.635 | 0.613 | 0.689 |
| gold_present | 90 | 0.849 | 0.635 | 0.726 | 0.600 |
| no-gold consistency | 116 | - | - | - | empty-pred rate 0.759 |
| per-record macro F1 (gold-present) | 90 | - | - | 0.711 | - |


## 3. Evidence retrieval

Pooled = all parsed records; gold-present = records with non-empty gold evidence only; per-record macro = mean per-record F1 over gold-present records. No-gold consistency is the rate at which predictions also cite no evidence on records with empty gold evidence; it is not retrieval quality and earns no credit.

| model | split | view | n | precision | recall | F1 | exact-match rate / consistency |
|---|---|---|---:|---:|---:|---:|---:|
| baseline | test | pooled | 206 | 0.476 | 0.415 | 0.443 | 0.714 |
| baseline | test | gold-present | 61 | 0.696 | 0.415 | 0.520 | 0.377 |
| baseline | test | per-record macro | 61 | - | - | 0.469 | - |
| baseline | test | no-gold consistency | 145 | - | - | - | empty-pred rate 0.855 |
| baseline | validation | pooled | 206 | 0.369 | 0.417 | 0.392 | 0.558 |
| baseline | validation | gold-present | 90 | 0.585 | 0.417 | 0.487 | 0.367 |
| baseline | validation | per-record macro | 90 | - | - | 0.465 | - |
| baseline | validation | no-gold consistency | 116 | - | - | - | empty-pred rate 0.707 |
| finetuned | test | pooled | 206 | 0.496 | 0.745 | 0.596 | 0.743 |
| finetuned | test | gold-present | 61 | 0.729 | 0.745 | 0.737 | 0.607 |
| finetuned | test | per-record macro | 61 | - | - | 0.748 | - |
| finetuned | test | no-gold consistency | 145 | - | - | - | empty-pred rate 0.800 |
| finetuned | validation | pooled | 206 | 0.593 | 0.635 | 0.613 | 0.689 |
| finetuned | validation | gold-present | 90 | 0.849 | 0.635 | 0.726 | 0.600 |
| finetuned | validation | per-record macro | 90 | - | - | 0.711 | - |
| finetuned | validation | no-gold consistency | 116 | - | - | - | empty-pred rate 0.759 |

## 4. Contradiction cases

Heuristic review flags, ranked by severity. Flagged lists show the most severe cases for manual inspection.

### baseline - test

| flag | count |
|---|---:|
| duplicate_evidence_ids | 1 |
| eligible_with_negated_evidence | 9 |
| not excluded_without_evidence | 1 |
| not included_without_evidence | 4 |

Conclusive predictions (included/excluded/not included/not excluded): 54, without cited evidence: 5 (rate 0.093). Absence of cited evidence is not proof of eligibility or ineligibility.

Most severe flagged cases:

| flag | patient | ann | gold | predicted | cited ids | detail |
|---|---|---|---|---|---|---|
| not excluded_without_evidence | sigir-201421 | 390 | not enough information | not excluded | - | Conclusive eligibility prediction with no cited evidence; absent evidence is not proof either way. |
| eligible_with_negated_evidence | sigir-201410 | 160 | included | included | 3 | Eligibility predicted while cited sentences contain negation cues. |
| eligible_with_negated_evidence | sigir-201410 | 178 | included | included | 3 | Eligibility predicted while cited sentences contain negation cues. |
| not included_without_evidence | sigir-201411 | 191 | not enough information | not included | - | Conclusive eligibility prediction with no cited evidence; absent evidence is not proof either way. |
| eligible_with_negated_evidence | sigir-201411 | 195 | included | included | 0 | Eligibility predicted while cited sentences contain negation cues. |
| not included_without_evidence | sigir-201417 | 305 | not enough information | not included | - | Conclusive eligibility prediction with no cited evidence; absent evidence is not proof either way. |
| eligible_with_negated_evidence | sigir-201417 | 308 | included | included | 5 | Eligibility predicted while cited sentences contain negation cues. |
| eligible_with_negated_evidence | sigir-201421 | 381 | included | included | 4 | Eligibility predicted while cited sentences contain negation cues. |
| eligible_with_negated_evidence | sigir-201421 | 382 | included | included | 2, 3, 4 | Eligibility predicted while cited sentences contain negation cues. |
| eligible_with_negated_evidence | sigir-201421 | 386 | included | included | 4 | Eligibility predicted while cited sentences contain negation cues. |

### baseline - validation

| flag | count |
|---|---:|
| eligible_with_negated_evidence | 23 |
| included_without_evidence | 1 |
| not included_without_evidence | 5 |

Conclusive predictions (included/excluded/not included/not excluded): 81, without cited evidence: 6 (rate 0.074). Absence of cited evidence is not proof of eligibility or ineligibility.

Most severe flagged cases:

| flag | patient | ann | gold | predicted | cited ids | detail |
|---|---|---|---|---|---|---|
| included_without_evidence | sigir-201522 | 944 | included | included | - | Conclusive eligibility prediction with no cited evidence; absent evidence is not proof either way. |
| eligible_with_negated_evidence | sigir-20141 | 1 | included | included | 1 | Eligibility predicted while cited sentences contain negation cues. |
| eligible_with_negated_evidence | sigir-20141 | 3 | included | included | 8 | Eligibility predicted while cited sentences contain negation cues. |
| eligible_with_negated_evidence | sigir-20141 | 4 | included | included | 8 | Eligibility predicted while cited sentences contain negation cues. |
| not included_without_evidence | sigir-201415 | 235 | not enough information | not included | - | Conclusive eligibility prediction with no cited evidence; absent evidence is not proof either way. |
| eligible_with_negated_evidence | sigir-201415 | 246 | included | included | 6 | Eligibility predicted while cited sentences contain negation cues. |
| not included_without_evidence | sigir-201415 | 249 | not enough information | not included | - | Conclusive eligibility prediction with no cited evidence; absent evidence is not proof either way. |
| eligible_with_negated_evidence | sigir-201415 | 251 | not excluded | not excluded | 3 | Eligibility predicted while cited sentences contain negation cues. |
| not included_without_evidence | sigir-20142 | 31 | not included | not included | - | Conclusive eligibility prediction with no cited evidence; absent evidence is not proof either way. |
| eligible_with_negated_evidence | sigir-20142 | 37 | included | included | 6 | Eligibility predicted while cited sentences contain negation cues. |

### finetuned - test

| flag | count |
|---|---:|
| eligible_with_negated_evidence | 15 |
| not excluded_without_evidence | 106 |
| not included_without_evidence | 3 |

Conclusive predictions (included/excluded/not included/not excluded): 150, without cited evidence: 109 (rate 0.727). Absence of cited evidence is not proof of eligibility or ineligibility.

Most severe flagged cases:

| flag | patient | ann | gold | predicted | cited ids | detail |
|---|---|---|---|---|---|---|
| not excluded_without_evidence | sigir-201410 | 165 | not applicable | not excluded | - | Conclusive eligibility prediction with no cited evidence; absent evidence is not proof either way. |
| not excluded_without_evidence | sigir-201410 | 167 | not excluded | not excluded | - | Conclusive eligibility prediction with no cited evidence; absent evidence is not proof either way. |
| not excluded_without_evidence | sigir-201410 | 168 | not excluded | not excluded | - | Conclusive eligibility prediction with no cited evidence; absent evidence is not proof either way. |
| not excluded_without_evidence | sigir-201410 | 170 | not excluded | not excluded | - | Conclusive eligibility prediction with no cited evidence; absent evidence is not proof either way. |
| not excluded_without_evidence | sigir-201410 | 171 | not excluded | not excluded | - | Conclusive eligibility prediction with no cited evidence; absent evidence is not proof either way. |
| not excluded_without_evidence | sigir-201410 | 172 | not excluded | not excluded | - | Conclusive eligibility prediction with no cited evidence; absent evidence is not proof either way. |
| not excluded_without_evidence | sigir-201410 | 174 | not enough information | not excluded | - | Conclusive eligibility prediction with no cited evidence; absent evidence is not proof either way. |
| not excluded_without_evidence | sigir-201410 | 180 | not excluded | not excluded | - | Conclusive eligibility prediction with no cited evidence; absent evidence is not proof either way. |
| not excluded_without_evidence | sigir-201410 | 182 | not excluded | not excluded | - | Conclusive eligibility prediction with no cited evidence; absent evidence is not proof either way. |
| not excluded_without_evidence | sigir-201410 | 183 | not excluded | not excluded | - | Conclusive eligibility prediction with no cited evidence; absent evidence is not proof either way. |

### finetuned - validation

| flag | count |
|---|---:|
| eligible_with_negated_evidence | 32 |
| not excluded_without_evidence | 88 |

Conclusive predictions (included/excluded/not included/not excluded): 159, without cited evidence: 88 (rate 0.553). Absence of cited evidence is not proof of eligibility or ineligibility.

Most severe flagged cases:

| flag | patient | ann | gold | predicted | cited ids | detail |
|---|---|---|---|---|---|---|
| not excluded_without_evidence | sigir-20141 | 6 | not excluded | not excluded | - | Conclusive eligibility prediction with no cited evidence; absent evidence is not proof either way. |
| not excluded_without_evidence | sigir-20141 | 7 | not excluded | not excluded | - | Conclusive eligibility prediction with no cited evidence; absent evidence is not proof either way. |
| not excluded_without_evidence | sigir-20141 | 9 | not excluded | not excluded | - | Conclusive eligibility prediction with no cited evidence; absent evidence is not proof either way. |
| not excluded_without_evidence | sigir-20141 | 10 | not excluded | not excluded | - | Conclusive eligibility prediction with no cited evidence; absent evidence is not proof either way. |
| not excluded_without_evidence | sigir-20141 | 11 | not excluded | not excluded | - | Conclusive eligibility prediction with no cited evidence; absent evidence is not proof either way. |
| not excluded_without_evidence | sigir-20141 | 12 | not excluded | not excluded | - | Conclusive eligibility prediction with no cited evidence; absent evidence is not proof either way. |
| not excluded_without_evidence | sigir-20141 | 14 | not excluded | not excluded | - | Conclusive eligibility prediction with no cited evidence; absent evidence is not proof either way. |
| not excluded_without_evidence | sigir-20141 | 15 | not enough information | not excluded | - | Conclusive eligibility prediction with no cited evidence; absent evidence is not proof either way. |
| not excluded_without_evidence | sigir-20141 | 18 | not enough information | not excluded | - | Conclusive eligibility prediction with no cited evidence; absent evidence is not proof either way. |
| not excluded_without_evidence | sigir-20141 | 20 | not excluded | not excluded | - | Conclusive eligibility prediction with no cited evidence; absent evidence is not proof either way. |

## 5. Representative errors

### baseline - test
* gold `not excluded` predicted `not applicable` (62 times):

| patient | ann | gold | predicted | cited ids | criterion | explanation |
|---|---|---|---|---|---|---|
| sigir-201410 | 167 | not excluded | not applicable | - | Previous history of upper limb deep vein thrombosis | The note mentions a history of cardiac catheterization via the right femoral artery, but no mention of deep vein thrombosis in either limb. |
| sigir-201410 | 168 | not excluded | not applicable | - | Patients on glibenclamide or nicorandil (these medications may interfere with RIPC) | The patient note does not mention whether the patient is taking glibenclamide or nicorandil. |

* gold `not excluded` predicted `not enough information` (22 times):

| patient | ann | gold | predicted | cited ids | criterion | explanation |
|---|---|---|---|---|---|---|
| sigir-201421 | 388 | not excluded | not enough information | 2, 3 | Active severe SLE flare with central nervous system (CNS) manifestations, active renal lupus, systemic vasculitis, active pericarditis, active pleuritis, active<br>**cited text:** Her lab shows normocytic anemia, thrombocytopenia, a 4/4 positive ANA and anti-dsDNA. Her urine is positive for protein and RBC casts. | The patient has thrombocytopenia, positive ANA and anti-dsDNA, proteinuria and RBC casts, which are suggestive of lupus nephritis, but the note does not explicitly state active renal lupus or other se |
| sigir-201421 | 402 | not excluded | not enough information | - | Current severe infection | The patient note does not mention any current infection. |

* gold `not enough information` predicted `not applicable` (20 times):

| patient | ann | gold | predicted | cited ids | criterion | explanation |
|---|---|---|---|---|---|---|
| sigir-201410 | 174 | not enough information | not applicable | - | Patients with previous axillary surgery | The patient note does not mention any previous axillary surgery. |
| sigir-201417 | 315 | not enough information | not applicable | - | BMI> 40 kg/m2. | The patient's BMI is not mentioned in the provided clinical note. |

* gold `not excluded` predicted `excluded` (7 times):

| patient | ann | gold | predicted | cited ids | criterion | explanation |
|---|---|---|---|---|---|---|
| sigir-201410 | 166 | not excluded | excluded | 2 | Significant upper limb peripheral arterial disease<br>**cited text:** Upon examination she has a pulsatile mass in her right groin with loss of distal pulses, and auscultation reveals a bruit over the point at which the right femoral artery was entered. | The patient has a pulsatile mass in her right groin with loss of distal pulses and a bruit over the femoral artery entry site, suggesting significant arterial compromise. |
| sigir-201410 | 173 | not excluded | excluded | 2 | Patients previously enrolled in the trial representing for a further procedure<br>**cited text:** Upon examination she has a pulsatile mass in her right groin with loss of distal pulses, and auscultation reveals a bruit over the point at which the right femoral artery was entered. | The patient underwent cardiac catheterization via the right femoral artery earlier in the morning, and now has a pulsatile mass, loss of distal pulses, and a bruit at the entry site, suggesting a pote |

* gold `not enough information` predicted `not included` (7 times):

| patient | ann | gold | predicted | cited ids | criterion | explanation |
|---|---|---|---|---|---|---|
| sigir-201410 | 175 | not enough information | not included | 2 | Chronic renal failure requiring hemodialysis.<br>**cited text:** Upon examination she has a pulsatile mass in her right groin with loss of distal pulses, and auscultation reveals a bruit over the point at which the right femoral artery was entered. | The patient's presentation is consistent with a potential vascular complication following femoral artery catheterization, but there is no mention of chronic renal failure or hemodialysis. |
| sigir-201411 | 191 | not enough information | not included | - | bilateral pulmonary infiltrates or a chest radiograph consistent with edema; | The patient's chest status is not mentioned in the provided note. There is no information about pulmonary infiltrates or edema. |

Missing information (`not enough information`) handling:

gold `not enough information`: 37/66 gold records were mispredicted (`over` rows predicted NEI when gold is a conclusive label: 27).

| patient | ann | gold | predicted | cited ids | criterion | explanation |
|---|---|---|---|---|---|---|
| sigir-201410 | 174 | not enough information | not applicable | - | Patients with previous axillary surgery | The patient note does not mention any previous axillary surgery. |
| sigir-201410 | 175 | not enough information | not included | 2 | Chronic renal failure requiring hemodialysis. | The patient's presentation is consistent with a potential vascular complication following femoral artery catheterization, but there is no mention of chronic renal failure or hemodialysis. |
| sigir-201411 | 189 | not enough information | included | 3 | at least 1 of 3 signs of hypoperfusion (urine output < 0.5mL/kg of body weight per hour for 1 hour or more; neurologic dysfunction defined by confusion, psychos | The patient's blood pressure is 80/60, which is low and suggests hypoperfusion. |
| sigir-201411 | 185 | included | not enough information | - | Patients were included if they were diagnosed with Shock, Acute respiratory distress syndrome (ARDS), or both. | The note mentions the patient is tachypneic and tachycardic with a low blood pressure (80/60), which could indicate shock. However, there is no definitive diagnosis of shock or ARDS stated in the note |
| sigir-201411 | 188 | not included | not enough information | 3 | The use of vasopressors to maintain a systolic blood pressure of at least 90mmHg despite fluid resuscitation, low dose of dopamine (≤ 5 μg/kg per minute), or do | The patient's blood pressure is 80/60, which is below the required 90mmHg systolic blood pressure for vasopressor use. However, the note does not state whether vasopressors have been used or if fluid  |

Not-applicable handling:

gold `not applicable`: 3/15 mispredicted.

| patient | ann | gold | predicted | cited ids | criterion | explanation |
|---|---|---|---|---|---|---|
| sigir-20158 | 693 | not applicable | not enough information | - | Suspected diagnosis of a sleep disorder other than OSA (i.e., periodic leg movements, narcolepsy, insomnia, central sleep apnea, sleep hypoventilation syndrome) | The note mentions nighttime snoring, pauses in breathing, and restlessness, which are suggestive of OSA, but does not explicitly state a suspected diagnosis of another sleep disorder. Therefore, it's  |
| sigir-201513 | 745 | not applicable | not included | - | female patients have adequate birth-control method | The patient is a 5-year-old boy, not a female, so the inclusion criterion regarding female patients and birth control is not applicable. |
| sigir-201513 | 767 | not applicable | not enough information | - | Symptoms that collectively suggest nonstreptococcal T/P (eg, laryngitis, coryza, conjunctivitis, diarrhea, cough) | The note mentions dysphagia, drooling, fever, and vocal changes, but does not provide enough information to determine if these symptoms collectively suggest nonstreptococcal T/P. |

### baseline - validation
* gold `not excluded` predicted `not applicable` (53 times):

| patient | ann | gold | predicted | cited ids | criterion | explanation |
|---|---|---|---|---|---|---|
| sigir-20141 | 13 | not excluded | not applicable | - | IDDM (Type-1 diabetes mellitus) | The patient denies a family history of heart disease, diabetes, or hypercholesterolemia, but the note does not explicitly state whether or not she has IDDM. |
| sigir-20141 | 16 | not excluded | not applicable | - | Known hypersensitivity to nicorandil | The patient note does not mention any known hypersensitivity to nicorandil. |

* gold `not excluded` predicted `not enough information` (28 times):

| patient | ann | gold | predicted | cited ids | criterion | explanation |
|---|---|---|---|---|---|---|
| sigir-20141 | 6 | not excluded | not enough information | - | Valvular heart disease and cardiomyopathy | The note does not mention any valvular heart disease or cardiomyopathy, but it also does not explicitly exclude these conditions. Therefore, there is not enough information to determine if the patient |
| sigir-20141 | 7 | not excluded | not enough information | - | Myocardial infarction in < 6 months | The note does not state whether the patient has had a myocardial infarction in the past 6 months. |

* gold `not excluded` predicted `excluded` (12 times):

| patient | ann | gold | predicted | cited ids | criterion | explanation |
|---|---|---|---|---|---|---|
| sigir-20141 | 5 | not excluded | excluded | 3 | Hypertension of > 170/100 mm of Hg<br>**cited text:** She is known to have hypertension and obesity. | The patient is known to have hypertension. |
| sigir-20141 | 25 | not excluded | excluded | 3 | Severe/malignant/secondary hypertensive patients<br>**cited text:** She is known to have hypertension and obesity. | The patient is known to have hypertension, which is considered a severe/malignant/secondary hypertensive condition according to the exclusion criterion. |

* gold `not enough information` predicted `included` (12 times):

| patient | ann | gold | predicted | cited ids | criterion | explanation |
|---|---|---|---|---|---|---|
| sigir-201420 | 375 | not enough information | included | 1, 4 | Trauma patients undergoing DC/OA management for traumatic injury<br>**cited text:** She has sustained multiple injuries including upper and lower extremity fractures. Physical examination reveals a tender abdomen with guarding and rebound in all four quadrants. | The patient sustained multiple injuries including fractures from a car accident, and has a tender abdomen with guarding and rebound, indicating trauma. |
| sigir-201427 | 502 | not enough information | included | 5 | WHO performance status ≤1<br>**cited text:** The patient will provide informed consent, and will comply with the trial protocol without any practical issues. | The note states the patient will comply with the trial protocol without any practical issues, implying they are eligible. |

* gold `not enough information` predicted `not applicable` (8 times):

| patient | ann | gold | predicted | cited ids | criterion | explanation |
|---|---|---|---|---|---|---|
| sigir-20141 | 15 | not enough information | not applicable | - | Pregnant and nursing women | The patient's pregnancy status is not mentioned in the provided clinical note. |
| sigir-20141 | 19 | not enough information | not applicable | - | Patients in whom beta blockers are contraindicated | The patient is currently taking no medications, so beta blockers are not a factor. |

Missing information (`not enough information`) handling:

gold `not enough information`: 31/53 gold records were mispredicted (`over` rows predicted NEI when gold is a conclusive label: 32).

| patient | ann | gold | predicted | cited ids | criterion | explanation |
|---|---|---|---|---|---|---|
| sigir-20141 | 8 | not enough information | excluded | 8 | Unstable angina | The patient presents with acute chest pain, which is a concern for unstable angina, and is being evaluated in the ER. Therefore, she is excluded from the trial. |
| sigir-20141 | 15 | not enough information | not applicable | - | Pregnant and nursing women | The patient's pregnancy status is not mentioned in the provided clinical note. |
| sigir-20141 | 19 | not enough information | not applicable | - | Patients in whom beta blockers are contraindicated | The patient is currently taking no medications, so beta blockers are not a factor. |
| sigir-20141 | 6 | not excluded | not enough information | - | Valvular heart disease and cardiomyopathy | The note does not mention any valvular heart disease or cardiomyopathy, but it also does not explicitly exclude these conditions. Therefore, there is not enough information to determine if the patient |
| sigir-20141 | 7 | not excluded | not enough information | - | Myocardial infarction in < 6 months | The note does not state whether the patient has had a myocardial infarction in the past 6 months. |

Not-applicable handling:

gold `not applicable`: 4/12 mispredicted.

| patient | ann | gold | predicted | cited ids | criterion | explanation |
|---|---|---|---|---|---|---|
| sigir-20142 | 33 | not applicable | not enough information | - | full of physical examination and laboratory examination have been carried out after one week，but still cannot make a definite diagnosis | The note mentions a chest x-ray and physical examination, but doesn't state that a full week has passed or that a definite diagnosis hasn't been made. Therefore, the inclusion criterion is not met. |
| sigir-201430 | 558 | not applicable | not excluded | 8 | Inability to provide informed consent. | The note states the patient 'will provide informed consent.' |
| sigir-201515 | 801 | not applicable | excluded | 2 | Definite small-vessel etiology by history or imaging | The transesophageal echocardiogram and magnetic resonance angiogram of brain and great vessels found no evidence of abnormalities, suggesting a definite small-vessel etiology is not present. |

### finetuned - test
* gold `not enough information` predicted `not excluded` (27 times):

| patient | ann | gold | predicted | cited ids | criterion | explanation |
|---|---|---|---|---|---|---|
| sigir-201410 | 174 | not enough information | not excluded | - | Patients with previous axillary surgery | - |
| sigir-201411 | 193 | not enough information | not excluded | 3 | Patients were moribund.<br>**cited text:** Her body temperature is normal and her blood pressure is 80/60. | - |

* gold `not applicable` predicted `not excluded` (10 times):

| patient | ann | gold | predicted | cited ids | criterion | explanation |
|---|---|---|---|---|---|---|
| sigir-201410 | 165 | not applicable | not excluded | - | Pregnancy | - |
| sigir-201513 | 750 | not applicable | not excluded | - | pregnancy | - |

* gold `included` predicted `not enough information` (8 times):

| patient | ann | gold | predicted | cited ids | criterion | explanation |
|---|---|---|---|---|---|---|
| sigir-201410 | 176 | included | not enough information | 0 | No medical history of central vena catheterization.<br>**cited text:** A physician is called to see a 67-year-old woman who underwent cardiac catheterization via the right femoral artery earlier in the morning. | - |
| sigir-201411 | 185 | included | not enough information | 2, 3 | Patients were included if they were diagnosed with Shock, Acute respiratory distress syndrome (ARDS), or both.<br>**cited text:** On examination she is pale and in moderate discomfort, as well as tachypneic and tachycardic. Her body temperature is normal and her blood pressure is 80/60. | - |

* gold `not included` predicted `not enough information` (4 times):

| patient | ann | gold | predicted | cited ids | criterion | explanation |
|---|---|---|---|---|---|---|
| sigir-201410 | 163 | not included | not enough information | 0 | Patients undergoing endovascular abdominal aneurysm repair<br>**cited text:** A physician is called to see a 67-year-old woman who underwent cardiac catheterization via the right femoral artery earlier in the morning. | - |
| sigir-201410 | 164 | not included | not enough information | 0 | Patients undergoing surgical lower limb revascularisation (suprainguinal or infrainguinal)<br>**cited text:** A physician is called to see a 67-year-old woman who underwent cardiac catheterization via the right femoral artery earlier in the morning. | - |

* gold `not included` predicted `not applicable` (1 times):

| patient | ann | gold | predicted | cited ids | criterion | explanation |
|---|---|---|---|---|---|---|
| sigir-201410 | 161 | not included | not applicable | 0 | Patients undergoing elective carotid endarterectomy<br>**cited text:** A physician is called to see a 67-year-old woman who underwent cardiac catheterization via the right femoral artery earlier in the morning. | - |

Missing information (`not enough information`) handling:

gold `not enough information`: 28/66 gold records were mispredicted (`over` rows predicted NEI when gold is a conclusive label: 13).

| patient | ann | gold | predicted | cited ids | criterion | explanation |
|---|---|---|---|---|---|---|
| sigir-201410 | 174 | not enough information | not excluded | - | Patients with previous axillary surgery | - |
| sigir-201410 | 175 | not enough information | not included | - | Chronic renal failure requiring hemodialysis. | - |
| sigir-201411 | 193 | not enough information | not excluded | 3 | Patients were moribund. | - |
| sigir-201410 | 163 | not included | not enough information | 0 | Patients undergoing endovascular abdominal aneurysm repair | - |
| sigir-201410 | 164 | not included | not enough information | 0 | Patients undergoing surgical lower limb revascularisation (suprainguinal or infrainguinal) | - |

Not-applicable handling:

gold `not applicable`: 11/15 mispredicted.

| patient | ann | gold | predicted | cited ids | criterion | explanation |
|---|---|---|---|---|---|---|
| sigir-201410 | 165 | not applicable | not excluded | - | Pregnancy | - |
| sigir-20158 | 683 | not applicable | not excluded | - | Previous tonsillectomy, including partial tonsillectomy | - |
| sigir-20158 | 693 | not applicable | not excluded | 0 | Suspected diagnosis of a sleep disorder other than OSA (i.e., periodic leg movements, narcolepsy, insomnia, central sleep apnea, sleep hypoventilation syndrome) | - |

### finetuned - validation
* gold `not enough information` predicted `not excluded` (26 times):

| patient | ann | gold | predicted | cited ids | criterion | explanation |
|---|---|---|---|---|---|---|
| sigir-20141 | 8 | not enough information | not excluded | 0, 1 | Unstable angina<br>**cited text:** A 58-year-old African-American woman presents to the ER with episodic pressing/burning anterior chest pain that began two days earlier for the first time in her life. The pain started while she was walking, radiates to the back, and is accompanied by nausea, diaphoresis and mild dyspnea, but is not  | - |
| sigir-20141 | 15 | not enough information | not excluded | - | Pregnant and nursing women | - |

* gold `included` predicted `not enough information` (14 times):

| patient | ann | gold | predicted | cited ids | criterion | explanation |
|---|---|---|---|---|---|---|
| sigir-20141 | 22 | included | not enough information | 3 | Clinical diagnosis of one or more risk factors, such as diabetes, smoking habit, lipid metabolism abnormality, history of ischemic heart disease (IHD) or cerebr<br>**cited text:** She is known to have hypertension and obesity. | - |
| sigir-201420 | 366 | included | not enough information | 4 | Has experienced abdominal trauma, blunt or penetrating, requiring open abdominal laparotomy with primary closure<br>**cited text:** Physical examination reveals a tender abdomen with guarding and rebound in all four quadrants. | - |

* gold `not applicable` predicted `not excluded` (8 times):

| patient | ann | gold | predicted | cited ids | criterion | explanation |
|---|---|---|---|---|---|---|
| sigir-20141 | 24 | not applicable | not excluded | - | Patients with IHD within 6 months after percutaneous coronary intervention(PCI), and who are stable but are going to implement PCI or coronary artery bypass gra | - |
| sigir-201430 | 551 | not applicable | not excluded | - | Pregnancy | - |

* gold `not included` predicted `not enough information` (3 times):

| patient | ann | gold | predicted | cited ids | criterion | explanation |
|---|---|---|---|---|---|---|
| sigir-20142 | 31 | not included | not enough information | 0 | patients have fever more than one week<br>**cited text:** An 8-year-old male presents in March to the ER with fever up to 39 C, dyspnea and cough for 2 days. | - |
| sigir-201522 | 934 | not included | not enough information | - | Taking anti-tuberculosis medicines (or NTM meds at NIH CC) during the time sputa are provided | - |

* gold `excluded` predicted `not excluded` (2 times):

| patient | ann | gold | predicted | cited ids | criterion | explanation |
|---|---|---|---|---|---|---|
| sigir-201415 | 263 | excluded | not excluded | 4 | 13. Evidence of uterine pathology other than leiomyoma<br>**cited text:** The ultrasound reports enlarged uterus (12 cm x 9 cm x 7 cms) with multiple cystic areas in the interior. | - |
| sigir-20142 | 41 | excluded | not excluded | 4 | Respiratory distress<br>**cited text:** On examination he is in respiratory distress and has bronchial respiratory sounds on the left. | - |

Missing information (`not enough information`) handling:

gold `not enough information`: 27/53 gold records were mispredicted (`over` rows predicted NEI when gold is a conclusive label: 18).

| patient | ann | gold | predicted | cited ids | criterion | explanation |
|---|---|---|---|---|---|---|
| sigir-20141 | 8 | not enough information | not excluded | 0, 1 | Unstable angina | - |
| sigir-20141 | 15 | not enough information | not excluded | - | Pregnant and nursing women | - |
| sigir-20141 | 18 | not enough information | not excluded | - | Patients not eligible for Tc 99m SPECT | - |
| sigir-20141 | 22 | included | not enough information | 3 | Clinical diagnosis of one or more risk factors, such as diabetes, smoking habit, lipid metabolism abnormality, history of ischemic heart disease (IHD) or cerebr | - |
| sigir-20142 | 31 | not included | not enough information | 0 | patients have fever more than one week | - |

Not-applicable handling:

gold `not applicable`: 9/12 mispredicted.

| patient | ann | gold | predicted | cited ids | criterion | explanation |
|---|---|---|---|---|---|---|
| sigir-20141 | 24 | not applicable | not excluded | - | Patients with IHD within 6 months after percutaneous coronary intervention(PCI), and who are stable but are going to implement PCI or coronary artery bypass gra | - |
| sigir-20142 | 33 | not applicable | not enough information | 0, 1, 2, 3, 4, 5 | full of physical examination and laboratory examination have been carried out after one week，but still cannot make a definite diagnosis | - |
| sigir-201430 | 551 | not applicable | not excluded | - | Pregnancy | - |
