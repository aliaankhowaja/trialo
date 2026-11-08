# Task 4: tiled attention with online softmax

device: `NVIDIA A100-SXM4-40GB` (42.4 GB), torch `2.14.1+cu130`, CUDA `13.0`, precision `bf16`, tile `128x128`, B=2 H=16, causal.

## Agreement (numerical correctness)

- fp32, tiled vs naive reference, 4 shapes (seq up to 512): max abs diff 9.54e-07; all shapes agree to rtol 1e-5 / atol 1e-5 (verified in tests).

| precision | comparator | seq | dim | max abs | max rel | mean abs |
|---|---|---:|---:|---:|---:|---:|
| fp32 | naive_vs_tiled | 16 | 64 | 3.576e-07 | 1.455e-03 | 2.150e-08 |
| fp32 | naive_vs_tiled | 128 | 64 | 9.537e-07 | 1.953e-02 | 2.856e-08 |
| fp32 | naive_vs_tiled | 512 | 64 | 7.153e-07 | 1.293e+02 | 2.615e-08 |
| fp32 | naive_vs_tiled | 128 | 128 | 6.557e-07 | 2.050e-02 | 2.824e-08 |
| fp16 | tiled_vs_sdpa_MATH | 128 | 64 | 4.883e-04 | 1.192e+05 | 4.410e-08 |
| fp16 | tiled_vs_sdpa_CUDNN_ATTENTION | 128 | 64 | 1.953e-03 | 2.604e+01 | 2.700e-05 |
| fp16 | tiled_vs_sdpa_EFFICIENT_ATTENTION | 128 | 64 | 1.953e-03 | 2.604e+01 | 2.700e-05 |
| fp16 | tiled_vs_sdpa_FLASH_ATTENTION | 128 | 64 | 1.953e-03 | 2.604e+01 | 2.729e-05 |
| fp16 | tiled_vs_sdpa_MATH | 1024 | 64 | 9.766e-04 | 2.000e-01 | 3.277e-08 |
| fp16 | tiled_vs_sdpa_CUDNN_ATTENTION | 1024 | 64 | 1.953e-03 | 4.470e+06 | 1.274e-05 |
| fp16 | tiled_vs_sdpa_EFFICIENT_ATTENTION | 1024 | 64 | 1.953e-03 | 4.470e+06 | 1.274e-05 |
| fp16 | tiled_vs_sdpa_FLASH_ATTENTION | 1024 | 64 | 1.953e-03 | 3.470e+02 | 1.285e-05 |
| bf16 | tiled_vs_sdpa_MATH | 128 | 64 | 3.906e-03 | 4.348e-02 | 3.376e-08 |
| bf16 | tiled_vs_sdpa_CUDNN_ATTENTION | 128 | 64 | 7.812e-03 | 2.866e+03 | 2.173e-04 |
| bf16 | tiled_vs_sdpa_EFFICIENT_ATTENTION | 128 | 64 | 7.812e-03 | 2.866e+03 | 2.173e-04 |
| bf16 | tiled_vs_sdpa_FLASH_ATTENTION | 128 | 64 | 7.812e-03 | 2.866e+03 | 2.196e-04 |
| bf16 | tiled_vs_sdpa_MATH | 1024 | 64 | 1.953e-03 | 4.911e-01 | 2.801e-08 |
| bf16 | tiled_vs_sdpa_CUDNN_ATTENTION | 1024 | 64 | 1.562e-02 | 7.670e+02 | 1.015e-04 |
| bf16 | tiled_vs_sdpa_EFFICIENT_ATTENTION | 1024 | 64 | 1.562e-02 | 7.670e+02 | 1.015e-04 |
| bf16 | tiled_vs_sdpa_FLASH_ATTENTION | 1024 | 64 | 1.562e-02 | 7.606e+03 | 1.026e-04 |

## SDPA backend availability

- bf16 CUDNN_ATTENTION: available
- bf16 EFFICIENT_ATTENTION: available
- bf16 FLASH_ATTENTION: available
- bf16 MATH: available
- fp16 CUDNN_ATTENTION: available
- fp16 EFFICIENT_ATTENTION: available
- fp16 FLASH_ATTENTION: available
- fp16 MATH: available

Note on the max-rel column: attention outputs contain entries that are
exactly zero or extremely small (softmax mass on masked-adjacent keys),
so the raw relative error at those entries explodes (values above 1e+2 in
the table). The trustworthy agreement metric is the absolute error against
the output scale: fp16 outputs are of magnitude ~1e+0 rounding at
representable steps ~5e-04, bf16 at ~8e-03, so measured max abs diffs of
2.0e-03 (fp16) and 1.6e-02 (bf16) are
one-to-at-most-two rounding steps. fp32 rows differ by <1e-06 in absolute
terms, well inside the 1e-05 test tolerance.

## Timing and peak memory

Peak memory is the allocated delta over the steady-state allocator level; naive materialises the full fp32 softmax so its peak grows like `O(S^2)`, tiled keeps `O(S * Dh)` buffers, and the fused SDPA kernels need only fixed per-tile intermediates.

| dtype | seq | dh | impl | median s | min s | p90 s | TFLOP/s | peak GB |
|---|---|---:|---|---:|---:|---:|---:|---:|
| fp16 | 256 | 64 | naive | 0.0005 | 0.0004 | 0.0005 | 0.5 | 0.026 |
| fp16 | 256 | 64 | tiled | 0.0019 | 0.0019 | 0.0019 | 0.1 | 0.018 |
| fp16 | 256 | 64 | sdpa_auto | 0.0001 | 0.0001 | 0.0001 | 4.3 | 0.001 |
| fp16 | 256 | 64 | sdpa_math | 0.0005 | 0.0005 | 0.0005 | 0.5 | 0.028 |
| fp16 | 256 | 64 | sdpa_cudnn_attention | 0.0001 | 0.0001 | 0.0002 | 2.3 | 0.001 |
| fp16 | 256 | 64 | sdpa_efficient_attention | 0.0001 | 0.0001 | 0.0001 | 2.6 | 0.001 |
| fp16 | 256 | 64 | sdpa_flash_attention | 0.0001 | 0.0001 | 0.0001 | 2.9 | 0.001 |
| fp16 | 256 | 128 | naive | 0.0005 | 0.0004 | 0.0005 | 1.1 | 0.036 |
| fp16 | 256 | 128 | tiled | 0.0019 | 0.0019 | 0.0019 | 0.3 | 0.029 |
| fp16 | 256 | 128 | sdpa_auto | 0.0001 | 0.0001 | 0.0001 | 8.7 | 0.002 |
| fp16 | 256 | 128 | sdpa_math | 0.0005 | 0.0005 | 0.0005 | 1.0 | 0.036 |
| fp16 | 256 | 128 | sdpa_cudnn_attention | 0.0001 | 0.0001 | 0.0002 | 4.4 | 0.002 |
| fp16 | 256 | 128 | sdpa_efficient_attention | 0.0001 | 0.0001 | 0.0001 | 4.7 | 0.002 |
| fp16 | 256 | 128 | sdpa_flash_attention | 0.0001 | 0.0001 | 0.0001 | 5.7 | 0.002 |
| fp16 | 512 | 64 | naive | 0.0007 | 0.0007 | 0.0008 | 1.4 | 0.086 |
| fp16 | 512 | 64 | tiled | 0.0055 | 0.0055 | 0.0056 | 0.2 | 0.026 |
| fp16 | 512 | 64 | sdpa_auto | 0.0001 | 0.0001 | 0.0001 | 15.7 | 0.002 |
| fp16 | 512 | 64 | sdpa_math | 0.0009 | 0.0009 | 0.0009 | 1.2 | 0.093 |
| fp16 | 512 | 64 | sdpa_cudnn_attention | 0.0001 | 0.0001 | 0.0001 | 9.6 | 0.002 |
| fp16 | 512 | 64 | sdpa_efficient_attention | 0.0001 | 0.0001 | 0.0001 | 8.7 | 0.002 |
| fp16 | 512 | 64 | sdpa_flash_attention | 0.0001 | 0.0001 | 0.0001 | 11.5 | 0.002 |
| fp16 | 512 | 128 | naive | 0.0008 | 0.0008 | 0.0008 | 2.6 | 0.105 |
| fp16 | 512 | 128 | tiled | 0.0056 | 0.0055 | 0.0057 | 0.4 | 0.046 |
| fp16 | 512 | 128 | sdpa_auto | 0.0001 | 0.0001 | 0.0001 | 23.2 | 0.004 |
| fp16 | 512 | 128 | sdpa_math | 0.0010 | 0.0010 | 0.0010 | 2.2 | 0.110 |
| fp16 | 512 | 128 | sdpa_cudnn_attention | 0.0001 | 0.0001 | 0.0001 | 17.1 | 0.004 |
| fp16 | 512 | 128 | sdpa_efficient_attention | 0.0001 | 0.0001 | 0.0002 | 16.1 | 0.004 |
| fp16 | 512 | 128 | sdpa_flash_attention | 0.0001 | 0.0001 | 0.0001 | 18.8 | 0.004 |
| fp16 | 1024 | 64 | naive | 0.0021 | 0.0021 | 0.0021 | 2.0 | 0.307 |
| fp16 | 1024 | 64 | tiled | 0.0186 | 0.0185 | 0.0188 | 0.2 | 0.043 |
| fp16 | 1024 | 64 | sdpa_auto | 0.0001 | 0.0001 | 0.0001 | 42.4 | 0.004 |
| fp16 | 1024 | 64 | sdpa_math | 0.0024 | 0.0024 | 0.0024 | 1.8 | 0.340 |
| fp16 | 1024 | 64 | sdpa_cudnn_attention | 0.0002 | 0.0002 | 0.0002 | 26.9 | 0.004 |
| fp16 | 1024 | 64 | sdpa_efficient_attention | 0.0002 | 0.0002 | 0.0002 | 21.9 | 0.004 |
| fp16 | 1024 | 64 | sdpa_flash_attention | 0.0001 | 0.0001 | 0.0001 | 36.0 | 0.004 |
| fp16 | 1024 | 128 | naive | 0.0024 | 0.0024 | 0.0024 | 3.6 | 0.345 |
| fp16 | 1024 | 128 | tiled | 0.0186 | 0.0185 | 0.0188 | 0.5 | 0.082 |
| fp16 | 1024 | 128 | sdpa_auto | 0.0002 | 0.0002 | 0.0002 | 56.1 | 0.009 |
| fp16 | 1024 | 128 | sdpa_math | 0.0027 | 0.0026 | 0.0027 | 3.2 | 0.373 |
| fp16 | 1024 | 128 | sdpa_cudnn_attention | 0.0002 | 0.0002 | 0.0002 | 46.3 | 0.008 |
| fp16 | 1024 | 128 | sdpa_efficient_attention | 0.0003 | 0.0002 | 0.0003 | 33.6 | 0.008 |
| fp16 | 1024 | 128 | sdpa_flash_attention | 0.0002 | 0.0002 | 0.0002 | 47.9 | 0.009 |
| fp16 | 2048 | 64 | naive | 0.0072 | 0.0072 | 0.0073 | 2.4 | 1.153 |
| fp16 | 2048 | 64 | tiled | 0.0665 | 0.0659 | 0.0676 | 0.3 | 0.081 |
| fp16 | 2048 | 64 | sdpa_auto | 0.0002 | 0.0002 | 0.0002 | 93.5 | 0.009 |
| fp16 | 2048 | 64 | sdpa_math | 0.0072 | 0.0072 | 0.0072 | 2.4 | 1.292 |
| fp16 | 2048 | 64 | sdpa_cudnn_attention | 0.0003 | 0.0003 | 0.0003 | 64.2 | 0.008 |
| fp16 | 2048 | 64 | sdpa_efficient_attention | 0.0004 | 0.0004 | 0.0004 | 46.0 | 0.008 |
| fp16 | 2048 | 64 | sdpa_flash_attention | 0.0002 | 0.0002 | 0.0002 | 82.2 | 0.009 |
| fp16 | 2048 | 128 | naive | 0.0073 | 0.0072 | 0.0073 | 4.7 | 1.229 |
| fp16 | 2048 | 128 | tiled | 0.0664 | 0.0660 | 0.0675 | 0.5 | 0.157 |
| fp16 | 2048 | 128 | sdpa_auto | 0.0003 | 0.0003 | 0.0003 | 109.3 | 0.017 |
| fp16 | 2048 | 128 | sdpa_math | 0.0082 | 0.0082 | 0.0082 | 4.2 | 1.359 |
| fp16 | 2048 | 128 | sdpa_cudnn_attention | 0.0003 | 0.0003 | 0.0004 | 102.8 | 0.017 |
| fp16 | 2048 | 128 | sdpa_efficient_attention | 0.0005 | 0.0005 | 0.0005 | 69.5 | 0.017 |
| fp16 | 2048 | 128 | sdpa_flash_attention | 0.0003 | 0.0003 | 0.0004 | 103.6 | 0.017 |
| fp16 | 4096 | 64 | naive | 0.0253 | 0.0252 | 0.0253 | 2.7 | 4.463 |
| fp16 | 4096 | 64 | tiled | 0.2616 | 0.2559 | 0.2629 | 0.3 | 0.156 |
| fp16 | 4096 | 64 | sdpa_auto | 0.0006 | 0.0006 | 0.0007 | 107.0 | 0.017 |
| fp16 | 4096 | 64 | sdpa_math | 0.0284 | 0.0283 | 0.0285 | 2.4 | 5.033 |
| fp16 | 4096 | 64 | sdpa_cudnn_attention | 0.0007 | 0.0007 | 0.0007 | 100.5 | 0.017 |
| fp16 | 4096 | 64 | sdpa_efficient_attention | 0.0011 | 0.0010 | 0.0011 | 64.7 | 0.017 |
| fp16 | 4096 | 64 | sdpa_flash_attention | 0.0005 | 0.0005 | 0.0005 | 127.3 | 0.017 |
| fp16 | 4096 | 128 | naive | 0.0289 | 0.0289 | 0.0289 | 4.8 | 4.614 |
| fp16 | 4096 | 128 | tiled | 0.2571 | 0.2562 | 0.2576 | 0.5 | 0.308 |
| fp16 | 4096 | 128 | sdpa_auto | 0.0011 | 0.0011 | 0.0011 | 124.7 | 0.034 |
| fp16 | 4096 | 128 | sdpa_math | 0.0321 | 0.0319 | 0.0321 | 4.3 | 5.168 |
| fp16 | 4096 | 128 | sdpa_cudnn_attention | 0.0009 | 0.0009 | 0.0009 | 147.0 | 0.034 |
| fp16 | 4096 | 128 | sdpa_efficient_attention | 0.0015 | 0.0015 | 0.0015 | 91.2 | 0.034 |
| fp16 | 4096 | 128 | sdpa_flash_attention | 0.0009 | 0.0009 | 0.0009 | 150.7 | 0.034 |
| bf16 | 256 | 64 | naive | 0.0005 | 0.0005 | 0.0005 | 0.6 | 0.026 |
| bf16 | 256 | 64 | tiled | 0.0019 | 0.0019 | 0.0019 | 0.1 | 0.018 |
| bf16 | 256 | 64 | sdpa_auto | 0.0001 | 0.0001 | 0.0001 | 4.5 | 0.001 |
| bf16 | 256 | 64 | sdpa_math | 0.0005 | 0.0005 | 0.0006 | 0.5 | 0.028 |
| bf16 | 256 | 64 | sdpa_cudnn_attention | 0.0001 | 0.0001 | 0.0001 | 2.4 | 0.001 |
| bf16 | 256 | 64 | sdpa_efficient_attention | 0.0001 | 0.0001 | 0.0001 | 2.6 | 0.001 |
| bf16 | 256 | 64 | sdpa_flash_attention | 0.0001 | 0.0001 | 0.0001 | 3.0 | 0.001 |
| bf16 | 256 | 128 | naive | 0.0005 | 0.0005 | 0.0005 | 1.1 | 0.036 |
| bf16 | 256 | 128 | tiled | 0.0019 | 0.0019 | 0.0019 | 0.3 | 0.029 |
| bf16 | 256 | 128 | sdpa_auto | 0.0001 | 0.0001 | 0.0001 | 8.7 | 0.002 |
| bf16 | 256 | 128 | sdpa_math | 0.0005 | 0.0005 | 0.0005 | 1.0 | 0.036 |
| bf16 | 256 | 128 | sdpa_cudnn_attention | 0.0001 | 0.0001 | 0.0001 | 4.4 | 0.002 |
| bf16 | 256 | 128 | sdpa_efficient_attention | 0.0001 | 0.0001 | 0.0001 | 4.6 | 0.002 |
| bf16 | 256 | 128 | sdpa_flash_attention | 0.0001 | 0.0001 | 0.0001 | 5.5 | 0.002 |
| bf16 | 512 | 64 | naive | 0.0007 | 0.0006 | 0.0007 | 1.6 | 0.086 |
| bf16 | 512 | 64 | tiled | 0.0055 | 0.0055 | 0.0056 | 0.2 | 0.026 |
| bf16 | 512 | 64 | sdpa_auto | 0.0001 | 0.0001 | 0.0001 | 16.8 | 0.002 |
| bf16 | 512 | 64 | sdpa_math | 0.0008 | 0.0008 | 0.0008 | 1.3 | 0.093 |
| bf16 | 512 | 64 | sdpa_cudnn_attention | 0.0001 | 0.0001 | 0.0001 | 9.2 | 0.002 |
| bf16 | 512 | 64 | sdpa_efficient_attention | 0.0001 | 0.0001 | 0.0001 | 9.7 | 0.002 |
| bf16 | 512 | 64 | sdpa_flash_attention | 0.0001 | 0.0001 | 0.0001 | 11.4 | 0.002 |
| bf16 | 512 | 128 | naive | 0.0007 | 0.0007 | 0.0007 | 3.0 | 0.105 |
| bf16 | 512 | 128 | tiled | 0.0056 | 0.0055 | 0.0056 | 0.4 | 0.046 |
| bf16 | 512 | 128 | sdpa_auto | 0.0001 | 0.0001 | 0.0001 | 27.6 | 0.004 |
| bf16 | 512 | 128 | sdpa_math | 0.0009 | 0.0009 | 0.0009 | 2.5 | 0.110 |
| bf16 | 512 | 128 | sdpa_cudnn_attention | 0.0001 | 0.0001 | 0.0001 | 19.4 | 0.004 |
| bf16 | 512 | 128 | sdpa_efficient_attention | 0.0001 | 0.0001 | 0.0001 | 18.1 | 0.004 |
| bf16 | 512 | 128 | sdpa_flash_attention | 0.0001 | 0.0001 | 0.0001 | 20.1 | 0.004 |
| bf16 | 1024 | 64 | naive | 0.0018 | 0.0018 | 0.0018 | 2.3 | 0.307 |
| bf16 | 1024 | 64 | tiled | 0.0186 | 0.0185 | 0.0188 | 0.2 | 0.043 |
| bf16 | 1024 | 64 | sdpa_auto | 0.0001 | 0.0001 | 0.0001 | 50.5 | 0.004 |
| bf16 | 1024 | 64 | sdpa_math | 0.0021 | 0.0021 | 0.0021 | 2.1 | 0.340 |
| bf16 | 1024 | 64 | sdpa_cudnn_attention | 0.0001 | 0.0001 | 0.0002 | 28.8 | 0.004 |
| bf16 | 1024 | 64 | sdpa_efficient_attention | 0.0002 | 0.0002 | 0.0002 | 26.3 | 0.004 |
| bf16 | 1024 | 64 | sdpa_flash_attention | 0.0001 | 0.0001 | 0.0001 | 40.9 | 0.004 |
| bf16 | 1024 | 128 | naive | 0.0021 | 0.0020 | 0.0021 | 4.2 | 0.345 |
| bf16 | 1024 | 128 | tiled | 0.0186 | 0.0186 | 0.0187 | 0.5 | 0.082 |
| bf16 | 1024 | 128 | sdpa_auto | 0.0001 | 0.0001 | 0.0002 | 59.9 | 0.009 |
| bf16 | 1024 | 128 | sdpa_math | 0.0025 | 0.0025 | 0.0025 | 3.5 | 0.373 |
| bf16 | 1024 | 128 | sdpa_cudnn_attention | 0.0002 | 0.0002 | 0.0002 | 49.6 | 0.008 |
| bf16 | 1024 | 128 | sdpa_efficient_attention | 0.0002 | 0.0002 | 0.0002 | 37.6 | 0.008 |
| bf16 | 1024 | 128 | sdpa_flash_attention | 0.0002 | 0.0002 | 0.0002 | 51.3 | 0.009 |
| bf16 | 2048 | 64 | naive | 0.0069 | 0.0069 | 0.0069 | 2.5 | 1.153 |
| bf16 | 2048 | 64 | tiled | 0.0675 | 0.0673 | 0.0677 | 0.3 | 0.081 |
| bf16 | 2048 | 64 | sdpa_auto | 0.0002 | 0.0002 | 0.0002 | 93.2 | 0.009 |
| bf16 | 2048 | 64 | sdpa_math | 0.0072 | 0.0072 | 0.0073 | 2.4 | 1.292 |
| bf16 | 2048 | 64 | sdpa_cudnn_attention | 0.0003 | 0.0003 | 0.0003 | 63.8 | 0.008 |
| bf16 | 2048 | 64 | sdpa_efficient_attention | 0.0004 | 0.0004 | 0.0004 | 46.7 | 0.008 |
| bf16 | 2048 | 64 | sdpa_flash_attention | 0.0002 | 0.0002 | 0.0002 | 83.3 | 0.009 |
| bf16 | 2048 | 128 | naive | 0.0073 | 0.0072 | 0.0073 | 4.7 | 1.229 |
| bf16 | 2048 | 128 | tiled | 0.0672 | 0.0666 | 0.0680 | 0.5 | 0.157 |
| bf16 | 2048 | 128 | sdpa_auto | 0.0003 | 0.0003 | 0.0003 | 112.6 | 0.017 |
| bf16 | 2048 | 128 | sdpa_math | 0.0082 | 0.0082 | 0.0082 | 4.2 | 1.359 |
| bf16 | 2048 | 128 | sdpa_cudnn_attention | 0.0003 | 0.0003 | 0.0004 | 102.3 | 0.017 |
| bf16 | 2048 | 128 | sdpa_efficient_attention | 0.0005 | 0.0005 | 0.0005 | 70.1 | 0.017 |
| bf16 | 2048 | 128 | sdpa_flash_attention | 0.0003 | 0.0003 | 0.0003 | 103.4 | 0.017 |
| bf16 | 4096 | 64 | naive | 0.0253 | 0.0251 | 0.0253 | 2.7 | 4.463 |
| bf16 | 4096 | 64 | tiled | 0.2567 | 0.2530 | 0.2583 | 0.3 | 0.156 |
| bf16 | 4096 | 64 | sdpa_auto | 0.0006 | 0.0006 | 0.0007 | 106.9 | 0.017 |
| bf16 | 4096 | 64 | sdpa_math | 0.0284 | 0.0283 | 0.0284 | 2.4 | 5.033 |
| bf16 | 4096 | 64 | sdpa_cudnn_attention | 0.0007 | 0.0007 | 0.0007 | 100.6 | 0.017 |
| bf16 | 4096 | 64 | sdpa_efficient_attention | 0.0011 | 0.0010 | 0.0011 | 64.7 | 0.017 |
| bf16 | 4096 | 64 | sdpa_flash_attention | 0.0005 | 0.0005 | 0.0005 | 128.2 | 0.017 |
| bf16 | 4096 | 128 | naive | 0.0289 | 0.0289 | 0.0289 | 4.8 | 4.614 |
| bf16 | 4096 | 128 | tiled | 0.2538 | 0.2505 | 0.2590 | 0.5 | 0.308 |
| bf16 | 4096 | 128 | sdpa_auto | 0.0011 | 0.0011 | 0.0011 | 123.4 | 0.034 |
| bf16 | 4096 | 128 | sdpa_math | 0.0321 | 0.0321 | 0.0321 | 4.3 | 5.168 |
| bf16 | 4096 | 128 | sdpa_cudnn_attention | 0.0009 | 0.0009 | 0.0009 | 146.8 | 0.034 |
| bf16 | 4096 | 128 | sdpa_efficient_attention | 0.0015 | 0.0015 | 0.0015 | 91.7 | 0.034 |
| bf16 | 4096 | 128 | sdpa_flash_attention | 0.0009 | 0.0009 | 0.0009 | 152.0 | 0.034 |

## Interpretation

### Why an educational tiled implementation may not outperform a fused kernel

Every tile in the educational path dispatches several eager PyTorch kernels
(matmul, masking, `exp`, accumulate, matmul to values). Python-loop and launch
overhead alone make it slower than the naive reference at these sizes: the naive
reference knocks out a few large high-utilization matmuls, while the tiled loop
issues hundreds of tiny kernels per call that never saturate the GPU. Meanwhile
`tiled` uses far less peak memory because it never materialises the `S x S`
softmax intermediates.

A fused kernel such as FlashAttention or the cuDNN SDPA backend fuses the
whole `QK^T -> mask -> online softmax -> PV` chain into one launch, keeps
tiles resident in registers/shared memory with asynchronous copies, and never
writes `S x S` scores to HBM at all. That removes both the launch overhead and
the global-memory round-trips, so the fused path wins on speed *and* memory.
The educational implementation gets the memory win but cannot get the speed
win without a real kernel (Triton/CUDA).

`torch.compile` and CUDA graphs can recapture part of the speed difference:
they fuse the tile ops into larger kernels or replay them without per-tile
eager dispatch, but they still cannot beat a hand-written kernel that eliminates
the `S x S` HBM traffic and schedules the memory hierarchy explicitly.

### Big-O memory scaling demonstration

log2 of the peak-growth factors between consecutive sequence lengths
(dim 64; slope ~ 2 means O(S^2), slope ~ 1 means O(S)):

| impl | peak GB at S=1024 -> 2048 -> 4096 | growth slope (log2) |
|---|---|---:|
| naive | 0.307 -> 1.153 -> 4.463 | 1.91, 1.95 |
| tiled | 0.043 -> 0.081 -> 0.156 | 0.91, 0.95 |
| sdpa_auto | 0.004 -> 0.009 -> 0.017 | 1.00, 1.00 |
| sdpa_flash_attention | 0.004 -> 0.009 -> 0.017 | 1.00, 1.00 |

The naive reference grows with slope ~ 2 (quadratic in S: the fp32
`S x S` scores plus softmax matrices), the educational tiled loop with
slope ~ 1 (the only per-call state that scales with S is the
`B * H * S * Dh` fp32 output buffer), and the fused kernels likewise
~ 1 because their per-tile buffers are of fixed size.