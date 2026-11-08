"""Task 4: reference scaled dot-product attention and tiled online softmax.

The module provides two mathematically equivalent implementations of causal
scaled dot-product attention on tensors of shape ``(B, H, S, Dh)``:

- ``reference_attention``: compute the full ``(S, S)`` score matrix, apply the
  causal mask with ``-inf``, take a numerically stable softmax, and multiply by
  the values. Simple and exact, but its peak memory is ``O(S^2)`` because the
  score/probability matrices are materialised entirely.

- ``tiled_online_attention``: FlashAttention-style blockwise iteration over
  ``(T_q, T_k)`` tiles of the score matrix. Each query-row group keeps a
  running state ``(m, l, acc)`` -- running maximum, running softmax
  denominator, and running weighted-value accumulator -- updated tile by tile
  (the "online softmax" recurrence):

      m_new   = max(m_old, rowmax(scores_tile))
      p_tile  = exp(scores_tile - m_new)
      l_new   = l_old * exp(m_old - m_new) + sum(p_tile)
      acc_new = acc_old * exp(m_old - m_new) + p_tile @ v_tile

  ``acc / l`` at the end equals the reference softmax attention exactly up to
  floating-point reassociation. No ``S x S`` matrix is ever allocated, so peak
  memory is ``O(S * Dh + T_q * T_k)`` instead of ``O(S^2)``; for causal
  attention, tiles that lie entirely above the diagonal are skipped (their
  softmax contributions are identically zero).

Both implementations upcast the inputs to fp32 internally for numerical
stability and return the output in the input dtype.
"""

from __future__ import annotations

import math
from typing import Any

import torch

__all__ = [
    "agreement_stats",
    "reference_attention",
    "tiled_online_attention",
]


def _validate_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, v must have shape (B, H, S, Dh)")
    if q.device != k.device or q.device != v.device:
        raise ValueError("q, k, v must live on the same device")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("q, k, v must share one dtype")
    if k.shape != q.shape or v.shape != q.shape:
        raise ValueError("self-attention assumed: q, k, v must have identical shapes")


def _resolve_scale(scale: float | None, head_dim: int) -> float:
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)
    scale = float(scale)
    if not math.isfinite(scale):
        raise ValueError("scale must be finite")
    return scale


def _causal_mask(rows: int, columns: int, device) -> torch.Tensor:
    """True where query index >= key index for a square (rows x columns) mask."""
    query_indices = torch.arange(rows, device=device).unsqueeze(-1)  # (rows, 1)
    key_indices = torch.arange(columns, device=device).unsqueeze(0)  # (1, columns)
    return query_indices >= key_indices


def reference_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = True,
    scale: float | None = None,
    return_attention_weights: bool = False,
):
    """Naive scaled dot-product attention with the full ``(S, S)`` matrix.

    Computed in fp32 internally (stable max-subtracted softmax) and returned in
    the input dtype. Peak memory is ``O(S^2)``: the honest educational baseline
    against which the tiled variant is measured.
    """
    _validate_qkv(q, k, v)
    batch, heads, seq, dim = q.shape
    scale = _resolve_scale(scale, dim)
    if causal:
        _require_same_query_key_length(q, k)

    compute_dtype = torch.float32
    q32 = q.to(compute_dtype)
    k32 = k.to(compute_dtype)
    v32 = v.to(compute_dtype)

    scores = torch.matmul(q32, k32.transpose(-2, -1)) * scale
    if causal:
        mask = _causal_mask(seq, seq, q.device)
        scores = scores.masked_fill(~mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    output = torch.matmul(probs, v32)

    if return_attention_weights:
        return output.to(q.dtype), probs
    return output.to(q.dtype)


def _require_same_query_key_length(q: torch.Tensor, k: torch.Tensor) -> None:
    if q.shape[2] != k.shape[2]:
        raise ValueError(
            "causal attention assumed self-attention; query and key sequence "
            f"lengths differ ({q.shape[2]} vs {k.shape[2]})"
        )


def tiled_online_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = True,
    scale: float | None = None,
    q_tile_size: int = 128,
    k_tile_size: int = 128,
) -> torch.Tensor:
    """Blockwise attention with the online softmax recurrence; no ``S x S`` tensor.

    Outer loop over query tiles, inner loop over key/value tiles that can still
    contribute (for causal attention, only tiles whose key range intersects each
    query range). Per query tile the state ``m`` (running max), ``l`` (running
    softmax denominator) and ``acc`` (running value accumulator) are updated
    with the online softmax recurrence; the final ``acc / l`` matches the
    reference output up to floating-point reassociation. Peak memory is
    ``O(S * Dh + T_q * T_k)``, never ``O(S^2)``.

    Computed in fp32 internally and returned in the input dtype. Runs anywhere
    PyTorch runs (pure tensor ops, no custom kernels); its cost is dominated by
    per-tile eager kernel dispatch, which is the reason educational tiled
    implementations do not match fused-kernel speed (see the Task 4 report).
    """
    _validate_qkv(q, k, v)
    if q_tile_size <= 0 or k_tile_size <= 0:
        raise ValueError("tile sizes must be positive")
    scale = _resolve_scale(scale, q.shape[-1])
    if causal:
        _require_same_query_key_length(q, k)
    batch, heads, seq, dim = q.shape

    q32 = q.to(torch.float32)
    k32 = k.to(torch.float32)
    v32 = v.to(torch.float32)

    output = torch.empty((batch, heads, seq, dim), dtype=torch.float32, device=q.device)
    kv_limit = seq if causal else k.shape[2]

    for q_start in range(0, seq, int(q_tile_size)):
        q_end = min(q_start + q_tile_size, seq)
        q_tile = q32[:, :, q_start:q_end, :]
        tile_rows = q_end - q_start

        running_max = torch.full(
            (batch, heads, tile_rows),
            float("-inf"),
            dtype=torch.float32,
            device=q.device,
        )
        running_norm = torch.zeros((batch, heads, tile_rows), dtype=torch.float32, device=q.device)
        accumulator = torch.zeros((batch, heads, tile_rows, dim), dtype=torch.float32, device=q.device)

        for kv_start in range(0, kv_limit, int(k_tile_size)):
            # Causal tiles entirely above the diagonal contribute exactly zero
            # to every row state, so skipping them is lossless, not a shortcut.
            if causal and kv_start >= q_end:
                break
            kv_end = min(kv_start + k_tile_size, seq)
            k_tile = k32[:, :, kv_start:kv_end, :]
            v_tile = v32[:, :, kv_start:kv_end, :]

            scores = torch.matmul(q_tile, k_tile.transpose(-2, -1)) * scale  # (B, H, Tq, Tk)
            if causal:
                # Global (query, key) indices: tile-local masks would be wrong
                # for any kv tile that starts after the query tile.
                query_indices = torch.arange(q_start, q_end, device=q.device).unsqueeze(-1)
                key_indices = torch.arange(kv_start, kv_end, device=q.device).unsqueeze(0)
                mask = query_indices >= key_indices  # (Tq, Tk)
                scores = scores.masked_fill(~mask, float("-inf"))

            tile_max = scores.amax(dim=-1)  # (B, H, Tq); every row has >= 1 valid entry
            new_max = torch.maximum(running_max, tile_max)
            probs = torch.exp(scores - new_max.unsqueeze(-1))  # masked entries are exp(-inf) == 0
            rescale = torch.exp(running_max - new_max)  # first tile: exp(-inf) == 0

            running_norm = running_norm * rescale + probs.sum(dim=-1)
            accumulator = accumulator * rescale.unsqueeze(-1) + torch.matmul(probs, v_tile)
            running_max = new_max

        output[:, :, q_start:q_end, :] = accumulator / running_norm.unsqueeze(-1)

    return output.to(q.dtype)


def agreement_stats(output: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    """Max abs / max rel / mean abs differences between two attention outputs.

    Relative error uses ``|a - b| / max(|b|, eps)`` with ``eps = 1e-12``; values
    agree only where both tensors are finite.
    """
    a = output.to(torch.float32)
    b = reference.to(torch.float32)
    both_finite = torch.isfinite(a) & torch.isfinite(b)
    diff = (a - b).abs()
    if both_finite.any():
        abs_vals = diff[both_finite]
        rel_vals = (diff / b.abs().clamp_min(1e-12))[both_finite]
        max_abs = float(abs_vals.max())
        max_rel = float(rel_vals.max())
        mean_abs = float(abs_vals.mean())
    else:
        max_abs = max_rel = mean_abs = float("nan")
    return {
        "max_abs_diff": max_abs,
        "max_rel_diff": max_rel,
        "mean_abs_diff": mean_abs,
        "nan_count": int(torch.isnan(a).sum() + torch.isnan(b).sum()),
        "inf_count": int(torch.isinf(a).sum() + torch.isinf(b).sum()),
    }
