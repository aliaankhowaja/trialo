"""Tests for Task 4 attention (tiled online softmax vs naive reference)."""

from __future__ import annotations

import unittest

import torch
from torch.nn.functional import scaled_dot_product_attention

from trailo.attention import (
    agreement_stats,
    reference_attention,
    tiled_online_attention,
)


def random_qkv(
    batch: int,
    heads: int,
    seq: int,
    dim: int,
    *,
    dtype: torch.dtype = torch.float32,
    seed: int = 0,
    magnitude: float = 1.0,
    device: str = "cpu",
):
    """Deterministic random q/k/v of shape (B, H, S, Dh) in the given dtype."""
    generator = torch.Generator().manual_seed(seed)
    shape = (batch, heads, seq, dim)
    q = torch.randn(shape, generator=generator, dtype=torch.float32) * magnitude
    k = torch.randn(shape, generator=generator, dtype=torch.float32) * magnitude
    v = torch.randn(shape, generator=generator, dtype=torch.float32) * magnitude
    q, k, v = (t.to(device) for t in (q, k, v))
    return q.to(dtype).contiguous(), k.to(dtype).contiguous(), v.to(dtype).contiguous()


class TestReferenceVsTiled(unittest.TestCase):
    """fp32 agreement sweep over shapes and tile configurations."""

    def test_causal_agreement_sweep(self):
        cases = [
            (1, 2, 1, 8),
            (2, 3, 2, 8),
            (1, 2, 3, 8),
            (2, 2, 7, 8),
            (1, 3, 17, 8),
            (2, 2, 17, 64),
            (1, 2, 32, 16),
        ]
        tile_grid = [(3, 5), (7, 17), (128, 128)]
        for index, (batch, heads, seq, dim) in enumerate(cases):
            q, k, v = random_qkv(batch, heads, seq, dim, seed=100 + index)
            reference = reference_attention(q, k, v, causal=True)
            for q_tile, k_tile in tile_grid:
                tiled = tiled_online_attention(
                    q, k, v, causal=True, q_tile_size=q_tile, k_tile_size=k_tile
                )
                self.assertTrue(
                    torch.allclose(tiled, reference, rtol=1e-5, atol=1e-5),
                    msg=f"causal mismatch {index}: stats={agreement_stats(tiled, reference)}",
                )

    def test_non_causal_agreement_sweep(self):
        cases = [(1, 2, 5, 8), (2, 2, 16, 16), (1, 2, 33, 8)]
        for index, (batch, heads, seq, dim) in enumerate(cases):
            q, k, v = random_qkv(batch, heads, seq, dim, seed=200 + index)
            reference = reference_attention(q, k, v, causal=False)
            tiled = tiled_online_attention(
                q, k, v, causal=False, q_tile_size=4, k_tile_size=6
            )
            self.assertTrue(
                torch.allclose(tiled, reference, rtol=1e-5, atol=1e-5),
                msg=f"non-causal mismatch {index}: stats={agreement_stats(tiled, reference)}",
            )

    def test_explicit_scale_is_honoured(self):
        q, k, v = random_qkv(1, 2, 13, 16, seed=300)
        scale = 0.345
        for causal in (True, False):
            reference = reference_attention(q, k, v, causal=causal, scale=scale)
            tiled = tiled_online_attention(q, k, v, causal=causal, scale=scale)
            self.assertTrue(torch.allclose(tiled, reference, rtol=1e-5, atol=1e-5))
        default = reference_attention(q, k, v, causal=True)
        custom = reference_attention(q, k, v, causal=True, scale=scale)
        self.assertGreater((custom - default).abs().max(), 1e-4)

    def test_non_contiguous_inputs(self):
        generator = torch.Generator().manual_seed(400)
        q0 = torch.randn((1, 11, 2, 8), generator=generator, dtype=torch.float32)
        k0 = torch.randn((1, 11, 2, 8), generator=generator, dtype=torch.float32)
        v0 = torch.randn((1, 11, 2, 8), generator=generator, dtype=torch.float32)
        # (B, S, H, Dh) storage viewed as (B, H, S, Dh): valid strides, not contiguous.
        q_t, k_t, v_t = (t.permute(0, 2, 1, 3) for t in (q0, k0, v0))
        q, k, v = (t.contiguous() for t in (q_t, k_t, v_t))
        self.assertFalse(q_t.is_contiguous())

        reference = reference_attention(q, k, v)
        tiled_views = tiled_online_attention(q_t, k_t, v_t)
        self.assertTrue(
            torch.allclose(tiled_views, reference, rtol=1e-5, atol=1e-5),
            msg=f"stats={agreement_stats(tiled_views, reference)}",
        )

    def test_tiles_larger_than_sequence(self):
        q, k, v = random_qkv(1, 2, 9, 8, seed=410)
        reference = reference_attention(q, k, v)
        tiled = tiled_online_attention(q, k, v, q_tile_size=64, k_tile_size=64)
        self.assertTrue(torch.allclose(tiled, reference, rtol=1e-5, atol=1e-5))

    def test_invalid_tile_sizes_raise(self):
        q, k, v = random_qkv(1, 2, 4, 8, seed=420)
        with self.assertRaises(ValueError):
            tiled_online_attention(q, k, v, q_tile_size=0)
        with self.assertRaises(ValueError):
            tiled_online_attention(q, k, v, k_tile_size=-3)

    def test_causal_cross_attention_rejected(self):
        q, k, v = random_qkv(1, 2, 4, 8, seed=430)
        with self.assertRaises(ValueError):
            reference_attention(q, k, torch.cat([v, v], dim=2), causal=True)
        with self.assertRaises(ValueError):
            tiled_online_attention(q, k, torch.cat([v, v], dim=2), causal=True)

    def test_output_has_no_nan_or_inf(self):
        q, k, v = random_qkv(2, 2, 19, 16, seed=440)
        tiled = tiled_online_attention(q, k, v, q_tile_size=7, k_tile_size=11)
        self.assertFalse(torch.isnan(tiled).any())
        self.assertFalse(torch.isinf(tiled).any())


class TestAgainstIndependentlyDerivedValues(unittest.TestCase):
    """Cross-check against hand-rolled row math and torch's SDPA."""

    def test_diagonal_boundary_rows(self):
        batch, heads, seq, dim = 1, 2, 7, 8
        q, k, v = random_qkv(batch, heads, seq, dim, seed=500)
        tiled = tiled_online_attention(q, k, v, q_tile_size=3, k_tile_size=5)

        # Row 0 attends only to itself -> softmax of a single entry is 1.
        self.assertTrue(
            torch.allclose(tiled[:, :, 0], v[:, :, 0].to(tiled.dtype), rtol=1e-4, atol=1e-4)
        )

        # Full manual prefix-softmax reference, row by row.
        scores_full = torch.einsum("bhsd,bhtd->bhst", q.float(), k.float()) * (dim**-0.5)
        manual = torch.empty_like(q.float())
        for row in range(seq):
            weights = torch.softmax(scores_full[:, :, row, : row + 1], dim=-1)
            values = v[:, :, : row + 1, :].float()
            manual[:, :, row, :] = torch.einsum("bht,bhtd->bhd", weights, values)
        self.assertTrue(
            torch.allclose(tiled, manual.to(tiled.dtype), rtol=1e-5, atol=1e-5),
            msg=f"stats={agreement_stats(tiled, manual)}",
        )

    def test_matches_torch_sdpa(self):
        for causal in (True, False):
            for seq, dim in [(1, 8), (7, 16), (23, 64)]:
                q, k, v = random_qkv(2, 2, seq, dim, seed=600 + seq)
                expected = scaled_dot_product_attention(q, k, v, is_causal=causal)
                tiled = tiled_online_attention(q, k, v, causal=causal)
                self.assertTrue(
                    torch.allclose(tiled, expected, rtol=1e-5, atol=1e-5),
                    msg=f"sdpa mismatch causal={causal} seq={seq} dim={dim}: "
                    f"{agreement_stats(tiled, expected)}",
                )

    def test_agreement_stats_reports_differences(self):
        base = torch.ones(2, 3)
        stats = agreement_stats(torch.ones(2, 3), base)
        self.assertEqual(stats["max_abs_diff"], 0.0)
        self.assertEqual(stats["nan_count"], 0)
        drifted = base.clone()
        drifted[0, 0] = 1.5
        drift_stats = agreement_stats(drifted, base)
        self.assertAlmostEqual(drift_stats["max_abs_diff"], 0.5, places=6)
        self.assertAlmostEqual(drift_stats["max_rel_diff"], 0.5, places=6)


class TestNumericalStability(unittest.TestCase):
    """Large-magnitude logits that overflow a naive fp16 softmax path."""

    def test_scaled_inputs_stay_finite_and_match_reference(self):
        for magnitude, scale in [(1e3, 64.0), (1e2, 128.0)]:
            q, k, v = random_qkv(1, 2, 24, 8, seed=700, magnitude=magnitude)
            tiled = tiled_online_attention(q, k, v, scale=scale)
            reference = reference_attention(q, k, v, scale=scale)
            self.assertFalse(
                torch.isnan(tiled).any(), msg=f"nan with magnitude={magnitude} scale={scale}"
            )
            self.assertFalse(
                torch.isinf(tiled).any(), msg=f"inf with magnitude={magnitude} scale={scale}"
            )
            self.assertTrue(
                torch.allclose(tiled, reference, rtol=1e-4, atol=1e-4),
                msg=f"stats={agreement_stats(tiled, reference)}",
            )

    def test_naive_fp16_softmax_overflows_but_tiled_does_not(self):
        q, k, v = random_qkv(1, 1, 16, 8, seed=800)
        q16, k16, v16 = q.to(torch.float16), k.to(torch.float16), v.to(torch.float16)
        # Deliberately naive fp16 softmax path: exp on raw scores with no
        # max subtraction and no fp32 upcast, exactly the failure mode online
        # softmax exists to prevent. Scaled scores far exceed fp16's exp range.
        scores16 = torch.matmul(q16, k16.transpose(-2, -1)) * 64.0
        probs16 = torch.exp(scores16) / torch.exp(scores16).sum(dim=-1, keepdim=True)
        naive = torch.matmul(probs16, v16)
        self.assertTrue(
            torch.isnan(naive).any() or torch.isinf(naive).any(),
            msg="expected the deliberately naive fp16 path to overflow",
        )
        tiled = tiled_online_attention(q16, k16, v16, scale=64.0)
        reference = reference_attention(q.float(), k.float(), v.float(), scale=64.0)
        self.assertFalse(torch.isnan(tiled).any())
        self.assertFalse(torch.isinf(tiled).any())
        self.assertTrue(
            torch.allclose(tiled.float(), reference, rtol=1e-2, atol=1e-2),
            msg=f"stats={agreement_stats(tiled, reference)}",
        )


class TestMemorySavings(unittest.TestCase):
    """The tiled path must not allocate the S^2 intermediate."""

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA memory profiling requires GPU")
    def test_peak_memory_scales_like_S_not_S2(self):
        batch, heads, seq, dim = 1, 8, 1024, 64
        q, k, v = random_qkv(batch, heads, seq, dim, seed=900, device="cuda")

        def peak_of(fn) -> float:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            fn()
            torch.cuda.synchronize()
            return torch.cuda.max_memory_allocated() - torch.cuda.memory_allocated()

        naive_peak = peak_of(lambda: reference_attention(q, k, v, causal=False))
        tiled_peak = peak_of(lambda: tiled_online_attention(q, k, v, causal=False))
        self.assertGreater(
            naive_peak,
            tiled_peak * 5,
            msg=(
                f"expected naive peak {naive_peak / 1e6:.1f}MB to be far above "
                f"tiled peak {tiled_peak / 1e6:.1f}MB"
            ),
        )


if __name__ == "__main__":
    unittest.main()
