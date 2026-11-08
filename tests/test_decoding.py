"""Tests for the Task 3 decoding paths (cached vs. no-cache recompute)."""

from __future__ import annotations

import unittest

import torch
from transformers import Gemma3ForCausalLM, Gemma3TextConfig

from trailo.eval.decoding import (
    greedy_generate_cached,
    greedy_generate_recompute,
    positions_from_mask,
)


def build_tiny_gemma3(seed: int = 0, num_layers: int = 3, sliding_window: int = 4):
    """Random-weight tiny Gemma3, mixing sliding and full attention layers."""
    config = Gemma3TextConfig(
        vocab_size=256,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=num_layers,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        query_pre_attn_scalar=16,
        sliding_window=sliding_window,
        layer_types=(
            ["sliding_attention", "full_attention"] * ((num_layers + 1) // 2)
        )[:num_layers],
        max_position_embeddings=256,
        hidden_activation="gelu_pytorch_tanh",
        rope_local_base_freq=10000.0,
        rope_theta=1000000.0,
        bos_token_id=2,
        eos_token_id=1,
        pad_token_id=0,
    )
    torch.manual_seed(seed)
    model = Gemma3ForCausalLM(config)
    model.eval()
    model.config.use_cache = True
    return model


def synthetic_prompt(batch: int = 1, length: int = 12) -> torch.Tensor:
    torch.manual_seed(42)
    # Avoid ids 0/1 (pad/eos) so prompts are free of stop tokens.
    return (2 + (torch.arange(length) * 7) % 253).unsqueeze(0)


class TestPositionsFromMask(unittest.TestCase):
    def test_unpadded_positions_are_arange(self):
        mask = torch.ones(1, 5)
        self.assertTrue(
            torch.equal(positions_from_mask(mask), torch.arange(5).unsqueeze(0))
        )

    def test_pinned_pad_positions(self):
        mask = torch.tensor([[1, 1, 0, 1]])
        positions = positions_from_mask(mask)
        self.assertEqual(positions.tolist(), [[0, 1, 1, 2]])
        self.assertEqual(positions.dtype, torch.long)


class TestCachedVsRecomputeAgreement(unittest.TestCase):
    """The cached path must produce the same tokens as the full recompute path."""

    def _run_both(self, model, prompt: torch.Tensor, max_new_tokens: int, seed: int = 0):
        result_cached = greedy_generate_cached(
            model,
            prompt,
            None,
            max_new_tokens,
            eos_token_ids=[],
            record_margins=True,
            track_memory=False,
        )
        result_recompute = greedy_generate_recompute(
            model,
            prompt,
            None,
            max_new_tokens,
            eos_token_ids=[],
            record_margins=True,
            track_memory=False,
        )
        return result_cached, result_recompute

    def test_token_agreement_within_window(self):
        model = build_tiny_gemma3()
        prompt = synthetic_prompt(length=12)
        cached, recompute = self._run_both(model, prompt, max_new_tokens=16)
        self.assertEqual(cached.generated_ids, recompute.generated_ids)
        self.assertEqual(len(cached.generated_ids), 16)

    def test_token_agreement_across_sliding_window_boundary(self):
        # Total sequence length passes sliding_window=4 several times, so the
        # incremental path exercises windowed masking over a growing cache.
        model = build_tiny_gemma3()
        prompt = synthetic_prompt(length=12)
        cached, recompute = self._run_both(model, prompt, max_new_tokens=24)
        self.assertEqual(cached.generated_ids, recompute.generated_ids)

    def test_argmax_margins_are_decisive(self):
        model = build_tiny_gemma3()
        prompt = synthetic_prompt(length=12)
        cached, _ = self._run_both(model, prompt, max_new_tokens=8)
        assert cached.logprob_margins is not None
        self.assertTrue(
            all(margin > 1e-6 for margin in cached.logprob_margins),
            f"unexpected near-tie margins: {cached.logprob_margins}",
        )


class TestEosStopping(unittest.TestCase):
    def test_both_paths_stop_on_eos_and_append_it(self):
        model = build_tiny_gemma3(seed=3)
        prompt = synthetic_prompt(length=12)
        # Run once to find the natural greedy stream, then inject its first
        # token id as an eos id: both paths must stop after one token.
        probe = greedy_generate_cached(
            model, prompt, None, 8, eos_token_ids=[], track_memory=False
        )
        inner_eos = {probe.generated_ids[3]}
        cached = greedy_generate_cached(
            model, prompt, None, 8, eos_token_ids=inner_eos, track_memory=False
        )
        recompute = greedy_generate_recompute(
            model, prompt, None, 8, eos_token_ids=inner_eos, track_memory=False
        )
        self.assertEqual(len(cached.generated_ids), 4)
        self.assertEqual(len(recompute.generated_ids), 4)
        self.assertEqual(cached.generated_ids, recompute.generated_ids)
        self.assertIn(inner_eos.pop(), cached.generated_ids)

    def test_results_memory_fields_present(self):
        model = build_tiny_gemma3()
        prompt = synthetic_prompt(length=12)
        result = greedy_generate_cached(
            model, prompt, None, 4, eos_token_ids=[], track_memory=True
        )
        self.assertGreater(result.peak_memory_gb, 0.0)
        self.assertGreater(result.cache_logical_bytes, 0.0)


class TestGenerationCrossCheck(unittest.TestCase):
    def test_cached_path_matches_model_generate(self):
        model = build_tiny_gemma3(seed=1)
        prompt = synthetic_prompt(length=12)
        from trailo.eval.decoding import generation_eos_token_ids

        class _TokenizerStub:
            eos_token_id = 1

            @staticmethod
            def convert_tokens_to_ids(token: str):
                return -2  # no <end_of_turn> in this toy model

        result = greedy_generate_cached(
            model,
            prompt,
            None,
            12,
            eos_token_ids=generation_eos_token_ids(model, _TokenizerStub()),
            track_memory=False,
        )
        with torch.inference_mode():
            generated = model.generate(
                input_ids=prompt,
                attention_mask=torch.ones_like(prompt),
                max_new_tokens=12,
                do_sample=False,
                num_beams=1,
                use_cache=True,
            )[0, prompt.shape[1] :]
        self.assertEqual(result.generated_ids, generated.tolist())


if __name__ == "__main__":
    unittest.main()
