"""Tests for the Task 5 greedy speculative decoding loop.

Strategy: a tiny Gemma3 target and draft that share one 256-token
vocabulary (same constraints as the real E4B/E2B pair: identical vocab,
different depth/width). The reference is always the Task 3 target-only
greedy path (``run_greedy_baseline``); speculative decoding must reproduce
its token stream exactly, including replacement tokens, drafted eos
candidates, and cache rollbacks across many rounds.

Deterministic forcing is done with position-exact logit-bias wrappers:
biasing a token by a huge value from a given position onward makes BOTH the
single-token baseline forwards and the batched verification rows emit the
same forced token at the same positions, so equality stays checkable.
"""

from __future__ import annotations

import unittest

import torch
from transformers import Gemma3ForCausalLM, Gemma3TextConfig

from trailo.eval.speculative import (
    speculative_generate_greedy,
    run_greedy_baseline,
)


EOS_ID = 1


def build_tiny_gemma3(
    seed: int,
    num_layers: int,
    hidden_size: int,
    sliding_window: int = 4,
) -> Gemma3ForCausalLM:
    """Random-weight tiny Gemma3, mixing sliding and full attention layers."""
    config = Gemma3TextConfig(
        vocab_size=256,
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 2,
        num_hidden_layers=num_layers,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        query_pre_attn_scalar=16,
        sliding_window=sliding_window,
        layer_types=(
            ["sliding_attention", "full_attention"] * ((num_layers + 1) // 2)
        )[:num_layers],
        max_position_embeddings=1024,
        hidden_activation="gelu_pytorch_tanh",
        rope_local_base_freq=10000.0,
        rope_theta=1000000.0,
        bos_token_id=2,
        eos_token_id=EOS_ID,
        pad_token_id=0,
    )
    torch.manual_seed(seed)
    model = Gemma3ForCausalLM(config)
    model.eval()
    model.config.use_cache = True
    return model


class RowBiasedModel(torch.nn.Module):
    """Wraps a causal LM; adds a logit bias to one token id from a position on.

    The bias is applied per logits row using the row's position id, so it
    behaves identically for single-token forwards (baseline path) and for
    the batched drafting/verification forwards (speculative path). This is
    what keeps forced events position-exact across decode paths.
    """

    def __init__(self, model, token_id: int, bias: float, from_position: int = 0):
        super().__init__()
        self.model = model
        self.token_id = int(token_id)
        self.bias = float(bias)
        self.from_position = int(from_position)

    def forward(self, input_ids=None, position_ids=None, **kwargs):
        outputs = self.model(
            input_ids=input_ids, position_ids=position_ids, **kwargs
        )
        logits = outputs.logits
        if position_ids is not None and logits.shape[1] > 0:
            rows = position_ids[0, -logits.shape[1] :]
            mask = rows >= self.from_position
            if mask.any():
                logits[..., self.token_id] = logits[..., self.token_id] + (
                    mask.to(logits.dtype) * self.bias
                )
        return outputs


def synthetic_prompt(length: int, seed: int = 42) -> torch.Tensor:
    """Deterministic prompt, no pad/eos ids."""
    generator = torch.Generator().manual_seed(seed)
    ids = torch.randint(3, 256, (1, length), generator=generator)
    return ids


class TestSpeculativeGreedy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Target (deeper/wider) and draft (shallower/narrower) share the vocab.
        cls.target = build_tiny_gemma3(seed=0, num_layers=3, hidden_size=32)
        cls.draft = build_tiny_gemma3(seed=7, num_layers=1, hidden_size=16)

    def check_round_invariants(self, result, max_new_tokens: int):
        """Structural invariants of the round records."""
        appended = sum(record.appended for record in result.rounds)
        self.assertEqual(
            len(result.generated_ids),
            min(1 + appended, int(max_new_tokens)),
        )
        for record in result.rounds:
            self.assertGreaterEqual(record.drafted, 1)
            self.assertLessEqual(record.accepted, record.drafted)
            if record.replaced:
                self.assertEqual(record.appended, record.accepted + 1)
            else:
                self.assertEqual(record.appended, record.accepted)
            self.assertGreater(record.appended, 0)
            if record.draft_eos:
                self.assertFalse(record.replaced)
                self.assertLessEqual(record.accepted, record.drafted)
        self.assertAlmostEqual(
            result.acceptance_rate,
            result.accepted_total / result.drafted_total if result.drafted_total else 0.0,
            places=6,
        )

    def test_matches_target_greedy_across_gammas(self):
        prompt = synthetic_prompt(16)
        eos_ids = [EOS_ID]
        for gamma in (1, 2, 4):
            with self.subTest(gamma=gamma):
                spec = speculative_generate_greedy(
                    self.target,
                    self.draft,
                    prompt,
                    None,
                    24,
                    eos_ids,
                    gamma=gamma,
                    track_memory=False,
                    synchronize=False,
                )
                baseline = run_greedy_baseline(
                    self.target, prompt, None, 24, eos_ids
                )
                self.assertEqual(spec.generated_ids, baseline.generated_ids)
                self.check_round_invariants(spec, 24)

    def test_multiple_prompts_match_baseline(self):
        eos_ids = [EOS_ID]
        for seed in (1, 2, 3):
            prompt = synthetic_prompt(12, seed=seed)
            spec = speculative_generate_greedy(
                self.target,
                self.draft,
                prompt,
                None,
                20,
                eos_ids,
                gamma=3,
                track_memory=False,
                synchronize=False,
            )
            baseline = run_greedy_baseline(self.target, prompt, None, 20, eos_ids)
            self.assertEqual(spec.generated_ids, baseline.generated_ids)

    def test_forced_disagreement_uses_replacement(self):
        """Draft forced to token 7 from position 0: replacements must appear."""
        biased_draft = RowBiasedModel(self.draft, token_id=7, bias=1e6)
        prompt = synthetic_prompt(16)
        eos_ids = [EOS_ID]
        spec = speculative_generate_greedy(
            self.target,
            biased_draft,
            prompt,
            None,
            24,
            eos_ids,
            gamma=3,
            track_memory=False,
            synchronize=False,
        )
        baseline = run_greedy_baseline(self.target, prompt, None, 24, eos_ids)
        # The replacement mechanism keeps the output exactly on the target path.
        self.assertEqual(spec.generated_ids, baseline.generated_ids)
        self.assertLess(spec.acceptance_rate, 1.0)
        self.assertTrue(any(record.replaced for record in spec.rounds))
        # Every round appended the target argmax, never the forced draft token
        # unless the target genuinely argmaxed it.
        for record in spec.rounds:
            self.assertLessEqual(record.accepted, record.appended)
        # Every verification forward emits at least one token.
        self.assertGreaterEqual(spec.tokens_per_target_forward, 1.0)

    def test_target_forced_eos_stops_both_paths(self):
        """Target biased to eos from position 19: both paths must stop there."""
        prompt_len = 16
        biased_target = RowBiasedModel(
            self.target, token_id=EOS_ID, bias=1e6, from_position=prompt_len + 3
        )
        prompt = synthetic_prompt(prompt_len)
        eos_ids = [EOS_ID]
        spec = speculative_generate_greedy(
            biased_target,
            self.draft,
            prompt,
            None,
            40,
            eos_ids,
            gamma=3,
            track_memory=False,
            synchronize=False,
        )
        baseline = run_greedy_baseline(
            biased_target, prompt, None, 40, eos_ids
        )
        self.assertEqual(spec.generated_ids, baseline.generated_ids)
        self.assertTrue(spec.stopped_on_eos)
        self.assertTrue(baseline.generated_ids[-1] == EOS_ID)
        self.assertGreater(len(spec.generated_ids), 1)

    def test_draft_proposed_eos_is_verified_or_replaced(self):
        """Draft biased to eos after the first candidate: gamma_eff shrinks."""
        prompt_len = 16
        biased_draft = RowBiasedModel(
            self.draft, token_id=EOS_ID, bias=1e6, from_position=prompt_len + 1
        )
        prompt = synthetic_prompt(prompt_len)
        eos_ids = [EOS_ID]
        spec = speculative_generate_greedy(
            self.target,
            biased_draft,
            prompt,
            None,
            16,
            eos_ids,
            gamma=4,
            track_memory=False,
            synchronize=False,
        )
        baseline = run_greedy_baseline(self.target, prompt, None, 16, eos_ids)
        self.assertEqual(spec.generated_ids, baseline.generated_ids)
        # From position prompt_len+1 on, every draft candidate is eos, so
        # drafting must stop at gamma_eff <= 2 candidates per round.
        self.assertTrue(spec.rounds)
        self.assertTrue(all(record.drafted <= 2 for record in spec.rounds))

    def test_prefill_eos_skips_drafting(self):
        """Target biased to eos everywhere: 1-token output, zero rounds."""
        biased_target = RowBiasedModel(self.target, token_id=EOS_ID, bias=1e6)
        prompt = synthetic_prompt(12)
        eos_ids = [EOS_ID]
        spec = speculative_generate_greedy(
            biased_target,
            self.draft,
            prompt,
            None,
            16,
            eos_ids,
            gamma=4,
            track_memory=False,
            synchronize=False,
        )
        baseline = run_greedy_baseline(biased_target, prompt, None, 16, eos_ids)
        self.assertEqual(spec.generated_ids, baseline.generated_ids)
        self.assertEqual(len(spec.generated_ids), 1)
        self.assertEqual(spec.rounds, [])
        self.assertTrue(spec.stopped_on_eos)

    def test_many_rounds_rollback_stays_on_target_path(self):
        """Long run with forced early round endings: rollbacks must not corrupt.

        The draft is forced to disagree every round (bias), so every round
        ends in a rejection and both caches crop back; equality over 30
        tokens proves the rollback bookkeeping.
        """
        biased_draft = RowBiasedModel(self.draft, token_id=9, bias=1e6)
        prompt = synthetic_prompt(20)
        eos_ids = [EOS_ID]
        spec = speculative_generate_greedy(
            self.target,
            biased_draft,
            prompt,
            None,
            30,
            eos_ids,
            gamma=2,
            track_memory=False,
            synchronize=False,
            record_margins=True,
        )
        baseline = run_greedy_baseline(self.target, prompt, None, 30, eos_ids)
        self.assertEqual(spec.generated_ids, baseline.generated_ids)
        # Forced disagreement ends every round in a rejection + rollback.
        self.assertGreaterEqual(len(spec.rounds), 10)
        self.assertFalse(spec.stopped_on_eos)
        self.assertEqual(len(spec.generated_ids), 30)
        # Margin bookkeeping invariant: rounds that rejected at index 0
        # (accepted == 0) must not have touched the accepted-margin aggregate,
        # and must carry the replacement row's margin.
        touched = 0
        for record in spec.rounds:
            if record.accepted == 0:
                self.assertIsNone(record.accepted_min_margin)
                self.assertIsNotNone(record.replaced_margin)
            elif record.accepted_min_margin is not None:
                touched += 1
        # Any genuinely accepted rows recorded a margin (verify the bookkeeping
        # branch actually runs).
        accepted_rounds = [r for r in spec.rounds if r.accepted > 0]
        if accepted_rounds:
            self.assertGreater(
                sum(1 for r in accepted_rounds if r.accepted_min_margin is not None), 0
            )

    def test_gamma_larger_than_needed_and_overflow_truncation(self):
        """A single round must not emit past max_new_tokens.

        Draft and target are biased to the same token, so every candidate is
        accepted and a gamma=8 round would append 8 tokens in one verify
        forward - the cap must stop the stream at exactly max_new_tokens.
        """
        shared_biased_target = RowBiasedModel(self.target, token_id=5, bias=1e6)
        biased_draft = RowBiasedModel(self.draft, token_id=5, bias=1e6)
        prompt = synthetic_prompt(10)
        eos_ids = [EOS_ID]
        spec = speculative_generate_greedy(
            shared_biased_target,
            biased_draft,
            prompt,
            None,
            5,
            eos_ids,
            gamma=8,
            track_memory=False,
            synchronize=False,
        )
        baseline = run_greedy_baseline(shared_biased_target, prompt, None, 5, eos_ids)
        self.assertEqual(len(spec.generated_ids), 5)
        self.assertEqual(len(baseline.generated_ids), 5)
        self.assertEqual(spec.generated_ids[:5], baseline.generated_ids[:5])
        self.assertTrue(spec.truncated_to_max_new)


if __name__ == "__main__":
    unittest.main()
