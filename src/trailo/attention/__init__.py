"""Task 4 attention implementations (reference and tiled online softmax)."""

from trailo.attention.online_attention import (
    agreement_stats,
    reference_attention,
    tiled_online_attention,
)

__all__ = ["agreement_stats", "reference_attention", "tiled_online_attention"]
