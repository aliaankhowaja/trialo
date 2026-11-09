"""Application settings, resolved from environment variables.

The served Transformer is fully configurable:

- ``TRIALO_MODEL_CONFIG`` selects a named preset (``smoke_cpu`` or
  ``final_a100``); the preset fixes base model id, LoRA adapter directory and
  precision.
- ``TRIALO_BASE_MODEL_ID`` / ``TRIALO_CHECKPOINT_DIR`` override the preset so a
  freshly fine-tuned adapter can be served without code changes.

``TRIALO_REPO_ROOT`` points at the mounted repository so the backend can import
``trailo`` (from ``<root>/src``) and read the recorded GPU speculative-decoding
benchmark artifacts (from ``<root>/src/trailo/results``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

DISCLAIMER = (
    "Educational screening prototype. This tool annotates eligibility criteria "
    "with a fine-tuned language model for teaching and demonstration purposes "
    "only. It is NOT a clinical decision system and must never be used for "
    "actual patient enrollment, clinical care, or regulatory decisions."
)

_LABEL_SETS = {
    "inclusion": ["included", "not applicable", "not enough information", "not included"],
    "exclusion": ["excluded", "not applicable", "not enough information", "not excluded"],
}

_MODEL_PRESETS: dict[str, dict] = {
    "final_prod": {
        "label": "final_prod: google/gemma-3-270m-it + full 4-epoch LoRA finetune (final_prod_seed42)",
        "base_model_id": "google/gemma-3-270m-it",
        "checkpoint_dir": "src/trailo/checkpoints/final_prod_seed42/best",
        "dtype": "bfloat16",  # downgraded to fp32 automatically on CPU devices
        "max_length": 1024,
    },
    "smoke_cpu": {
        "label": "CPU smoke: google/gemma-3-270m-it + 3-step smoke adapter (smoke_cpu_seed42)",
        "base_model_id": "google/gemma-3-270m-it",
        "checkpoint_dir": "src/trailo/checkpoints/smoke_cpu_seed42/best",
        "dtype": "float32",
        "max_length": 1024,
    },
    "final_a100": {
        "label": "GPU: google/gemma-3n-E4B-it + LoRA adapter (final_a100_seed42)",
        "base_model_id": "google/gemma-3n-E4B-it",
        "checkpoint_dir": "src/trailo/checkpoints/final_a100_seed42/best",
        "dtype": "bfloat16",
        "max_length": 1024,
    },
}


@dataclass(frozen=True)
class Settings:
    model_config_name: str
    base_model_id: str
    checkpoint_dir: str
    dtype_name: str
    max_length: int
    device_name: str  # "cpu" | "cuda"
    max_new_tokens: int
    repo_root: Path
    frontend_origin: str
    extras: dict = field(default_factory=dict)

    @property
    def repo_src_path(self) -> Path:
        return self.repo_root / "src"

    @property
    def results_dir(self) -> Path:
        return self.repo_root / "src" / "trailo" / "results"

    @property
    def label_sets(self) -> dict[str, list[str]]:
        return {k: list(v) for k, v in _LABEL_SETS.items()}


def _resolve_device() -> str:
    requested = os.environ.get("TRIALO_DEVICE", "auto").strip().lower()
    if requested in {"cpu", "cuda"}:
        return requested
    try:  # pragma: no cover - depends on host hardware
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def load_settings() -> Settings:
    repo_root = Path(os.environ.get("TRIALO_REPO_ROOT", "/workspace")).resolve()
    config_name = os.environ.get("TRIALO_MODEL_CONFIG", "final_prod").strip().lower()
    if config_name not in _MODEL_PRESETS:
        raise RuntimeError(
            f"unknown TRIALO_MODEL_CONFIG {config_name!r}; "
            f"expected one of {sorted(_MODEL_PRESETS)}"
        )
    preset = _MODEL_PRESETS[config_name]

    base_model_id = os.environ.get("TRIALO_BASE_MODEL_ID") or preset["base_model_id"]
    checkpoint = os.environ.get("TRIALO_CHECKPOINT_DIR") or preset["checkpoint_dir"]
    checkpoint_dir = Path(checkpoint)
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = repo_root / checkpoint_dir
    checkpoint_dir = checkpoint_dir.resolve()

    return Settings(
        model_config_name=config_name,
        base_model_id=base_model_id,
        checkpoint_dir=str(checkpoint_dir),
        dtype_name=os.environ.get("TRIALO_DTYPE") or preset["dtype"],
        max_length=int(os.environ.get("TRIALO_MAX_LENGTH", str(preset["max_length"]))),
        device_name=_resolve_device(),
        max_new_tokens=int(os.environ.get("TRIALO_MAX_NEW_TOKENS", "128")),
        repo_root=repo_root,
        frontend_origin=os.environ.get("TRIALO_FRONTEND_ORIGIN", "*"),
        extras={
            "preset_label": preset["label"],
            "adapter_overridden": bool(os.environ.get("TRIALO_CHECKPOINT_DIR")),
            "base_model_overridden": bool(os.environ.get("TRIALO_BASE_MODEL_ID")),
        },
    )
