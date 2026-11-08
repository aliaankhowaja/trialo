"""Model, tokenizer, precision, attention, and LoRA utilities."""

from __future__ import annotations

import contextlib
import json
import platform
import sys
from functools import lru_cache
from typing import Any, Iterator

import psutil
import torch
from huggingface_hub import HfApi
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, GemmaTokenizerFast

GIB = 1024**3


def load_tokenizer(model_id: str, revision: str | None = None):
    """Load a fast tokenizer, falling back to GemmaTokenizerFast explicitly."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        revision=revision,
        use_fast=True,
    )
    if not tokenizer.is_fast:
        tokenizer = GemmaTokenizerFast.from_pretrained(model_id, revision=revision)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
    return tokenizer


def cpu_bf16_supported() -> bool:
    """Return whether this CPU can execute bf16 matmuls."""
    try:
        left = torch.ones(2, 2, dtype=torch.bfloat16)
        result = left @ left
        return result.dtype == torch.bfloat16
    except Exception:
        return False


def resolve_dtype(precision: str, device: torch.device) -> torch.dtype:
    """Resolve a configured precision string to a torch dtype."""
    precision = precision.lower()
    if precision in {"bf16", "bfloat16"}:
        if device.type == "cpu" and not cpu_bf16_supported():
            raise RuntimeError("bf16 requested on a CPU without bf16 matmul support")
        return torch.bfloat16
    if precision in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"unsupported precision {precision!r}")


@contextlib.contextmanager
def autocast_context(device: torch.device, dtype: torch.dtype) -> Iterator[None]:
    """Use bf16 autocast when requested; fp32 needs no autocast."""
    if dtype == torch.bfloat16:
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            yield
    else:
        with contextlib.nullcontext():
            yield


@lru_cache(maxsize=8)
def checkpoint_size_gb(model_id: str, revision: str | None = None) -> float:
    """Return the total size of safetensors/checkpoint files in GB.

    When the Hub is unreachable (offline smoke tests), return 0 and let the
    caller rely on the local memory report instead of failing preflight.
    """
    api = HfApi()
    try:
        info = api.model_info(model_id, revision=revision, files_metadata=True)
    except Exception:
        return 0.0
    total = sum(
        (sibling.size or 0)
        for sibling in info.siblings
        if sibling.rfilename.endswith((".safetensors", ".bin", ".pt"))
    )
    return total / 1e9 if total else 0.0


def memory_report() -> dict[str, float]:
    """Return current system memory in GB."""
    memory = psutil.virtual_memory()
    return {
        "total_gb": memory.total / GIB,
        "available_gb": memory.available / GIB,
        "used_gb": memory.used / GIB,
    }


def assert_sufficient_memory(
    model_id: str,
    revision: str | None,
    device: torch.device,
    allow_oversize: bool = False,
) -> dict[str, Any]:
    """Refuse to load a checkpoint that is obviously too large for CPU RAM."""
    report = memory_report()
    size_gb = checkpoint_size_gb(model_id, revision)
    report["checkpoint_gb"] = size_gb
    required_gb = size_gb * 1.15
    report["required_gb"] = required_gb
    if (
        device.type == "cpu"
        and size_gb > 0
        and required_gb > report["available_gb"]
        and not allow_oversize
    ):
        raise RuntimeError(
            f"{model_id} needs about {required_gb:.1f} GB but only "
            f"{report['available_gb']:.1f} GB RAM is available. Use the documented "
            "CPU smoke checkpoint or pass allow_cpu_oversize=true explicitly."
        )
    return report


def load_model(
    model_id: str,
    revision: str | None,
    device: torch.device,
    precision: str,
    attn_implementation: str = "sdpa",
    allow_cpu_oversize: bool = False,
):
    """Load a base causal LM with a configurable device, dtype, and attention."""
    assert_sufficient_memory(model_id, revision, device, allow_cpu_oversize)
    dtype = resolve_dtype(precision, device)
    kwargs = {
        "revision": revision,
        "dtype": dtype,
        "low_cpu_mem_usage": True,
        "attn_implementation": attn_implementation,
    }
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    except TypeError:
        kwargs.pop("dtype")
        kwargs["torch_dtype"] = dtype
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    except (ValueError, ImportError) as error:
        if kwargs.get("attn_implementation") == "eager":
            raise
        import warnings

        warnings.warn(
            f"attn_implementation={kwargs.get('attn_implementation')!r} failed "
            f"({error}); retrying with eager attention"
        )
        kwargs["attn_implementation"] = "eager"
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.to(device)
    model.eval()
    return model


def attention_backend(model) -> str:
    """Return the attention implementation actually used by the model."""
    config = getattr(model, "config", None)
    backend = getattr(config, "_attn_implementation", None)
    if backend is None and hasattr(config, "text_config"):
        backend = getattr(config.text_config, "_attn_implementation", None)
    return str(backend or "unknown")


def flash_attention_available() -> bool:
    """Report whether a FlashAttention backend is available in torch."""
    checker = getattr(torch.backends.cuda, "is_flash_attention_available", None)
    return bool(checker()) if checker is not None else False


def resolve_text_target_modules(model, suffixes: list[str]) -> list[str]:
    """Expand projection suffixes to exact text-layer module names."""
    excluded = ("audio_tower", "vision_tower", "embed_vision", "conformer")
    names = []
    for name, _module in model.named_modules():
        if ".layers." not in name:
            continue
        if any(fragment in name for fragment in excluded):
            continue
        if any(name.endswith("." + suffix) for suffix in suffixes):
            names.append(name)
    return sorted(set(names))


def attach_lora(
    model,
    *,
    rank: int,
    alpha: int,
    dropout: float,
    target_suffixes: list[str],
) -> tuple[PeftModel, list[str]]:
    """Attach LoRA to text attention/MLP projections and freeze base weights."""
    target_modules = resolve_text_target_modules(model, target_suffixes)
    if not target_modules:
        raise RuntimeError(
            f"no text modules matched LoRA target suffixes {target_suffixes}"
        )
    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model, config)
    for name, parameter in peft_model.named_parameters():
        parameter.requires_grad = "lora_" in name
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    trainable = [p for p in peft_model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("LoRA attachment produced no trainable parameters")
    return peft_model, target_modules


def trainable_parameter_summary(model) -> dict[str, Any]:
    """Return trainable parameter counts and percentage."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_percent": 100.0 * trainable / total if total else 0.0,
    }


def environment_info(
    device: torch.device,
    precision: str,
    model,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Collect versions and hardware information for checkpoint sidecars."""
    import datasets
    import peft
    import transformers

    info = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "datasets": datasets.__version__,
        "peft": peft.__version__,
        "numpy": __import__("numpy").__version__,
        "device": str(device),
        "precision": precision,
        "cpu_bf16_supported": cpu_bf16_supported(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "attention_backend": attention_backend(model),
        "flash_attention_available": flash_attention_available(),
        "memory": memory_report(),
    }
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_total_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / GIB
    if config is not None:
        info["config"] = config
    return info


def write_json(path, payload: dict[str, Any]) -> None:
    """Write deterministic pretty JSON."""
    from pathlib import Path

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def load_adapter_model(base_model, adapter_path: str, trainable: bool = False):
    """Load a saved PEFT adapter on top of a base model."""
    return PeftModel.from_pretrained(
        base_model,
        adapter_path,
        is_trainable=trainable,
    )
