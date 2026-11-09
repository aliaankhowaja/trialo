"""Model engine: note normalization, prompt construction, and criterion scoring.

The heavy lifting (greedy decoding path selection, strict JSON parsing, timing)
is delegated to the project's own evaluation code (`trailo.eval.inference`,
`trailo.eval.decoding`, `trailo.eval.parsing`) so the app serves exactly the
machinery the research repo validated - no parallel implementation.
"""

from __future__ import annotations

import re
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any

from .config import Settings, load_settings

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass
class ScreenedCriterion:
    """One criterion-level result, mirroring `generate_prediction` output."""

    index: int
    criterion_type: str
    criterion_text: str
    label: str | None
    evidence_ids: list[int]
    explanation: str
    valid: bool
    errors: list[str]
    warnings: list[str]
    raw_text: str
    prompt_tokens: int
    generated_tokens: int
    stopped_on_eos: bool
    wall_seconds: float
    decoding_timing: dict[str, Any]
    token_ids: list[int]


class NoteFormatError(ValueError):
    """Raised when a patient note cannot be normalized into numbered sentences."""


def normalize_note(text: str) -> tuple[str, list[str]]:
    """Turn free-text (or already numbered) notes into the 'N. sentence' format.

    Plain text is split into declarative sentences; already numbered lines are
    kept as-is (their original numbering is respected). The output matches the
    training-time note format so the prompt template is reproduced bit-for-bit.
    """
    if not isinstance(text, str) or not text.strip():
        raise NoteFormatError("patient note is empty")

    source_lines = text.replace("\r\n", "\n").split("\n")
    numbered_present = all(
        re.match(r"^\s*\d+\.\s?\S", line) or not line.strip()
        for line in source_lines
        if line.strip()
    ) and any(line.strip() for line in source_lines)

    if numbered_present:
        sentences: list[str] = []
        for line in source_lines:
            m = re.match(r"^\s*(\d+)\.\s?(.*)$", line)
            sentences.append(m.group(2).strip() if m else line.strip())
        sentences = [s for s in sentences if s]
    else:
        sentences = []
        for paragraph in source_lines:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            for chunk in _SENTENCE_SPLIT.split(paragraph):
                chunk = chunk.strip()
                if chunk:
                    sentences.append(chunk)


    if not sentences:
        raise NoteFormatError("patient note produced no sentences")
    note = "\n".join(f"{i}. {s}" for i, s in enumerate(sentences))
    return note, sentences


class Engine:
    """Lazy-loaded Gemma + LoRA adapter wrapper around `trailo.eval`."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or load_settings()
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._load_error: str | None = None
        self._effective_dtype_name: str | None = None
        self._lock = threading.Lock()
        self._infer_lock = threading.Lock()  # serialize model forward passes

    # ------------------------------------------------------------------ load
    def _ensure_repo_on_path(self) -> None:
        src = str(self.settings.repo_src_path)
        if src not in sys.path:
            sys.path.insert(0, src)

    def ensure_loaded(self) -> None:
        with self._lock:
            if self._model is not None:
                return
            # A previous load failed: never proceed with a half-loaded engine.
            if self._load_error is not None:
                raise RuntimeError(f"model failed to load: {self._load_error}")
            self._load()

    def _load(self) -> None:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._ensure_repo_on_path()
        settings = self.settings
        device = settings.device_name
        dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
        dtype = dtype_map.get(settings.dtype_name.lower(), torch.float32)
        if device == "cpu" and dtype is torch.bfloat16:
            # bf16 CPU matmul is slow and numerically marginal; fp32 is the safe path.
            dtype = torch.float32

        adapter_dir = settings.checkpoint_dir
        try:
            # Prefer the tokenizer saved alongside the adapter (imports the
            # exact chat template the model was fine-tuned with).
            try:
                tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
            except Exception:
                tokenizer = AutoTokenizer.from_pretrained(settings.base_model_id)
            if getattr(tokenizer, "pad_token_id", None) is None:
                tokenizer.pad_token = tokenizer.eos_token

            kwargs = {
                "dtype": dtype,
                "low_cpu_mem_usage": True,
                "attn_implementation": "sdpa",
            }
            try:
                base = AutoModelForCausalLM.from_pretrained(settings.base_model_id, **kwargs)
            except TypeError:
                # transformers<5 expects torch_dtype instead of dtype
                kwargs.pop("dtype")
                kwargs["torch_dtype"] = dtype
                base = AutoModelForCausalLM.from_pretrained(settings.base_model_id, **kwargs)
            except (ValueError, ImportError):
                kwargs.pop("dtype", None)
                kwargs.pop("torch_dtype", None)
                kwargs["attn_implementation"] = "eager"
                base = AutoModelForCausalLM.from_pretrained(settings.base_model_id, **kwargs)

            model = PeftModel.from_pretrained(base, adapter_dir)
            model.eval()
            model.to(device)
            self._model = model
            self._tokenizer = tokenizer
            self._effective_dtype_name = str(dtype).replace("torch.", "")
        except Exception as exc:  # surface the load error via the API
            self._load_error = f"{type(exc).__name__}: {exc}"
            raise

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def status(self) -> dict[str, Any]:
        return {
            "model_preset": self.settings.model_config_name,
            "preset_label": self.settings.extras.get("preset_label"),
            "base_model_id": self.settings.base_model_id,
            "checkpoint_dir": self.settings.checkpoint_dir,
            "adapter_overridden": self.settings.extras.get("adapter_overridden", False),
            "base_model_overridden": self.settings.extras.get("base_model_overridden", False),
            "loaded": self._model is not None,
            "load_error": self._load_error,
            "device": self.settings.device_name,
            "dtype": self._effective_dtype_name or self.settings.dtype_name,
            "max_new_tokens_default": self.settings.max_new_tokens,
            "max_length": self.settings.max_length,
        }

    # ------------------------------------------------------------------ prompt
    def _build_example(
        self,
        note: str,
        n_sentences: int,
        criterion_type: str,
        criterion_text: str,
    ) -> dict[str, Any]:
        """Build the exact fine-tuning-time chat prompt for one criterion."""
        self._ensure_repo_on_path()
        from trailo.data.prompting import SYSTEM_PROMPT, build_user_prompt

        item = {
            "note": note.rstrip("\n"),
            "criterion_type": criterion_type,
            "criterion_text": criterion_text,
        }
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(item, self.settings.label_sets)},
        ]
        prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return {
            "prompt": prompt,
            "criterion_type": criterion_type,
            "n_sentences": n_sentences,
        }

    # ------------------------------------------------------------------ screen
    def screen_criterion(
        self,
        *,
        index: int,
        criterion_type: str,
        criterion_text: str,
        note: str,
        sentences: list[str],
        decoding_impl: str = "generate",
        max_new_tokens: int | None = None,
    ) -> ScreenedCriterion:
        """Score one criterion against the patient note.

        Evidence sentence IDs from the model are resolved to the exact sentence
        text so the UI can quote them verbatim.
        """
        self.ensure_loaded()
        self._ensure_repo_on_path()
        # trailo.eval.inference is used for its private single-example helpers
        # (_encode_prompt / _decode_with_generate) so the prompt encoding and
        # the library decode path match evaluation exactly; the public
        # generate_prediction wrapper is not used because it does not expose
        # the generated token ids, which the benchmark needs for the
        # stream-agreement comparison.
        import trailo.eval.inference as inference
        from trailo.eval.decoding import (
            generation_eos_token_ids,
            greedy_generate_cached,
            greedy_generate_recompute,
        )
        from trailo.eval.parsing import parse_and_validate

        n_sentences = len(sentences)
        example = self._build_example(note, n_sentences, criterion_type, criterion_text)
        max_new = max_new_tokens or self.settings.max_new_tokens

        with self._infer_lock:
            start = time.perf_counter()
            input_ids, attention_mask = inference._encode_prompt(
                self._model, self._tokenizer, example
            )
            eos_token_ids = generation_eos_token_ids(self._model, self._tokenizer)
            pad_token_id = self._tokenizer.pad_token_id
            if pad_token_id is None:
                pad_token_id = eos_token_ids[0]

            if decoding_impl == "generate":
                decoded = inference._decode_with_generate(
                    self._model,
                    input_ids,
                    attention_mask,
                    eos_token_ids,
                    pad_token_id,
                    max_new,
                )
                token_ids = decoded["token_ids"]
                timing: dict[str, Any] = {"decoding_impl": "generate"}
            elif decoding_impl in {"cache", "recompute"}:
                loop = (
                    greedy_generate_cached
                    if decoding_impl == "cache"
                    else greedy_generate_recompute
                )
                result = loop(
                    self._model,
                    input_ids,
                    attention_mask,
                    max_new,
                    eos_token_ids,
                )
                token_ids = list(result.generated_ids)
                timing = {"decoding_impl": decoding_impl, **result.to_dict()}
            else:
                raise ValueError(f"unknown decoding implementation {decoding_impl!r}")
            wall = time.perf_counter() - start

        raw_text = self._tokenizer.decode(token_ids, skip_special_tokens=True).strip()
        labels = self.settings.label_sets
        try:
            parsed = parse_and_validate(
                raw_text,
                criterion_type=criterion_type,
                n_sentences=n_sentences,
                allowed_labels=labels[criterion_type],
            )
            prediction = parsed.prediction or {}
            errors = list(parsed.errors)
            warnings = list(parsed.warnings)
        except Exception as exc:  # parse failure surfaces as invalid, not a crash
            prediction = {}
            errors = [f"{type(exc).__name__}: {exc}"]
            warnings = []

        evidence_ids = prediction.get("evidence_ids") or []
        evidence_ids = [
            int(e) for e in evidence_ids
            if isinstance(e, (int, float)) and 0 <= int(e) < n_sentences
        ]
        return ScreenedCriterion(
            index=index,
            criterion_type=criterion_type,
            criterion_text=criterion_text,
            label=prediction.get("label"),
            evidence_ids=evidence_ids,
            explanation=str(prediction.get("explanation", "")),
            valid=not errors,
            errors=errors,
            warnings=warnings,
            raw_text=raw_text,
            prompt_tokens=int(input_ids.shape[1]),
            generated_tokens=len(token_ids),
            stopped_on_eos=bool(token_ids) and token_ids[-1] in set(eos_token_ids),
            wall_seconds=wall,
            decoding_timing=timing,
            token_ids=token_ids,
        )


_engine: Engine | None = None
_engine_lock = threading.Lock()


def get_engine() -> Engine:
    global _engine
    with _engine_lock:
        if _engine is None:
            _engine = Engine()
        return _engine
