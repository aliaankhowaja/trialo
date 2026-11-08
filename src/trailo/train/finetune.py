"""LoRA fine-tuning for TrialGPT criterion matching."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import logging
import math
import os
import random
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from trailo.data.prompting import tokenize_ids
from trailo.paths import CHECKPOINTS_DIR, LOGS_DIR, repo_path
from trailo.train.config import config_hash, get_section, load_config
from trailo.train.data import CausalLMCollator, PreparedCriterionDataset, load_jsonl
from trailo.train.modeling import (
    GIB,
    attach_lora,
    attention_backend,
    autocast_context,
    environment_info,
    flash_attention_available,
    load_adapter_model,
    load_model,
    load_tokenizer,
    memory_report,
    resolve_dtype,
    trainable_parameter_summary,
)

LOGGER = logging.getLogger("trailo.finetune")


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and torch for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def sha256_file(path: str | Path) -> str:
    """Return the SHA256 digest of a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_checkpoint(
    model,
    tokenizer,
    path: Path,
    *,
    config: dict[str, Any],
    config_path: Path,
    environment: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Save adapter, tokenizer, config, environment, metadata, and checksum."""
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    shutil.copy2(config_path, path / "config_source.yaml")
    (path / "config_resolved.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n"
    )
    (path / "environment.json").write_text(
        json.dumps(environment, indent=2, sort_keys=True) + "\n"
    )
    adapter_path = path / "adapter_model.safetensors"
    checksum = sha256_file(adapter_path) if adapter_path.exists() else ""
    metadata = dict(metadata)
    metadata["adapter_sha256"] = checksum
    metadata["saved_at"] = datetime.now(timezone.utc).isoformat()
    (path / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    if checksum:
        (path / "SHA256SUMS").write_text(
            f"{checksum}  adapter_model.safetensors\n"
        )
    return metadata


def evaluate_loss(
    model,
    loader: DataLoader,
    device: torch.device,
    dtype: torch.dtype,
) -> float:
    """Compute token-weighted validation loss."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad(), autocast_context(device, dtype):
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            output = model(**batch)
            tokens = int((batch["labels"] != -100).sum().item())
            total_loss += float(output.loss.item()) * tokens
            total_tokens += tokens
    model.train()
    return total_loss / total_tokens if total_tokens else float("nan")


def forward_last_logits(model, tokenizer, record: dict[str, Any], device, dtype):
    """Return the final-position logits for a fixed example."""
    input_ids = tokenize_ids(tokenizer, record["full_text"])
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_tensor)
    model.eval()
    with torch.no_grad(), autocast_context(device, dtype):
        output = model(input_ids=input_tensor, attention_mask=attention_mask)
    return output.logits[0, -1].float().cpu()


def configure_logging(log_path: Path) -> None:
    """Configure stdout and per-run file logging."""
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(console)
    LOGGER.addHandler(file_handler)


def lr_lambda_for(total_steps: int, warmup_steps: int):
    """Linear warmup followed by linear decay."""

    def schedule(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        remaining = max(1, total_steps - warmup_steps)
        return max(0.0, (total_steps - step) / remaining)

    return schedule


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--subset-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--precision", default=None)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--verify-reload", dest="verify_reload", action="store_true")
    parser.add_argument("--no-verify-reload", dest="verify_reload", action="store_false")
    parser.set_defaults(verify_reload=True)
    parser.add_argument("--wandb", dest="wandb_enabled", action="store_true")
    parser.add_argument("--no-wandb", dest="wandb_enabled", action="store_false")
    parser.set_defaults(wandb_enabled=None)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-mode", default=None,
                        choices=("online", "offline", "disabled"))
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    config = load_config(config_path)
    model_config = get_section(config, "model")
    data_config = get_section(config, "data")
    lora_config = get_section(config, "lora")
    train_config = get_section(config, "train")

    seed = int(args.seed if args.seed is not None else config["seed"])
    device = torch.device(args.device or train_config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but no GPU is available")
    precision = str(args.precision or train_config["precision"])
    dtype = resolve_dtype(precision, device)
    model_id = args.model_id or model_config["model_id"]
    revision = args.revision or model_config.get("revision")
    max_length = int(args.max_length or data_config["max_length"])
    max_steps = int(
        args.max_steps
        if args.max_steps is not None
        else train_config.get("max_steps", -1)
    )
    subset_size = (
        args.subset_size
        if args.subset_size is not None
        else train_config.get("subset_size")
    )
    run_name = config.get("run_name", "run")
    run_id = args.run_id or f"{run_name}_seed{seed}"
    output_dir = Path(args.output_dir or CHECKPOINTS_DIR / run_id)
    log_dir = LOGS_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(log_dir / f"{run_id}_stdout.log")

    resolved_config = json.loads(json.dumps(config))
    resolved_config["run_name"] = run_name
    resolved_config["run_id"] = run_id
    resolved_config["seed"] = seed
    resolved_config["model"]["model_id"] = model_id
    resolved_config["model"]["revision"] = revision
    resolved_config["data"]["max_length"] = max_length
    resolved_config["train"]["device"] = str(device)
    resolved_config["train"]["precision"] = precision
    resolved_config["train"]["max_steps"] = max_steps
    resolved_config["train"]["subset_size"] = subset_size

    wandb_section = dict(config.get("wandb") or {})
    if args.wandb_enabled is not None:
        wandb_section["enabled"] = bool(args.wandb_enabled)
    if args.wandb_project is not None:
        wandb_section["project"] = args.wandb_project
    if args.wandb_entity is not None:
        wandb_section["entity"] = args.wandb_entity
    if args.wandb_mode is not None:
        wandb_section["mode"] = args.wandb_mode
    resolved_config["wandb"] = wandb_section

    set_seed(seed)
    total_start = time.time()
    LOGGER.info("run_id=%s model=%s revision=%s", run_id, model_id, revision)
    LOGGER.info("device=%s precision=%s dtype=%s", device, precision, dtype)
    LOGGER.info("memory=%s", memory_report())

    processed_dir = repo_path(data_config["processed_dir"])
    train_records = load_jsonl(processed_dir / "train.jsonl")
    validation_records = load_jsonl(processed_dir / "validation.jsonl")
    if "test" in str(processed_dir):
        raise RuntimeError("training must never load the test split")
    if subset_size is not None:
        train_records = train_records[: int(subset_size)]
        validation_records = validation_records[: int(subset_size)]
        LOGGER.info(
            "smoke subset: train=%d validation=%d",
            len(train_records),
            len(validation_records),
        )

    tokenizer = load_tokenizer(model_id, revision)
    model = load_model(
        model_id,
        revision,
        device,
        precision,
        model_config.get("attn_implementation", "sdpa"),
        bool(model_config.get("allow_cpu_oversize", False)),
    )
    model.config.use_cache = False
    model, target_modules = attach_lora(
        model,
        rank=int(lora_config["rank"]),
        alpha=int(lora_config["alpha"]),
        dropout=float(lora_config["dropout"]),
        target_suffixes=list(lora_config["target_modules"]),
    )
    model.to(device)
    if bool(train_config.get("gradient_checkpointing", False)):
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        LOGGER.info("gradient checkpointing enabled")
    LOGGER.info("LoRA target modules: %d", len(target_modules))
    LOGGER.info("LoRA targets: %s", target_modules[:3] + target_modules[-3:])
    parameter_summary = trainable_parameter_summary(model)
    LOGGER.info("parameters=%s", parameter_summary)
    train_attention_backend = attention_backend(model)
    LOGGER.info("attention backend=%s", train_attention_backend)

    wandb_run = None
    if wandb_section.get("enabled", False):
        try:
            import wandb
        except ImportError as error:
            raise SystemExit(
                "wandb is enabled but not installed; run `pip install wandb` "
                "or set `wandb.enabled: false`"
            ) from error
        init_kwargs = {
            "project": wandb_section.get("project", "trialo"),
            "entity": wandb_section.get("entity"),
            "name": run_id,
            "config": resolved_config,
            "mode": wandb_section.get("mode", "offline"),
            "dir": str(log_dir),
            "tags": wandb_section.get("tags") or [run_name],
            "reinit": True,
        }
        init_kwargs = {
            key: value for key, value in init_kwargs.items() if value is not None
        }
        wandb_run = wandb.init(**init_kwargs)
        if wandb_section.get("log_gradients", False):
            wandb.watch(model, log="gradients",
                        log_freq=int(train_config["logging_steps"]))
        LOGGER.info("wandb run started: mode=%s id=%s",
                    init_kwargs.get("mode"), wandb_run.id)

    environment = environment_info(device, precision, model, resolved_config)
    env_path = log_dir / f"{run_id}_environment.json"
    env_path.write_text(json.dumps(environment, indent=2, sort_keys=True) + "\n")

    train_dataset = PreparedCriterionDataset(
        train_records, tokenizer, max_length, drop_truncated=True
    )
    validation_dataset = PreparedCriterionDataset(
        validation_records, tokenizer, max_length, drop_truncated=False
    )
    if len(train_dataset) == 0 or len(validation_dataset) == 0:
        raise SystemExit("empty training or validation dataset")
    collator = CausalLMCollator(tokenizer.pad_token_id)
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=int(train_config["batch_size"]),
        shuffle=False,
        collate_fn=collator,
    )

    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(train_config["learning_rate"]),
        weight_decay=float(train_config["weight_decay"]),
    )
    gradient_accumulation = int(train_config["gradient_accumulation"])
    epochs = int(train_config["epochs"])
    steps_per_epoch = math.ceil(len(train_dataset) / (
        int(train_config["batch_size"]) * gradient_accumulation
    ))
    planned_steps = steps_per_epoch * epochs
    if max_steps > 0:
        planned_steps = min(planned_steps, max_steps)
    planned_steps = max(1, planned_steps)
    warmup_steps = int(round(float(train_config["warmup_ratio"]) * planned_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda_for(planned_steps, warmup_steps)
    )

    curves_path = log_dir / f"{run_id}_training_curves.csv"
    events_path = log_dir / f"{run_id}_training_events.jsonl"
    curves_handle = curves_path.open("w", newline="")
    curves_writer = csv.writer(curves_handle)
    curves_writer.writerow(
        [
            "step",
            "epoch",
            "train_loss",
            "learning_rate",
            "grad_norm",
            "val_loss",
            "examples_per_second",
            "tokens_per_second",
            "peak_memory_gb",
            "wall_time_seconds",
        ]
    )
    events_handle = events_path.open("w")
    best_val_loss = float("inf")
    best_step = -1
    evals_without_improvement = 0
    last_eval_step = -1
    global_step = 0
    epoch = 0
    seen_examples = 0
    seen_tokens = 0
    train_seconds = 0.0
    peak_memory_gb = 0.0
    stop_training = False
    last_train_loss = float("nan")
    last_grad_norm = float("nan")
    last_val_loss = float("nan")
    training_start = time.time()

    try:
        for epoch in range(1, epochs + 1):
            generator = torch.Generator().manual_seed(seed + epoch)
            train_loader = DataLoader(
                train_dataset,
                batch_size=int(train_config["batch_size"]),
                shuffle=True,
                generator=generator,
                collate_fn=collator,
            )
            model.train()
            optimizer.zero_grad(set_to_none=True)
            micro_in_step = 0
            loss_sum = 0.0
            for batch_index, batch in enumerate(train_loader, start=1):
                batch = {key: value.to(device) for key, value in batch.items()}
                with autocast_context(device, dtype):
                    output = model(**batch)
                    raw_loss = output.loss
                loss = raw_loss / gradient_accumulation
                loss.backward()
                micro_in_step += 1
                loss_sum += float(raw_loss.item())
                seen_examples += int(batch["input_ids"].shape[0])
                seen_tokens += int(batch["attention_mask"].sum().item())

                should_step = (
                    micro_in_step >= gradient_accumulation
                    or batch_index == len(train_loader)
                )
                if not should_step:
                    continue
                last_grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        trainable_parameters,
                        float(train_config["max_grad_norm"]),
                    )
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                last_train_loss = loss_sum / max(1, micro_in_step)
                micro_in_step = 0
                loss_sum = 0.0
                global_step += 1
                train_seconds = time.time() - training_start

                should_evaluate = (
                    int(train_config["eval_every_steps"]) > 0
                    and global_step % int(train_config["eval_every_steps"]) == 0
                )
                val_loss = None
                if should_evaluate:
                    val_loss = evaluate_loss(
                        model, validation_loader, device, dtype
                    )
                    last_eval_step = global_step
                    last_val_loss = val_loss
                    event = {
                        "event": "validation",
                        "step": global_step,
                        "epoch": epoch,
                        "val_loss": val_loss,
                        "wall_time_seconds": time.time() - total_start,
                    }
                    events_handle.write(json.dumps(event) + "\n")
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_step = global_step
                        evals_without_improvement = 0
                        save_checkpoint(
                            model,
                            tokenizer,
                            output_dir / "best",
                            config=resolved_config,
                            config_path=config_path,
                            environment=environment,
                            metadata={
                                "step": global_step,
                                "epoch": epoch,
                                "val_loss": val_loss,
                                "train_loss": last_train_loss,
                                "is_best": True,
                            },
                        )
                        LOGGER.info(
                            "new best validation loss %.6f at step %d",
                            best_val_loss,
                            best_step,
                        )
                    else:
                        evals_without_improvement += 1
                    LOGGER.info(
                        "step=%d epoch=%d val_loss=%.6f best=%.6f",
                        global_step,
                        epoch,
                        val_loss,
                        best_val_loss,
                    )
                    if wandb_run is not None:
                        wandb_run.log(
                            {
                                "val/loss": val_loss,
                                "val/best_loss": best_val_loss,
                                "epoch": epoch,
                            },
                            step=global_step,
                        )

                if device.type == "cuda":
                    peak_memory_gb = max(
                        peak_memory_gb,
                        torch.cuda.max_memory_allocated(device) / GIB,
                    )
                else:
                    peak_memory_gb = max(
                        peak_memory_gb,
                        __import__("psutil").Process().memory_info().rss / GIB,
                    )
                elapsed = max(1e-9, time.time() - training_start)
                examples_per_second = seen_examples / elapsed
                tokens_per_second = seen_tokens / elapsed
                row = [
                    global_step,
                    epoch,
                    last_train_loss,
                    scheduler.get_last_lr()[0],
                    last_grad_norm,
                    "" if val_loss is None else val_loss,
                    examples_per_second,
                    tokens_per_second,
                    peak_memory_gb,
                    time.time() - total_start,
                ]
                curves_writer.writerow(row)
                curves_handle.flush()
                train_event = {
                    "event": "train_step",
                    "step": global_step,
                    "epoch": epoch,
                    "train_loss": last_train_loss,
                    "learning_rate": scheduler.get_last_lr()[0],
                    "grad_norm": last_grad_norm,
                    "examples_per_second": examples_per_second,
                    "tokens_per_second": tokens_per_second,
                    "peak_memory_gb": peak_memory_gb,
                }
                events_handle.write(json.dumps(train_event) + "\n")
                events_handle.flush()
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "train/loss": last_train_loss,
                            "train/lr": scheduler.get_last_lr()[0],
                            "train/grad_norm": last_grad_norm,
                            "train/examples_per_second": examples_per_second,
                            "train/tokens_per_second": tokens_per_second,
                            "epoch": epoch,
                        },
                        step=global_step,
                    )
                if global_step % int(train_config["logging_steps"]) == 0:
                    LOGGER.info(
                        "step=%d epoch=%d train_loss=%.6f lr=%.3e grad_norm=%.3f",
                        global_step,
                        epoch,
                        last_train_loss,
                        scheduler.get_last_lr()[0],
                        last_grad_norm,
                    )

                if max_steps > 0 and global_step >= max_steps:
                    stop_training = True
                    break
                if (
                    evals_without_improvement
                    >= int(train_config["early_stopping_patience"])
                ):
                    LOGGER.info("early stopping after %d non-improving evaluations",
                                evals_without_improvement)
                    stop_training = True
                    break

            if last_eval_step != global_step:
                val_loss = evaluate_loss(model, validation_loader, device, dtype)
                last_eval_step = global_step
                last_val_loss = val_loss
                event = {
                    "event": "validation",
                    "step": global_step,
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "wall_time_seconds": time.time() - total_start,
                }
                events_handle.write(json.dumps(event) + "\n")
                events_handle.flush()
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_step = global_step
                    evals_without_improvement = 0
                    save_checkpoint(
                        model,
                        tokenizer,
                        output_dir / "best",
                        config=resolved_config,
                        config_path=config_path,
                        environment=environment,
                        metadata={
                            "step": global_step,
                            "epoch": epoch,
                            "val_loss": val_loss,
                            "train_loss": last_train_loss,
                            "is_best": True,
                        },
                    )
                    LOGGER.info(
                        "new best validation loss %.6f at step %d",
                        best_val_loss,
                        best_step,
                    )
                else:
                    evals_without_improvement += 1
                LOGGER.info(
                    "epoch=%d step=%d val_loss=%.6f best=%.6f",
                    epoch,
                    global_step,
                    val_loss,
                    best_val_loss,
                )
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "val/loss": val_loss,
                            "val/best_loss": best_val_loss,
                            "epoch": epoch,
                        },
                        step=global_step,
                    )

            save_checkpoint(
                model,
                tokenizer,
                output_dir / f"epoch-{epoch}",
                config=resolved_config,
                config_path=config_path,
                environment=environment,
                metadata={
                    "step": global_step,
                    "epoch": epoch,
                    "val_loss": last_val_loss,
                    "train_loss": last_train_loss,
                    "is_best": False,
                },
            )
            if (
                evals_without_improvement
                >= int(train_config["early_stopping_patience"])
            ):
                LOGGER.info(
                    "early stopping after %d non-improving evaluations",
                    evals_without_improvement,
                )
                stop_training = True
            if stop_training:
                break
    finally:
        curves_handle.close()
        events_handle.close()

    training_seconds = max(0.0, time.time() - training_start)
    final_metadata = save_checkpoint(
        model,
        tokenizer,
        output_dir / "final",
        config=resolved_config,
        config_path=config_path,
        environment=environment,
        metadata={
            "step": global_step,
            "epoch": epoch,
            "val_loss": None if math.isnan(last_val_loss) else last_val_loss,
            "train_loss": last_train_loss,
            "is_best": False,
        },
    )
    if best_step < 0:
        shutil.copytree(output_dir / "final", output_dir / "best", dirs_exist_ok=True)

    reload_check = {"performed": False, "passed": None}
    reload_attention_backend = None
    if args.verify_reload:
        LOGGER.info("verifying checkpoint reload on one example")
        fixed_record = validation_records[0]
        reference_logits = forward_last_logits(
            model, tokenizer, fixed_record, device, dtype
        )
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        base_model = load_model(
            model_id,
            revision,
            device,
            precision,
            model_config.get("attn_implementation", "sdpa"),
            bool(model_config.get("allow_cpu_oversize", False)),
        )
        reloaded = load_adapter_model(base_model, str(output_dir / "final"))
        reloaded.to(device)
        reload_attention_backend = attention_backend(reloaded)
        reloaded_logits = forward_last_logits(
            reloaded, tokenizer, fixed_record, device, dtype
        )
        max_difference = float((reference_logits - reloaded_logits).abs().max())
        passed = bool(
            torch.allclose(reference_logits, reloaded_logits, atol=1e-4, rtol=1e-3)
        )
        reload_check = {
            "performed": True,
            "passed": passed,
            "max_abs_difference": max_difference,
            "reference_shape": list(reference_logits.shape),
            "reloaded_shape": list(reloaded_logits.shape),
            "annotation_id": fixed_record["annotation_id"],
        }
        (output_dir / "reload_check.json").write_text(
            json.dumps(reload_check, indent=2, sort_keys=True) + "\n"
        )
        LOGGER.info("reload check: passed=%s max_abs_diff=%.3e", passed, max_difference)
        del reloaded
        del base_model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    total_seconds = time.time() - total_start
    peak_allocated = (
        torch.cuda.max_memory_allocated(device) / GIB
        if device.type == "cuda"
        else None
    )
    peak_reserved = (
        torch.cuda.max_memory_reserved(device) / GIB
        if device.type == "cuda"
        else None
    )
    hardware = {
        "run_id": run_id,
        "run_name": run_name,
        "model_id": model_id,
        "model_revision": revision,
        "device": str(device),
        "precision": precision,
        "attention_backend": train_attention_backend,
        "reload_attention_backend": reload_attention_backend,
        "flash_attention_available": flash_attention_available(),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "peak_gpu_memory_allocated_gb": peak_allocated,
        "peak_gpu_memory_reserved_gb": peak_reserved,
        "peak_cpu_rss_gb": (
            __import__("psutil").Process().memory_info().rss / GIB
            if device.type == "cpu"
            else None
        ),
        "parameter_summary": parameter_summary,
        "lora_target_modules": target_modules,
        "training_seconds": training_seconds,
        "total_seconds": total_seconds,
        "optimizer_steps": global_step,
        "examples_processed": seen_examples,
        "examples_per_second": seen_examples / training_seconds
        if training_seconds
        else None,
        "tokens_per_second": seen_tokens / training_seconds
        if training_seconds
        else None,
        "best_val_loss": None if best_step < 0 else best_val_loss,
        "best_step": best_step,
        "final_adapter_sha256": final_metadata["adapter_sha256"],
        "config_sha256": config_hash(resolved_config),
        "reload_check": reload_check,
        "cuda_version": torch.version.cuda,
        "torch_version": torch.__version__,
        "wandb": {
            "enabled": wandb_run is not None,
            "project": wandb_section.get("project"),
            "mode": wandb_section.get("mode") if wandb_run is not None else None,
            "run_id": wandb_run.id if wandb_run is not None else None,
            "url": getattr(wandb_run, "url", None) if wandb_run is not None else None,
        },
    }
    (log_dir / f"{run_id}_hardware.json").write_text(
        json.dumps(hardware, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "hardware.json").write_text(
        json.dumps(hardware, indent=2, sort_keys=True) + "\n"
    )
    LOGGER.info("hardware summary: %s", hardware)
    print(json.dumps(hardware, indent=2, sort_keys=True))
    if wandb_run is not None:
        wandb_run.summary["best_val_loss"] = hardware["best_val_loss"]
        wandb_run.summary["optimizer_steps"] = global_step
        wandb_run.summary["tokens_per_second"] = hardware["tokens_per_second"]
        wandb_run.summary["final_adapter_sha256"] = final_metadata["adapter_sha256"]
        wandb_run.finish()
    if args.verify_reload and not reload_check.get("passed"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
