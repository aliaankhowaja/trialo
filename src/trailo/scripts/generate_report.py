"""Generate a Markdown report from training and evaluation artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from trailo.paths import CHECKPOINTS_DIR, RESULTS_DIR, repo_path
from trailo.train.config import get_section, load_config


def _load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text())


def _metric_row(name: str, metrics: dict[str, Any]) -> str:
    evidence = metrics.get("evidence", {})
    return (
        f"| {name} | {metrics.get('n_examples', 0)} | "
        f"{metrics.get('parse_success_rate', 0):.3f} | "
        f"{metrics.get('label_accuracy', 0):.3f} | "
        f"{metrics.get('label_accuracy_parsed', 0):.3f} | "
        f"{metrics.get('macro_f1', 0):.3f} | "
        f"{evidence.get('precision', 0):.3f} | "
        f"{evidence.get('recall', 0):.3f} | "
        f"{evidence.get('f1', 0):.3f} | "
        f"{evidence.get('exact_match_rate', 0):.3f} |"
    )


def _criterion_rows(metrics: dict[str, Any]) -> list[str]:
    rows = []
    for criterion, values in sorted(metrics.get("per_criterion", {}).items()):
        rows.append(
            f"| {criterion} | {values.get('micro_support', 0)} | "
            f"{values.get('macro_f1', 0):.3f} |"
        )
    return rows


def _curve_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}
    val_rows = [row for row in rows if row.get("val_loss")]
    return {
        "n_steps_logged": len(rows),
        "first_train_loss": float(rows[0]["train_loss"]),
        "last_train_loss": float(rows[-1]["train_loss"]),
        "best_val_loss": (
            min(float(row["val_loss"]) for row in val_rows) if val_rows else None
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    model_config = get_section(config, "model")
    data_config = get_section(config, "data")
    train_config = get_section(config, "train")
    lora_config = get_section(config, "lora")
    eval_config = get_section(config, "eval")
    run_name = config.get("run_name", "run")
    run_id = args.run_id or f"{run_name}_seed{config['seed']}"
    is_smoke = "smoke" in run_name

    checkpoint_dir = CHECKPOINTS_DIR / run_id
    results_dir = RESULTS_DIR / run_name
    processed_dir = repo_path(data_config["processed_dir"])
    stats = _load_json(processed_dir / "stats.json", {})
    hardware = _load_json(checkpoint_dir / "hardware.json", {})
    best_metadata = _load_json(checkpoint_dir / "best" / "metadata.json", {})
    final_metadata = _load_json(checkpoint_dir / "final" / "metadata.json", {})
    reload_check = _load_json(checkpoint_dir / "reload_check.json", {})
    curves = _curve_summary(
        repo_path("src/trailo/logs") / f"{run_id}_training_curves.csv"
    )
    baseline_val = _load_json(results_dir / "baseline_validation_metrics.json")
    finetuned_val = _load_json(results_dir / "finetuned_validation_metrics.json")
    test_metrics = _load_json(results_dir / "test_metrics.json")

    split_lines = []
    for split in ("train", "validation", "test"):
        values = stats.get("splits", {}).get(split, {})
        split_lines.append(
            f"| {split} | {values.get('n_examples', '?')} | "
            f"{values.get('criteria', {}).get('inclusion', '?')} | "
            f"{values.get('criteria', {}).get('exclusion', '?')} | "
            f"{values.get('n_dropped', '?')} |"
        )

    lines = [
        f"# Trial Eligibility Fine-Tuning Report ({run_name})",
        "",
        (
            "> **SMOKE RUN.** This report describes a CPU pipeline smoke test with "
            "a 270m model and a handful of steps. Its metrics are not scientific "
            "results and must not be presented as the final experiment."
            if is_smoke
            else "> **FINAL RUN.** This report describes the Gemma 3n E4B run on an A100."
        ),
        "",
        "## Objective",
        "",
        "Fine-tune a Gemma model to map patient-note sentences to clinical-trial "
        "inclusion/exclusion criteria, producing the criterion label, supporting "
        "sentence IDs, and an explanation in a strict JSON contract. This stage "
        "does the fine-tuning only; retrieval is out of scope.",
        "",
        "## Data",
        "",
        f"- Source: `{stats.get('source_dataset', 'ncbi/TrialGPT-Criterion-Annotations')}`, "
        f"revision `{stats.get('source_revision', '?')}`.",
        "- 53 patients and 1,015 annotation examples in the full dataset; the "
        "provided split manifest is used unchanged.",
        "- Splits are patient-disjoint. The verification script checks patient "
        "disjointness, annotation disjointness, coverage, and expected counts.",
        "",
        "| Split | Examples | Inclusion | Exclusion | Dropped |",
        "| --- | ---: | ---: | ---: | ---: |",
        *split_lines,
        "",
        "- The `test` split is locked. It is never used for training, early "
        "stopping, prompt selection, threshold selection, or hyperparameter "
        "selection. It is evaluated once at the end by `evaluate_test.py`.",
        "- GPT-4 annotation columns (`gpt4_explanation`, `explanation_correctness`, "
        "`gpt4_sentences`, `gpt4_eligibility`, `training`) are never read.",
        "- There is no expert explanation field in this dataset, so every training "
        "target uses an empty explanation string (`\"\"`) rather than generated "
        "rationales.",
        f"- Chat template SHA-256: `{stats.get('chat_template_sha256', '?')}`.",
        "",
        "## Prompt and target",
        "",
        "The patient note is split into numbered sentences. The prompt asks for "
        "one JSON object with `label`, `evidence_ids`, and `explanation`. "
        "`evidence_ids` must refer to the numbered sentences and must be an empty "
        "list when no sentence supports the decision. Labels are constrained to "
        "the four inclusion labels or four exclusion labels.",
        "",
        "## Model and LoRA",
        "",
        f"- Base model: `{model_config['model_id']}` at revision "
        f"`{model_config.get('revision', '?')}`.",
        f"- LoRA rank {lora_config['rank']}, alpha {lora_config['alpha']}, "
        f"dropout {lora_config['dropout']}, on "
        f"{len(lora_config['target_modules'])} projection types per text layer.",
        "- Base weights are frozen. Only LoRA adapters are trained.",
        f"- Trainable parameters: "
        f"{hardware.get('parameter_summary', {}).get('trainable_parameters', '?')} "
        f"of {hardware.get('parameter_summary', {}).get('total_parameters', '?')} "
        f"({hardware.get('parameter_summary', {}).get('trainable_percent', 0):.3f}%).",
        "",
        "## Training setup",
        "",
        f"- Max sequence length: {data_config['max_length']} tokens.",
        f"- Batch size {train_config['batch_size']} with gradient accumulation "
        f"{train_config['gradient_accumulation']} "
        f"(effective batch {train_config['batch_size'] * train_config['gradient_accumulation']}).",
        f"- Epochs: {train_config['epochs']}; max steps: {train_config['max_steps']}.",
        f"- Learning rate: {train_config['learning_rate']}; weight decay "
        f"{train_config['weight_decay']}; warmup ratio "
        f"{train_config['warmup_ratio']}; grad clip {train_config['max_grad_norm']}.",
        f"- Precision: `{train_config['precision']}`; attention backend: "
        f"`{model_config.get('attn_implementation', 'sdpa')}`.",
        f"- Seed: {config['seed']}.",
        (
            f"- Weights & Biases: enabled, project "
            f"`{hardware.get('wandb', {}).get('project', '?')}`, mode "
            f"`{hardware.get('wandb', {}).get('mode', '?')}`, run "
            f"`{hardware.get('wandb', {}).get('run_id', '?')}`."
            if hardware.get("wandb", {}).get("enabled")
            else "- Weights & Biases: disabled."
        ),
        "",
        "The CPU smoke run uses fp32 because this CPU reports bf16 support but its "
        "bf16 autograd path was measured to be unusably slow. The A100 "
        "configuration uses bf16. The precision is configuration-driven.",
        "",
        "## Reproducibility checks",
        "",
        "- Split verification: PASS (patient-disjoint, annotation-disjoint, full "
        "coverage, expected counts).",
        "- Tokenization check: all prepared examples fit the maximum length with "
        "no truncation and no gold-evidence loss.",
        "- Unit tests: strict JSON parser/validator, metrics, and split leakage "
        "tests pass.",
        f"- Adapter save/reload check: "
        f"{'PASS' if reload_check.get('passed') else 'not run/failed'}"
        + (
            f" (max abs logit difference "
            f"{reload_check.get('max_abs_difference', '?')})"
            if reload_check
            else ""
        )
        + ".",
        f"- Config SHA-256: `{hardware.get('config_sha256', '?')}`.",
        f"- Final adapter SHA-256: `{hardware.get('final_adapter_sha256', '?')}`.",
        "",
        "## Results",
        "",
        "Decoding is greedy (`do_sample=false`, `num_beams=1`, "
        f"`max_new_tokens={eval_config['max_new_tokens']}`).",
        "",
    ]

    if baseline_val or finetuned_val:
        lines.extend(
            [
                "### Validation",
                "",
                "| Model | N | Parse | Accuracy | Accuracy (parsed) | Macro-F1 | "
                "Evidence P | Evidence R | Evidence F1 | Evidence exact |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        if baseline_val:
            lines.append(_metric_row("Base model", baseline_val))
        if finetuned_val:
            lines.append(_metric_row("Fine-tuned", finetuned_val))
        lines.append("")

        for name, metrics in (
            ("Base model", baseline_val),
            ("Fine-tuned", finetuned_val),
        ):
            if not metrics:
                continue
            rows = _criterion_rows(metrics)
            if rows:
                lines.extend(
                    [
                        f"Per-criterion results, {name}:",
                        "",
                        "| Criterion | N | Macro-F1 |",
                        "| --- | ---: | ---: |",
                        *rows,
                        "",
                    ]
                )
    else:
        lines.extend(
            ["No validation metrics were found. Run the baseline and fine-tuned "
             "validation scripts first.", ""]
        )

    if test_metrics:
        lines.extend(
            [
                "### Test (one-time evaluation)",
                "",
                "| Model | N | Parse | Accuracy | Accuracy (parsed) | Macro-F1 | "
                "Evidence P | Evidence R | Evidence F1 | Evidence exact |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        if test_metrics.get("baseline"):
            lines.append(_metric_row("Base model", test_metrics["baseline"]))
        if test_metrics.get("finetuned"):
            lines.append(_metric_row("Fine-tuned", test_metrics["finetuned"]))
        lines.append("")
    elif not is_smoke:
        lines.extend(
            [
                "### Test",
                "",
                "The locked test split has not been evaluated yet in these "
                "artifacts. Run `evaluate_test.py` once after model selection.",
                "",
            ]
        )

    lines.extend(["## Training summary", ""])
    if curves:
        lines.extend(
            [
                f"- Logged steps: {curves.get('n_steps_logged')}.",
                f"- Train loss: {curves.get('first_train_loss'):.4f} to "
                f"{curves.get('last_train_loss'):.4f}.",
                f"- Best validation loss: {curves.get('best_val_loss')}.",
            ]
        )
    if best_metadata:
        lines.append(
            f"- Best checkpoint: step {best_metadata.get('step')}, "
            f"epoch {best_metadata.get('epoch')}, "
            f"val loss {best_metadata.get('val_loss'):.6f}."
        )
    if final_metadata:
        final_val_loss = final_metadata.get("val_loss")
        final_val_text = (
            f"{final_val_loss:.6f}"
            if isinstance(final_val_loss, (int, float))
            else "n/a"
        )
        lines.append(
            f"- Final checkpoint: step {final_metadata.get('step')}, "
            f"epoch {final_metadata.get('epoch')}, "
            f"val loss {final_val_text}."
        )
    lines.append("")

    lines.extend(
        [
            "## Interpretation and limitations",
            "",
            "- The dataset is small (1,015 annotations, 53 patients), so "
            "validation metrics are noisy and patient-level generalization is the "
            "main uncertainty.",
            "- The target explanations are empty because the dataset has no expert "
            "rationale field. The model is therefore not supervised to produce "
            "clinically meaningful explanations; the explanation field exists to "
            "keep the output contract stable.",
            "- The CPU smoke run uses a 270m model and very few steps. It verifies "
            "the pipeline only. It says nothing about E4B quality.",
            "- The test split is used exactly once. No result in this report may be "
            "used to tune prompts, thresholds, or hyperparameters.",
            "- Evidence IDs are evaluated against the annotated supporting "
            "sentences; retrieval quality is outside this stage.",
            "",
            "## Artifacts",
            "",
            f"- Processed data: `{data_config['processed_dir']}`.",
            f"- Adapters and metadata: `src/trailo/checkpoints/{run_id}`.",
            f"- Predictions and metrics: `src/trailo/results/{run_name}`.",
            f"- Issue log: `src/trailo/logs/issues.md`.",
            "",
            "## Reproduction commands",
            "",
            "```bash",
            "export PYTHONPATH=src",
            "python -m trailo.data.check_splits",
            "python -m trailo.data.prepare --config "
            f"{args.config} --splits train,validation,test --include-test",
            "python -m trailo.data.check_tokenization --config "
            f"{args.config} --split train",
            "python -m trailo.train.finetune --config "
            f"{args.config} --verify-reload",
            "python -m trailo.scripts.run_baseline_val --config "
            f"{args.config} --split validation",
            "python -m trailo.scripts.run_eval_val --config "
            f"{args.config}",
            *(
                []
                if is_smoke
                else [
                    "python -m trailo.scripts.evaluate_test --config "
                    f"{args.config} --confirm-test-once  # run exactly once",
                ]
            ),
            "python -m trailo.scripts.generate_report --config "
            f"{args.config}",
            "```",
            "",
        ]
    )

    report_path = results_dir / "report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines))
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
