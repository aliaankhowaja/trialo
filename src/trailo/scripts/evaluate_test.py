"""One-time final evaluation on the locked test split."""

from __future__ import annotations

import argparse
import gc
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from trailo.eval.run import run_records
from trailo.paths import CHECKPOINTS_DIR, RESULTS_DIR, repo_path
from trailo.train.config import get_section, load_config
from trailo.train.modeling import (
    load_adapter_model,
    load_model,
    load_tokenizer,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--subset-size", type=int, default=None)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--confirm-test-once", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    run_name = config.get("run_name", "run")
    results_dir = RESULTS_DIR / run_name
    marker = results_dir / "TEST_EVALUATED.json"
    results_dir.mkdir(parents=True, exist_ok=True)
    if marker.exists() and not args.force:
        raise SystemExit(
            f"test split was already evaluated once ({marker}); refusing to rerun"
        )
    if not args.confirm_test_once:
        raise SystemExit(
            "pass --confirm-test-once to acknowledge the one-time test evaluation"
        )

    model_config = get_section(config, "model")
    data_config = get_section(config, "data")
    train_config = get_section(config, "train")
    eval_config = get_section(config, "eval")
    model_id = args.model_id or model_config["model_id"]
    revision = args.revision or model_config.get("revision")
    device = torch.device(train_config["device"])
    adapter = Path(
        args.adapter or CHECKPOINTS_DIR / f"{run_name}_seed{config['seed']}" / "best"
    )
    if not adapter.exists():
        raise SystemExit(f"adapter not found: {adapter}")

    processed_dir = repo_path(data_config["processed_dir"])
    label_sets = json.loads((processed_dir / "label_sets.json").read_text())
    records = [
        json.loads(line)
        for line in (processed_dir / "test.jsonl").read_text().splitlines()
        if line
    ]
    subset_size = (
        args.subset_size
        if args.subset_size is not None
        else eval_config.get("subset_size")
    )
    decoding_config = {
        "max_new_tokens": int(eval_config["max_new_tokens"]),
        "do_sample": False,
        "num_beams": 1,
        "label_sets": label_sets,
    }
    metadata_base = {
        "run_name": run_name,
        "model_id": model_id,
        "model_revision": revision,
        "split": "test",
        "is_smoke": "smoke" in run_name,
    }
    tokenizer = load_tokenizer(model_id, revision)

    base_model = load_model(
        model_id,
        revision,
        device,
        train_config["precision"],
        model_config.get("attn_implementation", "sdpa"),
        bool(model_config.get("allow_cpu_oversize", False)),
    )
    base_metrics = run_records(
        base_model,
        tokenizer,
        records,
        decoding_config,
        output_path=results_dir / "baseline_test_predictions.jsonl",
        metrics_path=results_dir / "baseline_test_metrics.json",
        metadata={**metadata_base, "adapter": None},
        limit=subset_size,
    )
    del base_model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    base_model = load_model(
        model_id,
        revision,
        device,
        train_config["precision"],
        model_config.get("attn_implementation", "sdpa"),
        bool(model_config.get("allow_cpu_oversize", False)),
    )
    fine_tuned = load_adapter_model(base_model, str(adapter))
    fine_tuned.to(device)
    fine_tuned.eval()
    fine_tuned.config.use_cache = True
    fine_tuned_metrics = run_records(
        fine_tuned,
        tokenizer,
        records,
        decoding_config,
        output_path=results_dir / "finetuned_test_predictions.jsonl",
        metrics_path=results_dir / "finetuned_test_metrics.json",
        metadata={**metadata_base, "adapter": str(adapter)},
        limit=subset_size,
    )

    combined = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_name": run_name,
        "model_id": model_id,
        "model_revision": revision,
        "adapter": str(adapter),
        "decoding": {
            "max_new_tokens": int(eval_config["max_new_tokens"]),
            "do_sample": False,
            "num_beams": 1,
        },
        "baseline": base_metrics,
        "finetuned": fine_tuned_metrics,
    }
    (results_dir / "test_metrics.json").write_text(
        json.dumps(combined, indent=2, sort_keys=True) + "\n"
    )
    marker.write_text(
        json.dumps(
            {
                "evaluated_at": datetime.now(timezone.utc).isoformat(),
                "adapter": str(adapter),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(
        f"Test baseline parse={base_metrics['parse_success_rate']:.3f} "
        f"accuracy={base_metrics['label_accuracy']:.3f} "
        f"macro_f1={base_metrics['macro_f1']:.3f}"
    )
    print(
        f"Test fine-tuned parse={fine_tuned_metrics['parse_success_rate']:.3f} "
        f"accuracy={fine_tuned_metrics['label_accuracy']:.3f} "
        f"macro_f1={fine_tuned_metrics['macro_f1']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
