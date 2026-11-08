"""Run the untouched base model on validation examples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from trailo.eval.run import run_records
from trailo.paths import RESULTS_DIR, repo_path
from trailo.train.config import get_section, load_config
from trailo.train.modeling import load_model, load_tokenizer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--subset-size", type=int, default=None)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--metrics", default=None)
    args = parser.parse_args(argv)

    if args.split == "test":
        raise SystemExit(
            "run_baseline_val.py only evaluates validation; the test baseline is "
            "produced once by evaluate_test.py after the final adapter is selected"
        )

    config = load_config(args.config)
    model_config = get_section(config, "model")
    data_config = get_section(config, "data")
    train_config = get_section(config, "train")
    eval_config = get_section(config, "eval")
    run_name = config.get("run_name", "run")
    model_id = args.model_id or model_config["model_id"]
    revision = args.revision or model_config.get("revision")
    device = torch.device(train_config["device"])

    processed_dir = repo_path(data_config["processed_dir"])
    label_sets = json.loads((processed_dir / "label_sets.json").read_text())
    records = [
        json.loads(line)
        for line in (processed_dir / f"{args.split}.jsonl").read_text().splitlines()
        if line
    ]
    subset_size = args.subset_size
    if subset_size is None:
        subset_size = eval_config.get("subset_size")

    tokenizer = load_tokenizer(model_id, revision)
    model = load_model(
        model_id,
        revision,
        device,
        train_config["precision"],
        model_config.get("attn_implementation", "sdpa"),
        bool(model_config.get("allow_cpu_oversize", False)),
    )
    decoding_config = {
        "max_new_tokens": int(eval_config["max_new_tokens"]),
        "do_sample": False,
        "num_beams": 1,
        "label_sets": label_sets,
    }

    results_dir = RESULTS_DIR / run_name
    output = Path(
        args.output
        or results_dir / f"baseline_{args.split}_predictions.jsonl"
    )
    metrics_path = Path(
        args.metrics or results_dir / f"baseline_{args.split}_metrics.json"
    )
    metrics = run_records(
        model,
        tokenizer,
        records,
        decoding_config,
        output_path=output,
        metrics_path=metrics_path,
        metadata={
            "run_name": run_name,
            "model_id": model_id,
            "model_revision": revision,
            "split": args.split,
            "adapter": None,
            "is_smoke": "smoke" in run_name,
        },
        limit=subset_size,
    )
    print(
        f"Baseline {args.split}: parse={metrics['parse_success_rate']:.3f} "
        f"accuracy={metrics['label_accuracy']:.3f} macro_f1={metrics['macro_f1']:.3f}"
    )
    print(f"Raw predictions: {output}")
    print(f"Metrics: {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
