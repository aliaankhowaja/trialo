"""Run the inference helper on one train and one validation example."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from trailo.eval.inference import generate_prediction
from trailo.paths import CHECKPOINTS_DIR, repo_path
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
    parser.add_argument("--no-adapter", action="store_true")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    model_config = get_section(config, "model")
    data_config = get_section(config, "data")
    train_config = get_section(config, "train")
    eval_config = get_section(config, "eval")
    run_name = config.get("run_name", "run")
    run_id = f"{run_name}_seed{config['seed']}"
    device = torch.device(train_config["device"])

    processed_dir = repo_path(data_config["processed_dir"])
    label_sets = json.loads((processed_dir / "label_sets.json").read_text())
    examples = {
        split: json.loads((processed_dir / f"{split}.jsonl").read_text().splitlines()[0])
        for split in ("train", "validation")
    }

    tokenizer = load_tokenizer(model_config["model_id"], model_config.get("revision"))
    model = load_model(
        model_config["model_id"],
        model_config.get("revision"),
        device,
        train_config["precision"],
        model_config.get("attn_implementation", "sdpa"),
        bool(model_config.get("allow_cpu_oversize", False)),
    )
    if not args.no_adapter:
        adapter = Path(args.adapter or CHECKPOINTS_DIR / run_id / "best")
        if not adapter.exists():
            raise SystemExit(f"adapter not found: {adapter}")
        model = load_adapter_model(model, str(adapter))
        model.to(device)
    model.eval()
    model.config.use_cache = True

    decoding_config = {
        "max_new_tokens": int(eval_config["max_new_tokens"]),
        "do_sample": False,
        "num_beams": 1,
        "label_sets": label_sets,
    }
    for split, example in examples.items():
        result = generate_prediction(model, tokenizer, example, decoding_config)
        print(
            f"{split}: annotation_id={example['annotation_id']} "
            f"valid={result['valid']} errors={result['errors']}"
        )
        print(f"  raw_text={result['raw_text'][:300]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
