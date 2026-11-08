"""Build the Task 2 criterion/evidence analysis report from existing eval outputs.

Reads saved prediction JSONL files (no inference), runs the full analysis in
``trailo.eval.analysis``, and writes:

- ``results/<run_name>/task2/task2_analysis.json``   full numbers per model/split
- ``results/<run_name>/task2/evidence_*_<split>.jsonl``  Task 2 schema records
- ``results/<run_name>/task2/report.md``            human-readable report
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trailo.eval.analysis import (
    EVIDENCE_POLICY,
    analyze_records,
    enriched_schema,
)
from trailo.paths import RESULTS_DIR, repo_path
from trailo.train.config import get_section, load_config


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def fmt(value: float) -> str:
    return f"{value:.3f}"


def matrix_table(view: dict[str, Any]) -> str:
    """Markdown confusion matrix with gold rows and predicted columns."""
    labels = view["labels"]
    header = "| gold \\ pred | " + " | ".join(labels) + " |"
    separator = "|" + "---|" * (len(labels) + 1)
    rows = []
    for gold, row in zip(labels, view["row_normalized"]):
        cells = []
        for gold_i, value in enumerate(row):
            count = view["counts"][labels.index(gold)][gold_i]
            cells.append(f"{fmt(value)} ({count})")
        rows.append(f"| **{gold}** | " + " | ".join(cells) + " |")
    return "\n".join([header, separator, *rows])


def per_class_table(view: dict[str, Any]) -> str:
    header = "| label | support | predicted | precision | recall | F1 |"
    separator = "|---|---:|---:|---:|---:|---:|"
    rows = []
    for label, scores in sorted(view["per_class"].items()):
        rows.append(
            f"| {label} | {scores['support']} | {scores['predicted']} "
            f"| {fmt(scores['precision'])} | {fmt(scores['recall'])} | {fmt(scores['f1'])} |"
        )
    return "\n".join([header, separator, *rows])


def analysis_table(all_analyses: dict[tuple[str, str], dict[str, Any]]) -> str:
    header = "| model | split | n | parse | accuracy | macro-F1 (declared) | macro-F1 (observed) |"
    separator = "|---|---|---:|---:|---:|---:|---:|"
    rows = [
        f"| {model} | {split} | {analysis['n_examples']} "
        f"| {fmt(analysis['parse_success_rate'])} "
        f"| {fmt(analysis['classification']['overall']['accuracy'])} "
        f"| {fmt(analysis['classification']['overall']['macro_f1'])} "
        f"| {fmt(analysis['classification']['overall']['macro_f1_observed'])} |"
        for (model, split), analysis in sorted(all_analyses.items())
    ]
    return "\n".join([header, separator, *rows])


def evidence_table(all_analyses: dict[tuple[str, str], dict[str, Any]]) -> str:
    header = (
        "| model | split | view | n | precision | recall | F1 | "
        "exact-match rate / consistency |"
    )
    separator = "|---|---|---|---:|---:|---:|---:|---:|"
    rows = []
    for key, analysis in sorted(all_analyses.items()):
        model, split = key
        evidence = analysis["evidence"]
        pooled = evidence["pooled"]
        rows.append(
            f"| {model} | {split} | pooled | {evidence['pooled']['n_records']} "
            f"| {fmt(pooled['precision'])} | {fmt(pooled['recall'])} | {fmt(pooled['f1'])} "
            f"| {fmt(pooled['exact_match_rate'])} |"
        )
        gp = evidence["gold_present"]
        rows.append(
            f"| {model} | {split} | gold-present | {gp['n_records']} "
            f"| {fmt(gp['precision'])} | {fmt(gp['recall'])} | {fmt(gp['f1'])} "
            f"| {fmt(gp['exact_match_rate'])} |"
        )
        pr = evidence["per_record"]
        rows.append(
            f"| {model} | {split} | per-record macro | {pr['n_records']} "
            f"| — | — | {fmt(pr['macro_f1'])} | — |"
        )
        ng = evidence["no_gold_evidence"]
        rows.append(
            f"| {model} | {split} | no-gold consistency | {ng['n_records']} "
            f"| — | — | — "
            f"| empty-pred rate {fmt(ng['empty_prediction_rate'])} |"
        )
    return "\n".join([header, separator, *rows])


def example_rows(entries: list[dict[str, Any]], with_sentences: bool = False) -> str:
    header = (
        "| patient | ann | gold | predicted | cited ids | criterion | explanation |"
    )
    separator = "|---|---|---|---|---|---|---|"
    rows = [header, separator]
    for entry in entries:
        criterion = (entry.get("criterion") or entry.get("criterion_text") or "")
        cited = ", ".join(str(i) for i in entry.get("predicted_evidence_ids", [])) or "—"
        explanation = (entry.get("explanation") or "").replace("|", "\\|")[:200] or "—"
        excerpt = criterion.replace("|", "\\|")[:160] or "—"
        cited_text = ""
        if with_sentences and entry.get("sentences"):
            cited_text = " ".join(
                item["text"] or f"[invalid id {item['id']}]"
                for item in (
                    {
                        "id": i,
                        "text": entry["sentences"].get(i),
                    }
                    for i in entry.get("predicted_evidence_ids", [])
                )
            ).replace("|", "\\|")
            if cited_text:
                excerpt = f"{excerpt}<br>**cited text:** {cited_text[:300]}"
        rows.append(
            f"| {entry.get('patient_id')} | {entry.get('annotation_id')} "
            f"| {entry.get('gold_label')} | {entry.get('predicted_label')} | {cited} "
            f"| {excerpt} | {explanation} |"
        )
    return "\n".join(rows)


def build_report(
    analyses: dict[tuple[str, str], dict[str, Any]],
    run_name: str,
) -> str:
    parts = [
        "# Task 2: Criterion-Level Eligibility and Evidence Verification",
        "",
        f"Generated `{datetime.now(timezone.utc).isoformat()}` for run `{run_name}` "
        "from saved prediction files; no model was loaded and no inference was run "
        "for this report.",
        "",
        "## Label policy",
        "",
        "The full four-way expert label set is preserved per criterion type: "
        "`included`/`not applicable`/`not enough information`/`not included` for "
        "inclusion and `excluded`/`not applicable`/`not enough information`/"
        "`not excluded` for exclusion. `not applicable` and `not enough "
        "information` are never collapsed into yes/no outcomes.",
        "",
        "### Macro-F1 convention",
        "",
        "Two macro-F1 numbers are reported everywhere. **Declared** averages "
        "over the full expert label universe, so a class the model never "
        "predicts (F1 = 0, e.g. `excluded` in the fine-tuned run) still "
        "penalizes the score. **Observed** averages only over labels with gold "
        "support or predictions, exactly matching the headline `macro_f1` in "
        "eval/metrics.py so the numbers here stay comparable with the "
        "validation/test metric files.",
        "",
        "## Evidence policy",
        "",
        EVIDENCE_POLICY,
        "",
        "## 1. Criterion classification (overall)",
        "",
        analysis_table(analyses),
        "",
        "## 2. Confusion matrices",
        "",
    ]
    for key in sorted(analyses):
        model, split = key
        analysis = analyses[key]
        for view_name, view in (
            ("overall", analysis["classification"]["overall"]),
            ("inclusion criteria", analysis["classification"]["inclusion"]),
            ("exclusion criteria", analysis["classification"]["exclusion"]),
        ):
            parts.append(f"### {model} — {split} — {view_name}")
            parts.append("")
            parts.append(
                f"`accuracy={fmt(view['accuracy'])}` "
                f"`macro-F1 (declared)={fmt(view['macro_f1'])}` "
                f"`macro-F1 (observed)={fmt(view['macro_f1_observed'])}` "
                f"`n={view['n_records']}`"
            )
            parts.append("")
            parts.append(matrix_table(view))
            parts.append("")
            parts.append("Per-class precision/recall/F1:")
            parts.append("")
            parts.append(per_class_table(view))
            parts.append("")
        parts.append(
            f"##### {model} — {split} — evidence retrieval"
        )
        parts.append("")
        parts.append(evidence_rows(analysis))
        parts.append("")
    parts += [
        "## 3. Evidence retrieval",
        "",
        "Pooled = all parsed records; gold-present = records with non-empty gold "
        "evidence only; per-record macro = mean per-record F1 over gold-present "
        "records. No-gold consistency is the rate at which predictions also cite "
        "no evidence on records with empty gold evidence; it is not retrieval "
        "quality and earns no credit.",
        "",
        evidence_table(analyses),
        "",
        "## 4. Contradiction cases",
        "",
        "Heuristic review flags, ranked by severity. Flagged lists show the "
        "most severe cases for manual inspection.",
        "",
    ]
    for key in sorted(analyses):
        model, split = key
        contradictions = analyses[key]["contradictions"]
        parts.append(f"### {model} — {split}")
        parts.append("")
        counts = contradictions["flag_counts"] or {}
        count_rows = "| flag | count |\n|---|---:|\n" + "\n".join(
            f"| {flag} | {count} |" for flag, count in sorted(counts.items())
        )
        if not counts:
            count_rows = "_No contradiction flags raised._"
        parts.append(count_rows)
        parts.append("")
        parts.append(
            f"Conclusive predictions (included/excluded/not included/not "
            f"excluded): {contradictions['conclusive_predictions']}, without "
            f"cited evidence: {contradictions['conclusive_without_evidence']} "
            f"(rate {fmt(contradictions['conclusive_without_evidence_rate'])}). "
            "Absence of cited evidence is not proof of eligibility or "
            "ineligibility."
        )
        parts.append("")
        worst = contradictions["flagged"][:10]
        if worst:
            parts.append("Most severe flagged cases:")
            parts.append("")
            parts.append(flag_table(worst))
            parts.append("")
    parts += ["## 5. Representative errors", ""]
    for key in sorted(analyses):
        model, split = key
        errors = analyses[key]["errors"]
        parts.append(f"### {model} — {split}")
        for gallery in errors["confusion_pairs"][:5]:
            parts.append(
                f"* gold `{gallery['gold']}` predicted "
                f"`{gallery['predicted']}` ({gallery['count']} times):"
            )
            parts.append("")
            parts.append(example_rows(gallery["examples"], with_sentences=True))
            parts.append("")
        parts += [
            "Missing information (`not enough information`) handling:",
            "",
            missing_information_rows(errors["missing_information"]),
            "",
            "Not-applicable handling:",
            "",
            not_applicable_rows(errors["not_applicable"]),
            "",
        ]
    return "\n".join(parts).rstrip() + "\n"


_missing_information_template = (
    "gold `not enough information`: {gold}/{total} gold records were "
    "mispredicted (`over` rows predicted NEI when gold is a conclusive label: "
    "{over_count})."
)


def missing_information_rows(errors: dict[str, Any]) -> str:
    summary = _missing_information_template.format(
        gold=errors["gold_nei_mispredicted_count"],
        total=errors["gold_nei_total"],
        over_count=errors["predicted_nei_when_not_gold_count"],
    )
    examples = errors["gold_nei_mispredicted"][:3] + errors["predicted_nei_when_not_gold"][:2]
    return f"{summary}\n\n{example_rows(examples, with_sentences=False)}"


def not_applicable_rows(errors: dict[str, Any]) -> str:
    summary = (
        f"gold `not applicable`: {errors['mispredicted_count']}/"
        f"{errors['gold_na_total']} mispredicted."
    )
    examples = errors["mispredicted"][:3]
    return f"{summary}\n\n{example_rows(examples, with_sentences=False)}"


def flag_table(entries: list[dict[str, Any]]) -> str:
    header = "| flag | patient | ann | gold | predicted | cited ids | detail |"
    separator = "|---|---|---|---|---|---|---|"
    rows = [header, separator]
    for entry in entries:
        cited = ", ".join(str(i) for i in entry.get("predicted_evidence_ids", [])) or "—"
        detail = entry["detail"].replace("|", "\\|")
        rows.append(
            f"| {entry['flag']} | {entry['patient_id']} | {entry['annotation_id']} "
            f"| {entry['gold_label']} | {entry['predicted_label']} | {cited} "
            f"| {detail} |"
        )
    return "\n".join(rows)


def evidence_rows(analysis: dict[str, Any]) -> str:
    view = analysis["evidence"]
    rows = "| view | n | precision | recall | F1 | exact match |\n"
    rows += "|---|---:|---:|---:|---:|---:|\n"
    for name in ("pooled", "gold_present"):
        v = view[name]
        rows += (
            f"| {name} | {v['n_records']} | {fmt(v['precision'])} "
            f"| {fmt(v['recall'])} | {fmt(v['f1'])} | {fmt(v['exact_match_rate'])} |\n"
        )
    consistency = view["no_gold_evidence"]
    rows += (
        f"| no-gold consistency | {consistency['n_records']} | — | — | — "
        f"| empty-pred rate {fmt(consistency['empty_prediction_rate'])} |\n"
    )
    per_record = view["per_record"]
    rows += (
        f"| per-record macro F1 (gold-present) | {per_record['n_records']} "
        f"| — | — | {fmt(per_record['macro_f1'])} | — |\n"
    )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--models", default="baseline,finetuned")
    parser.add_argument("--splits", default="validation,test")
    parser.add_argument("--top-pairs", type=int, default=5)
    parser.add_argument("--examples-per-pair", type=int, default=2)
    parser.add_argument("--max-examples", type=int, default=5)
    parser.add_argument("--flagged", type=int, default=10)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    run_name = config.get("run_name", "run")
    data_config = get_section(config, "data")
    processed_dir = repo_path(data_config["processed_dir"])
    label_sets = json.loads((processed_dir / "label_sets.json").read_text())
    results_dir = RESULTS_DIR / run_name / "task2"
    results_dir.mkdir(parents=True, exist_ok=True)

    analyses: dict[tuple[str, str], dict[str, Any]] = {}
    for model in [m.strip() for m in args.models.split(",") if m.strip()]:
        for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
            predictions_path = (
                RESULTS_DIR / run_name / f"{model}_{split}_predictions.jsonl"
            )
            if not predictions_path.exists():
                raise SystemExit(f"predictions not found: {predictions_path}")
            predictions = load_jsonl(predictions_path)
            split_records = load_jsonl(processed_dir / f"{split}.jsonl")
            analyses[(model, split)] = analyze_records(
                predictions,
                split_records,
                label_sets,
                top_pairs=args.top_pairs,
                examples_per_pair=args.examples_per_pair,
                max_examples=args.max_examples,
            )
            en_schema = enriched_schema(predictions, split_records)
            en_path = results_dir / f"evidence_{model}_{split}.jsonl"
            with en_path.open("w") as handle:
                for entry in en_schema:
                    handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
            # keep only the top-k flagged entries in the persisted JSON
            contradictions = analyses[(model, split)]["contradictions"]
            contradictions["flagged"] = contradictions["flagged"][: args.flagged]
            print(f"{model}/{split}: analyzed {len(predictions)} records -> {en_path}")

    analysis_path = results_dir / "task2_analysis.json"
    analysis_path.write_text(
        json.dumps(
            {
                "run_name": run_name,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "models": sorted({model for model, _ in analyses}),
                "analysis": {
                    f"{model}/{split}": analysis
                    for (model, split), analysis in sorted(analyses.items())
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    report_path = results_dir / "report.md"
    report_path.write_text(build_report(analyses, run_name))
    print(f"Analysis JSON: {analysis_path}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
