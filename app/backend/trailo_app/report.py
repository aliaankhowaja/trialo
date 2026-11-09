"""Readable eligibility screening report generation (Markdown / HTML / PDF)."""

from __future__ import annotations

import datetime
from typing import Any

LABEL_ORDER = [
    "included",
    "excluded",
    "not included",
    "not applicable",
    "not enough information",
    "unparsable",
]


def _result_label(result: dict[str, Any]) -> str:
    return result.get("label") or ("unparsable" if not result.get("valid") else "unknown")


def build_markdown(
    *,
    meta: dict[str, Any],
    results: list[dict[str, Any]],
    trial_title: str,
    patient_id: str,
    sentences: list[str],
    benchmark_snapshot: dict[str, Any] | None = None,
) -> str:
    now = datetime.datetime.now().replace(microsecond=0).isoformat()
    disclaimer = meta["disclaimer"]

    lines: list[str] = []
    lines.append("# Eligibility Screening Report")
    lines.append("")
    lines.append(f"> **{disclaimer}**")
    lines.append("")
    lines.append("## Run summary")
    lines.append("")
    lines.append(f"- Patient record: `{patient_id or 'unnamed'}` ({meta['n_sentences']} sentences)")
    lines.append(f"- Trial: {trial_title}")
    lines.append(f"- Model preset: `{meta['model_preset']}` - {meta['base_model_id']}")
    lines.append(f"- Checkpoint (LoRA adapter): `{meta['checkpoint_dir']}`")
    lines.append(f"- Device / decoding: {meta['device']} / `{meta['decoding_impl']}`")
    lines.append(f"- Generated at: {now}")
    lines.append("")

    lines.append("## Aggregate verdict")
    lines.append("")
    counts: dict[str, int] = {}
    for result in results:
        counts[_result_label(result)] = counts.get(_result_label(result), 0) + 1
    lines.append("| Verdict | Criteria |")
    lines.append("| --- | ---: |")
    for label in [l for l in LABEL_ORDER if l in counts] + sorted(
        l for l in counts if l not in LABEL_ORDER
    ):
        lines.append(f"| {label} | {counts[label]} |")
    lines.append("")
    lines.append(
        f"{counts.get('included', 0)} of {len(results)} inclusion-relevant "
        "verdicts read as eligible under the model's annotation; verify every "
        "criterion against the quoted evidence below."
    )
    lines.append("")

    lines.append("## Criterion-by-criterion results")
    lines.append("")
    for rank, result in enumerate(results, start=1):
        lines.append(
            f"### {rank}. [{result['criterion_type'].capitalize()}] {result['criterion_text']}"
        )
        lines.append("")
        label = _result_label(result)
        lines.append(f"- **Verdict:** `{label}`")
        if result.get("warnings"):
            lines.append(f"- **Warnings:** {', '.join(result['warnings'])}")
        if result.get("errors"):
            lines.append(f"- **Parse errors:** {', '.join(result['errors'])}")
        evidence = result.get("evidence") or []
        if evidence:
            lines.append("- **Supporting evidence (exact sentences):**")
            for span in evidence:
                lines.append(f'  - Sentence {span["sentence_id"]}: "{span["text"]}"')
        else:
            lines.append("- **Supporting evidence:** none cited by the model")
        explanation = (result.get("explanation") or "").strip()
        if explanation:
            lines.append(f"- **Model explanation:** {explanation}")
        lines.append(
            f"- *Model output: {result['generated_tokens']} tokens, "
            f"{result['wall_seconds']:.2f} s, stopped_on_eos={result['stopped_on_eos']}*"
        )
        lines.append("")

    if benchmark_snapshot:
        lines.append("## Decode benchmark snapshot")
        lines.append("")
        lines.append("| Decoding mechanism | Wall time (s) | Tokens | Tokens/s | Stream agreement |")
        lines.append("| --- | ---: | ---: | ---: | --- |")
        agreement = benchmark_snapshot.get("agreement", [])
        all_equal = all(entry.get("all_equal") for entry in agreement) if agreement else None
        agreement_text = (
            "identical token streams" if all_equal else "divergent (see app)"
        )
        for row in benchmark_snapshot.get("rows", []):
            lines.append(
                f"| `{row['impl']}` | {row['wall_seconds']} | {row['generated_tokens']} "
                f"| {row['tokens_per_second']} | "
                f"{'ok' if row['ok'] else row['error']} |"
            )
        lines.append(f"- Agreement across paths: {agreement_text} on the benchmarked subset")
        lines.append("")

    lines.append("---")
    lines.append(f"**{disclaimer}**")
    lines.append("")
    lines.append(
        f"*Sentences were auto-numbered ({len(sentences)} sentences). Gold "
        "references shown in the app (when screening a bundled sample record) "
        "come from expert annotations and are NOT part of the model output.*"
    )
    return "\n".join(lines)


def markdown_to_html(markdown_text: str) -> str:
    import markdown as _markdown

    body = _markdown.markdown(
        markdown_text,
        extensions=["tables", "fenced_code"],
        output_format="html5",
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Eligibility Screening Report</title>
<style>
  body {{ font-family: Georgia, 'Times New Roman', serif; max-width: 44em;
         margin: 2em auto; padding: 0 1em; color: #1a1a1a; line-height: 1.55; }}
  table {{ border-collapse: collapse; margin: 1em 0; }}
  th, td {{ border: 1px solid #bbb; padding: 4px 10px; text-align: left; }}
  blockquote {{ border-left: 4px solid #b45309; background: #fffbeb;
                padding: 0.6em 1em; margin: 1.2em 0; }}
  h1 {{ border-bottom: 2px solid #1e3a8a; padding-bottom: 0.3em; }}
  h3 {{ margin-bottom: 0.2em; }}
  code {{ background: #f1f5f9; padding: 1px 4px; border-radius: 3px; }}
  hr {{ margin: 2em 0; border: none; border-top: 1px solid #ccc; }}
</style>
</head>
<body>
{body}
</body>
</html>"""


def markdown_to_pdf(markdown_text: str) -> bytes:
    from weasyprint import HTML

    html = markdown_to_html(markdown_text)
    return HTML(string=html).write_pdf()
