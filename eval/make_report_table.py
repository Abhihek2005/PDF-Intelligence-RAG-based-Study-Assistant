"""
Converts eval/comparison_results.json into a readable Markdown table
you can paste straight into your project report.

Usage:
    python -m eval.make_report_table
"""

import json
import os

INPUT_PATH = "eval/comparison_results.json"
OUTPUT_PATH = "eval/comparison_table.md"


def make_table(results: list) -> str:
    lines = [
        "# Baseline vs RAG — Comparison Results",
        "",
        "| # | Question | Baseline Answer (no PDF context) | RAG Answer (from PDF) | Source Pages |",
        "|---|----------|-----------------------------------|-------------------------|--------------|",
    ]

    for i, r in enumerate(results, start=1):
        question = r["question"].replace("|", "\\|")
        baseline = r["baseline_answer"].replace("|", "\\|").replace("\n", " ")
        rag = r["rag_answer"].replace("|", "\\|").replace("\n", " ")
        pages = ", ".join(str(s["page"]) for s in r["rag_sources"]) or "—"

        # Keep table cells readable -- trim very long answers, full text stays in the JSON
        if len(baseline) > 300:
            baseline = baseline[:300] + "..."
        if len(rag) > 300:
            rag = rag[:300] + "..."

        lines.append(f"| {i} | {question} | {baseline} | {rag} | {pages} |")

    return "\n".join(lines)


if __name__ == "__main__":
    if not os.path.exists(INPUT_PATH):
        print(f"Couldn't find {INPUT_PATH}. Run eval.compare_baseline first.")
        exit(1)

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        results = json.load(f)

    table_md = make_table(results)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(table_md)

    print(f"Table written to {OUTPUT_PATH}")
    print("Open it in VS Code (or paste into Word) to use in your report.")
