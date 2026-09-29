"""
Week 3: Baseline vs RAG comparison.

This is what proves your RAG system is actually better than "just asking
an LLM directly" -- the core justification for building a RAG system at all.

BASELINE = ask the LLM the question directly, no PDF context, just its
           own general knowledge (or it might refuse / hallucinate specifics)
RAG      = your existing pipeline: retrieve relevant chunks from the PDF,
           then answer grounded in that retrieved context

We run both on the same set of questions and save results to a JSON file
so you can build a comparison table for your report.

Usage:
    python -m eval.compare_baseline "data/your_textbook.pdf" questions.txt
"""

import os
import sys
import json
import time
from dotenv import load_dotenv
from groq import Groq, RateLimitError
from ingestion.embed_store import VectorStore
from api.query import answer_question

load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

BASELINE_MODEL = "openai/gpt-oss-20b"


def call_with_retry(fn, max_retries: int = 5):
    """
    Free-tier Groq accounts have a strict tokens-per-minute limit.
    On a RateLimitError, wait and retry instead of crashing the whole run.
    """
    for attempt in range(max_retries):
        try:
            return fn()
        except RateLimitError as e:
            wait_seconds = 15 * (attempt + 1)  # 15s, 30s, 45s... backs off progressively
            print(f"  Rate limit hit, waiting {wait_seconds}s before retry ({attempt + 1}/{max_retries})...")
            time.sleep(wait_seconds)
    # last attempt, let it raise if it still fails
    return fn()


def get_baseline_answer(question: str) -> str:
    """Ask the LLM directly, no retrieved context -- just its own knowledge."""
    def _call():
        response = client.chat.completions.create(
            model=BASELINE_MODEL,
            messages=[
                {"role": "system", "content": "Answer the question concisely based on your general knowledge."},
                {"role": "user", "content": question},
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()

    return call_with_retry(_call)


def run_comparison(index_path: str, meta_path: str, questions: list, output_path: str):
    store = VectorStore().load(index_path, meta_path)
    results = []

    for i, q in enumerate(questions, start=1):
        print(f"[{i}/{len(questions)}] {q}")

        baseline_answer = get_baseline_answer(q)
        time.sleep(2)  # gentler pacing to avoid tripping the per-minute token limit

        rag_result = call_with_retry(lambda: answer_question(store, q))
        time.sleep(2)

        results.append({
            "question": q,
            "baseline_answer": baseline_answer,
            "rag_answer": rag_result["answer"],
            "rag_sources": [{"page": s["page"], "score": s["score"]} for s in rag_result["sources"]],
        })

        # Save after every question -- if something crashes later, earlier work isn't lost.
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    return results


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python -m eval.compare_baseline <pdf_path> <questions_file.txt>")
        print("questions_file.txt should have one question per line.")
        sys.exit(1)

    pdf_path = sys.argv[1]
    questions_file = sys.argv[2]

    with open(questions_file, "r", encoding="utf-8") as f:
        questions = [line.strip() for line in f if line.strip()]

    print(f"Running {len(questions)} questions through baseline vs RAG...\n")

    output_path = "eval/comparison_results.json"
    os.makedirs("eval", exist_ok=True)

    results = run_comparison("data/index.faiss", "data/meta.pkl", questions, output_path)

    print(f"\nDone. Results saved to {output_path}")
    print("Open that file to read baseline vs RAG answers side by side.")
