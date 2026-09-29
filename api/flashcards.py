"""
Auto-generate practice flashcards (short question/answer pairs) from
the currently indexed document(s) -- useful for quick revision.

Unlike the main query pipeline (which retrieves chunks relevant to a
SPECIFIC question), this samples chunks spread across the whole document
so the flashcards cover a broad range of topics, not just one area.
"""

import os
import json
from dotenv import load_dotenv
from groq import Groq
from ingestion.embed_store import VectorStore

load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

FLASHCARD_MODEL = "openai/gpt-oss-20b"

SYSTEM_PROMPT = """You generate exam-style flashcards from study material.
Given several excerpts from a textbook/notes document, create concise
question-and-answer flashcards a student could use for quick revision.

Respond with ONLY valid JSON in this exact shape, no markdown fences:
{"flashcards": [{"question": "...", "answer": "...", "page": 4}, ...]}

Rules:
- Each flashcard's "page" must be the page number of the excerpt it came from.
- Keep answers short (1-3 sentences) -- these are for quick recall, not essays.
- Cover a spread of different topics from the excerpts, don't cluster on one idea.
- Skip excerpts that are too fragmentary to make a clear question from."""


def sample_chunks(store: VectorStore, max_chunks: int = 10):
    """
    Picks chunks spread evenly across the document(s) so flashcards cover
    a broad range of topics instead of clustering in one area.
    """
    all_chunks = store.chunks_meta
    if not all_chunks:
        return []

    if len(all_chunks) <= max_chunks:
        return all_chunks

    step = len(all_chunks) / max_chunks
    sampled = [all_chunks[int(i * step)] for i in range(max_chunks)]
    return sampled


def generate_flashcards(store: VectorStore, num_cards: int = 8, model: str = FLASHCARD_MODEL):
    chunks = sample_chunks(store, max_chunks=max(num_cards, 6))

    if not chunks:
        return {"flashcards": []}

    excerpt_blocks = []
    for c in chunks:
        excerpt_blocks.append(f"[Page {c['page']}]\n{c['text'][:600]}")
    excerpts = "\n\n---\n\n".join(excerpt_blocks)

    prompt = f"""Excerpts from the document:

{excerpts}

Generate up to {num_cards} flashcards following the rules. Return the JSON object."""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.4,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content
    try:
        parsed = json.loads(raw)
        cards = parsed.get("flashcards", [])
    except (json.JSONDecodeError, AttributeError):
        cards = []

    return {"flashcards": cards}


if __name__ == "__main__":
    store = VectorStore().load("data/index.faiss", "data/meta.pkl")
    result = generate_flashcards(store)

    print(f"\nGenerated {len(result['flashcards'])} flashcards:\n")
    for i, card in enumerate(result["flashcards"], start=1):
        print(f"{i}. Q: {card['question']}")
        print(f"   A: {card['answer']}  (page {card.get('page', '?')})\n")
