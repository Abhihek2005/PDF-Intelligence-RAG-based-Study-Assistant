"""
Step 4: Given a question, retrieve relevant chunks from FAISS and ask
Groq's LLM to answer using only that retrieved context. This is the
core RAG loop: Retrieve -> Augment prompt -> Generate.

WEEK 2 UPGRADE - Reliable citations:
Earlier, we asked the LLM to type out "(Page 4)" itself inside free text.
Problem: the LLM can misremember or skip this, since it's just generating
text -- nothing forces the page number to be correct.

Now, each retrieved chunk gets a short ID (C1, C2, C3...). We ask the LLM
to return JSON: the answer text, plus which chunk IDs it actually used.
Because WE (the code) already know which page each chunk ID maps to,
the final page numbers shown to the user are looked up from our own data,
never typed freely by the LLM. This makes citations guaranteed-correct
instead of "hopefully correct."

WEEK 5 UPGRADE - Reranking + conversational memory:
1. Reranking: FAISS retrieves a wider shortlist (e.g. 15 chunks), then a
   cross-encoder re-scores that shortlist for precision before the LLM
   ever sees it. Slower per-item but only run on a small shortlist, so
   the added latency is small.
2. Conversation history: answer_question now accepts recent turns so
   follow-up questions like "explain that more" can be understood in
   context, without breaking the guaranteed-citation mechanism above.
"""

import os
import json
from dotenv import load_dotenv
from groq import Groq
from ingestion.embed_store import VectorStore
from ingestion.rerank import rerank

# Loads GROQ_API_KEY from a .env file in the project root
load_dotenv()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

SYSTEM_PROMPT = """You are a study assistant answering questions about a textbook/PDF.
Only use the provided numbered context blocks to answer. If the answer isn't in the
context, say so clearly in the answer field and leave used_chunk_ids empty.

If recent conversation turns are provided, use them only to understand what the
user is referring to (e.g. pronouns like "it" or "that"), not as a source of facts.
All facts in your answer must still come from the context blocks.

Respond with ONLY valid JSON in this exact shape, nothing else, no markdown fences:
{"answer": "your answer text here", "used_chunk_ids": ["C1", "C3"]}

used_chunk_ids must list only the chunk IDs whose content you actually relied on
to write the answer."""


def build_prompt(question: str, retrieved_chunks: list, history: list = None) -> str:
    context_blocks = []
    for i, c in enumerate(retrieved_chunks, start=1):
        chunk_id = f"C{i}"
        context_blocks.append(f"[{chunk_id}]\n{c['text']}")
    context = "\n\n---\n\n".join(context_blocks)

    history_block = ""
    if history:
        turns = []
        for turn in history[-3:]:  # only the last few turns, to keep prompts short
            turns.append(f"Q: {turn['question']}\nA: {turn['answer']}")
        history_block = "Recent conversation (for context only):\n" + "\n\n".join(turns) + "\n\n"

    return f"""{history_block}Context blocks from the document:

{context}

Question: {question}

Answer using only the context above. Return the JSON object as instructed."""


def answer_question(
    store: VectorStore,
    question: str,
    top_k: int = 5,
    model: str = "openai/gpt-oss-20b",
    history: list = None,
    use_reranking: bool = True,
):
    # Retrieve a wider shortlist than we need so the reranker has something
    # to actually re-sort. Falls back to top_k directly if reranking is off.
    fetch_k = top_k * 3 if use_reranking else top_k
    retrieved = store.search(question, top_k=fetch_k)

    if not retrieved:
        return {"answer": "No relevant content found in the document.", "sources": []}

    if use_reranking:
        retrieved = rerank(question, retrieved, top_k=top_k)
    else:
        retrieved = retrieved[:top_k]

    # Map chunk IDs (C1, C2...) to their real page/text/score -- this mapping
    # lives in our code, so the LLM can never "get it wrong."
    id_to_chunk = {f"C{i}": c for i, c in enumerate(retrieved, start=1)}

    prompt = build_prompt(question, retrieved, history=history)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,  # low temperature = more factual, less creative drift
        response_format={"type": "json_object"},  # forces valid JSON output
    )

    raw = response.choices[0].message.content

    try:
        parsed = json.loads(raw)
        answer_text = parsed.get("answer", "").strip()
        used_ids = parsed.get("used_chunk_ids", [])
    except (json.JSONDecodeError, AttributeError):
        # Fallback: if the model ever breaks format, don't crash --
        # just show raw text and treat all retrieved chunks as sources.
        answer_text = raw
        used_ids = list(id_to_chunk.keys())

    # Build the final source list from OUR data, using only the chunk IDs
    # the model says it used. Preserves order, drops duplicates.
    seen_pages = set()
    sources = []
    for cid in used_ids:
        chunk = id_to_chunk.get(cid)
        if chunk is None or chunk["page"] in seen_pages:
            continue
        seen_pages.add(chunk["page"])
        sources.append({
            "page": chunk["page"],
            "source": chunk.get("source", "unknown"),
            "score": round(chunk["score"], 3),
            "snippet": chunk["text"][:200].strip() + ("..." if len(chunk["text"]) > 200 else ""),
        })

    # If the model returned no usable IDs, fall back to top retrieved chunks
    # so the user still sees *something* traceable.
    if not sources:
        for c in retrieved[:3]:
            sources.append({
                "page": c["page"],
                "source": c.get("source", "unknown"),
                "score": round(c["score"], 3),
                "snippet": c["text"][:200].strip() + ("..." if len(c["text"]) > 200 else ""),
            })

    return {"answer": answer_text, "sources": sources}


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python api/query.py '<your question>'")
        sys.exit(1)

    question = sys.argv[1]
    store = VectorStore().load("data/index.faiss", "data/meta.pkl")
    result = answer_question(store, question)

    print("\n--- ANSWER ---")
    print(result["answer"])
    print("\n--- SOURCES (guaranteed by code, not the LLM) ---")
    for s in result["sources"]:
        print(f"\nPage {s['page']} (relevance: {s['score']})")
        print(f"  \"{s['snippet']}\"")
