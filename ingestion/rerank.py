"""
Reranking: after FAISS gives us the top candidates (fast but approximate),
a cross-encoder re-scores each (question, chunk) pair together for much
higher precision. This is slower per-pair, so we only run it on a small
shortlist (e.g. top 15 from FAISS), not the whole document.

Why this helps: FAISS embeds the question and each chunk SEPARATELY, then
compares vectors. A cross-encoder reads the question and chunk TOGETHER in
one pass, so it can judge relevance far more precisely -- catching cases
where two chunks have similar embeddings but only one actually answers
the question.
"""

from sentence_transformers import CrossEncoder

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_reranker = None


def get_reranker():
    """Lazy-load the cross-encoder model only when reranking is actually used."""
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANK_MODEL)
    return _reranker


def rerank(question: str, candidates: list, top_k: int = 5):
    """
    candidates: list of dicts from VectorStore.search() (page, text, source, score)
    Returns the same dicts, re-sorted by cross-encoder relevance, trimmed to top_k.
    Adds a "rerank_score" field to each result.
    """
    if not candidates:
        return []

    model = get_reranker()
    pairs = [[question, c["text"]] for c in candidates]
    scores = model.predict(pairs)

    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)

    reranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
    return reranked[:top_k]
