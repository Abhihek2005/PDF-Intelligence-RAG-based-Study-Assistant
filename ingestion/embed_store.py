"""
Step 3: Convert chunks to embeddings and store them in a FAISS index
for fast similarity search.

We use a small, fast sentence-transformer model (all-MiniLM-L6-v2) —
good enough accuracy for a student project, and fast on CPU (no GPU needed).

MULTI-PDF UPGRADE: each chunk now carries a "source" field (the filename
it came from). build() creates a fresh index; add() appends more chunks
(from another PDF) into the SAME index, so questions can be answered
using content pooled across every uploaded document.
"""

import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"


class VectorStore:
    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME)
        self.index = None
        self.chunks_meta = []  # keeps page/source/text info aligned with index order

    def _embed(self, texts: list):
        embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        embeddings = embeddings.astype("float32")
        faiss.normalize_L2(embeddings)
        return embeddings

    def build(self, chunks: list):
        """
        chunks: list of dicts like {"chunk_id":.., "page":.., "text":.., "source": ..}
        Builds the FAISS index from scratch (replaces anything already loaded).
        """
        texts = [c["text"] for c in chunks]
        embeddings = self._embed(texts)

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # inner product = cosine sim (since normalized)
        self.index.add(embeddings)

        self.chunks_meta = chunks
        return self

    def add(self, chunks: list):
        """
        Appends chunks from another document into the existing index,
        instead of replacing it. Used for multi-PDF mode.
        """
        if self.index is None:
            return self.build(chunks)

        texts = [c["text"] for c in chunks]
        embeddings = self._embed(texts)
        self.index.add(embeddings)
        self.chunks_meta.extend(chunks)
        return self

    def list_sources(self):
        """Returns the distinct document filenames currently indexed."""
        return sorted(set(c.get("source", "unknown") for c in self.chunks_meta))

    def search(self, query: str, top_k: int = 5):
        """
        Returns top_k most relevant chunks with their similarity scores.
        """
        if self.index is None or self.index.ntotal == 0:
            return []

        q_emb = self.model.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(q_emb)
        top_k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(q_emb, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            meta = self.chunks_meta[idx]
            results.append({
                "page": meta["page"],
                "text": meta["text"],
                "source": meta.get("source", "unknown"),
                "score": float(score)
            })
        return results

    def save(self, index_path: str, meta_path: str):
        faiss.write_index(self.index, index_path)
        with open(meta_path, "wb") as f:
            pickle.dump(self.chunks_meta, f)

    def load(self, index_path: str, meta_path: str):
        self.index = faiss.read_index(index_path)
        with open(meta_path, "rb") as f:
            self.chunks_meta = pickle.load(f)
        return self


if __name__ == "__main__":
    import sys
    from ingestion.chunk import chunk_pdf

    if len(sys.argv) < 2:
        print("Usage: python embed_store.py <path_to_pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    print("Chunking PDF...")
    chunks = chunk_pdf(pdf_path)
    for c in chunks:
        c["source"] = pdf_path.split("/")[-1].split("\\")[-1]
    print(f"{len(chunks)} chunks created. Building embeddings + FAISS index...")

    store = VectorStore().build(chunks)
    store.save("data/index.faiss", "data/meta.pkl")
    print("Index built and saved to data/index.faiss and data/meta.pkl")

    # quick sanity test
    query = "What is this document about?"
    results = store.search(query, top_k=3)
    print(f"\n--- Test query: '{query}' ---")
    for r in results:
        print(f"[page {r['page']}, score {r['score']:.3f}] {r['text'][:150]}...")
