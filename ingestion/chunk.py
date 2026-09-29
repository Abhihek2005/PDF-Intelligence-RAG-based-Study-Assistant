"""
Step 2: Split each page's text into overlapping chunks.
Why overlap? So a sentence that gets cut at a chunk boundary doesn't lose
context. Why chunk at all? Because embedding a whole page is too coarse
for precise retrieval, especially for math/formula-heavy textbook pages.
"""

from ingestion.extract import extract_pages


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100):
    """
    Splits text into word-based chunks with overlap.
    chunk_size and overlap are in words, not characters (easier to reason about).
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap  # slide window forward with overlap
    return chunks


def chunk_pdf(pdf_path: str, chunk_size: int = 500, overlap: int = 100):
    """
    Returns a list of dicts, each a chunk with its source page number:
    [{"page": 1, "chunk_id": 0, "text": "..."}, ...]
    """
    pages = extract_pages(pdf_path)
    all_chunks = []
    chunk_id = 0
    for page in pages:
        page_chunks = chunk_text(page["text"], chunk_size, overlap)
        for c in page_chunks:
            all_chunks.append({
                "chunk_id": chunk_id,
                "page": page["page"],
                "text": c
            })
            chunk_id += 1
    return all_chunks


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python chunk.py <path_to_pdf>")
        sys.exit(1)

    chunks = chunk_pdf(sys.argv[1])
    print(f"Created {len(chunks)} chunks.")
    print("--- First chunk ---")
    print(chunks[0])
