"""
Step 1: Extract text from PDF, page by page.
We keep page numbers attached to every chunk of text so that later,
when we answer a question, we can cite exactly which page it came from.
"""

import fitz  # pymupdf


def extract_pages(pdf_path: str):
    """
    Returns a list of dicts: [{"page": 1, "text": "..."}, {"page": 2, "text": "..."}, ...]
    """
    doc = fitz.open(pdf_path)
    pages = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        if text.strip():  # skip blank pages
            pages.append({"page": page_num + 1, "text": text})
    doc.close()
    return pages


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python extract.py <path_to_pdf>")
        sys.exit(1)

    pages = extract_pages(sys.argv[1])
    print(f"Extracted {len(pages)} pages.")
    print("--- Preview of page 1 ---")
    print(pages[0]["text"][:500] if pages else "No text found.")
