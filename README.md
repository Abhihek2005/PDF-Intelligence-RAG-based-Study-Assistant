# PDF Intelligence — RAG System (Week 1 Starter)

Minimal working pipeline: PDF → chunks → embeddings → FAISS index → question → cited answer.

## Setup (run on your own machine — needs internet for model download)

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Get a free Groq API key from https://console.groq.com (sign up → API Keys → Create API Key).

Then create a file named `.env` in the project root (copy `.env.example` and rename it) and paste your key in:

```
GROQ_API_KEY=gsk_your_actual_key_here
```

That's it — the code loads this automatically, no terminal export needed. Just make sure `.env` is never committed to GitHub (already covered by `.gitignore` below).

## Test the pipeline step by step

```bash
# 1. Extract + chunk a PDF (put a textbook PDF in data/ first)
python -m ingestion.chunk data/your_textbook.pdf

# 2. Build embeddings + FAISS index (downloads all-MiniLM-L6-v2 model on first run, ~90MB)
python -m ingestion.embed_store data/your_textbook.pdf

# 3. Ask a question against the built index
python -m api.query "What is a stack data structure?"
```

## Run as an API (for Week 4 frontend to call)

```bash
uvicorn api.main:app --reload
```
Then open http://127.0.0.1:8000/docs — you'll get a Swagger UI to test /upload and /query directly in the browser.

## Project structure

```
pdf-intelligence/
├── ingestion/
│   ├── extract.py       # PDF → page-wise text
│   ├── chunk.py          # page text → overlapping chunks (keeps page numbers)
│   └── embed_store.py     # chunks → embeddings → FAISS index (save/load)
├── api/
│   ├── query.py          # retrieval + Groq LLM call → cited answer
│   └── main.py            # FastAPI app (/upload, /query endpoints)
├── data/                  # sample PDFs, saved index files go here
└── requirements.txt
```

## What's already working (Week 1 target)
- Page-aware PDF extraction ✅
- Overlapping chunking (500 words, 100 overlap — tune these numbers later) ✅
- FAISS semantic search ✅
- Groq LLM answer generation with page citations ✅
- FastAPI endpoints ✅

## What's NOT done yet (later weeks)
- Cross-encoder reranking layer (Week 3 — improves retrieval precision)
- Evaluation/comparison against baseline (Week 3)
- React frontend (Week 4)
- Multi-PDF support — current code handles one PDF at a time

## Notes on the numbers you'll want to tune
- `chunk_size=500, overlap=100` in `chunk.py` — good starting point, but for math-heavy
  pages with formulas, smaller chunks (e.g. 300) sometimes retrieve more precisely. Test both.
- `top_k=5` in `query.py` — how many chunks get sent to the LLM. More = more context but
  slower/costlier. This is a good knob to show in your evaluation section (Week 3).
