"""
FastAPI app: backend + the frontend UI.

Endpoints:
1. POST /upload      -> accepts a PDF, adds it to the (possibly multi-doc) index
2. POST /query       -> accepts a question, returns an answer with citations
                         (uses cross-encoder reranking + recent conversation history)
3. GET  /flashcards  -> generates revision flashcards from the indexed document(s)
4. GET  /documents   -> lists which PDFs are currently indexed
5. POST /reset       -> clears the index and starts fresh
6. GET  /            -> serves the frontend (static/index.html)

Run with: uvicorn api.main:app --reload
Then open http://127.0.0.1:8000 in your browser for the UI,
or http://127.0.0.1:8000/docs for the raw API (Swagger UI).
"""

import os
import shutil
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from ingestion.chunk import chunk_pdf
from ingestion.embed_store import VectorStore
from api.query import answer_question
from api.flashcards import generate_flashcards

app = FastAPI(title="PDF Intelligence - RAG API")

# Allows the frontend HTML (even if opened directly as a file, or served
# from a different port during development) to call these endpoints.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "data/uploads"
INDEX_PATH = "data/index.faiss"
META_PATH = "data/meta.pkl"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# In-memory state for this local, single-user server session.
_store_cache = {"store": None}
_conversation_history = []  # list of {"question": ..., "answer": ...}
MAX_HISTORY_TURNS = 6


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Adds a PDF to the index. Multiple uploads accumulate into ONE combined
    index (multi-PDF mode) -- questions can then draw on content from any
    of the uploaded documents. Use /reset to start over with a clean slate.
    """
    save_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    chunks = chunk_pdf(save_path)
    for c in chunks:
        c["source"] = file.filename

    store = _store_cache["store"]
    if store is None:
        store = VectorStore().build(chunks)
    else:
        store.add(chunks)

    store.save(INDEX_PATH, META_PATH)
    _store_cache["store"] = store

    return {
        "message": f"Indexed '{file.filename}' successfully.",
        "num_chunks": len(chunks),
        "documents": store.list_sources(),
    }


@app.post("/reset")
async def reset_index():
    """Clears the current index and conversation history to start fresh."""
    global _conversation_history
    _store_cache["store"] = None
    _conversation_history = []

    for path in (INDEX_PATH, META_PATH):
        if os.path.exists(path):
            os.remove(path)

    return {"message": "Index and conversation history cleared."}


@app.get("/documents")
async def list_documents():
    store = _store_cache["store"]
    if store is None:
        return {"documents": []}
    return {"documents": store.list_sources()}


@app.post("/query")
async def query_pdf(question: str = Form(...), use_history: bool = Form(True)):
    store = _store_cache["store"]
    if store is None:
        # fall back to loading from disk if server was restarted
        store = VectorStore().load(INDEX_PATH, META_PATH)
        _store_cache["store"] = store

    history = _conversation_history if use_history else None
    result = answer_question(store, question, history=history)

    # Keep a short rolling history for follow-up questions like "explain more".
    _conversation_history.append({"question": question, "answer": result["answer"]})
    if len(_conversation_history) > MAX_HISTORY_TURNS:
        del _conversation_history[0]

    return result


@app.get("/flashcards")
async def get_flashcards(count: int = 8):
    store = _store_cache["store"]
    if store is None:
        store = VectorStore().load(INDEX_PATH, META_PATH)
        _store_cache["store"] = store

    return generate_flashcards(store, num_cards=count)


@app.get("/")
async def root():
    return FileResponse("static/index.html")


# Serves anything else in static/ (not strictly needed yet, but handy
# if you add CSS/JS files separately later)
app.mount("/static", StaticFiles(directory="static"), name="static")
