"""Central FastAPI service for the GenAI knowledge assistant.

Run locally:
    uvicorn api:app --reload --port 8000

Architecture:
    Customer Streamlit -> /api/chat -> Chatbot -> Pinecone + Groq
    Backoffice Streamlit -> /api/kb/* -> KnowledgeBase -> Pinecone + SQLite/files
"""
import logging
import os
import re
import tempfile
from pathlib import Path
from dotenv import load_dotenv
ENV_FILE = Path(__file__).resolve().parent / ".env"
load_dotenv(ENV_FILE, override=True)
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

log = logging.getLogger("chat-api")
SHOW_SOURCES = os.getenv("SHOW_SOURCES", "true").lower() == "true"
API_TOKEN = os.getenv("BO_API_TOKEN", "").strip()
log.info("Backoffice API token configured: %s", bool(API_TOKEN))
app = FastAPI(title="Support Chat API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[x.strip() for x in os.getenv("ALLOWED_ORIGINS", "*").split(",") if x.strip()],
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Lazy singletons avoid loading Pinecone/embedding/LLM resources until they are actually needed.
_bot = None
_kb = None
sessions: dict[str, list] = {}


def get_bot():
    global _bot
    if _bot is None:
        from chatbot import Chatbot
        _bot = Chatbot()
    return _bot


def get_kb():
    global _kb
    if _kb is None:
        from kb import KnowledgeBase
        _kb = KnowledgeBase()
    return _kb


def require_admin(x_api_key: str | None):
    if not API_TOKEN:
        raise HTTPException(503, "Backoffice API token is not configured on the server.")
    if not x_api_key or x_api_key != API_TOKEN:
        raise HTTPException(401, "Invalid API key.")


class ChatIn(BaseModel):
    session_id: str = Field(min_length=1, max_length=64)
    message: str = Field(min_length=1, max_length=1000)


class ResetIn(BaseModel):
    session_id: str = Field(min_length=1, max_length=64)


class UrlIn(BaseModel):
    url: str = Field(min_length=5, max_length=2000)
    max_pages: int = Field(default=10, ge=1, le=50)
    note: str = Field(default="", max_length=500)


class ChunkingIn(BaseModel):
    size: int = Field(default=800, ge=300, le=1500)
    overlap: int = Field(default=120, ge=0, le=300)


@app.post("/api/chat")
def chat(body: ChatIn):
    history = sessions.get(body.session_id, [])
    try:
        res = get_bot().ask(body.message.strip(), history)
    except Exception:
        log.exception("chat failed")
        raise HTTPException(500, "Sorry, something went wrong. Please try again.")

    history += [("user", body.message), ("assistant", res["answer"])]
    sessions[body.session_id] = history[-12:]
    if len(sessions) > 500:
        sessions.pop(next(iter(sessions)))

    answer = res["answer"]
    sources = []
    if SHOW_SOURCES:
        sources = [
            {"source": s["source"], "url": s.get("url"), "page": s.get("page")}
            for s in res["sources"]
        ]
    else:
        answer = re.sub(r"\s*\[\d+(?:\s*,\s*\d+)*\]", "", answer)
    return {"answer": answer, "sources": sources}


@app.post("/api/reset")
def reset(body: ResetIn):
    sessions.pop(body.session_id, None)
    return {"ok": True}


@app.get("/api/health")
def health():
    # Health must remain cheap: don't initialize the LLM/Pinecone just to answer health checks.
    from registry import Registry
    return {"status": "ok", "kb_version": Registry(os.getenv("KB_DB", "data/kb.db")).kb_version()}


# --------------------------- backoffice API ---------------------------
@app.get("/api/kb/stats")
def kb_stats(x_api_key: str | None = Header(default=None)):
    require_admin(x_api_key)
    return get_kb().stats()


@app.get("/api/kb/docs")
def kb_docs(
    status: str | None = None,
    search: str | None = None,
    doc_type: list[str] | None = None,
    x_api_key: str | None = Header(default=None),
):
    require_admin(x_api_key)
    if status not in (None, "active", "archived"):
        raise HTTPException(400, "status must be active or archived")
    return get_kb().list_docs(doc_types=doc_type or None, status=status, search=search or None)


@app.post("/api/kb/chunking")
def kb_chunking(body: ChunkingIn, x_api_key: str | None = Header(default=None)):
    require_admin(x_api_key)
    get_kb().set_chunking(body.size, body.overlap)
    return {"ok": True, "size": body.size, "overlap": body.overlap}


@app.post("/api/kb/files")
async def kb_upload(
    file: UploadFile = File(...),
    note: str = "",
    x_api_key: str | None = Header(default=None),
):
    require_admin(x_api_key)
    if not file.filename:
        raise HTTPException(400, "Filename is required.")

    suffix = Path(file.filename).suffix
    try:
        from kb import KnowledgeBase
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            temp_path = Path(tmp.name)
            while chunk := await file.read(1024 * 1024):
                tmp.write(chunk)
        try:
            result = get_kb().add_file(str(temp_path), file.filename, note)
            return result
        finally:
            temp_path.unlink(missing_ok=True)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception:
        log.exception("file ingestion failed")
        raise HTTPException(500, "Could not process the uploaded file.")


@app.post("/api/kb/url")
def kb_url(body: UrlIn, x_api_key: str | None = Header(default=None)):
    require_admin(x_api_key)
    try:
        return get_kb().add_url(body.url, body.max_pages, body.note)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception:
        log.exception("website ingestion failed")
        raise HTTPException(500, "Could not crawl and index that website.")


@app.post("/api/kb/docs/{doc_id}/disable")
def kb_disable(doc_id: int, x_api_key: str | None = Header(default=None)):
    require_admin(x_api_key)
    get_kb().disable(doc_id)
    return {"ok": True}


@app.post("/api/kb/docs/{doc_id}/restore")
def kb_restore(doc_id: int, x_api_key: str | None = Header(default=None)):
    require_admin(x_api_key)
    try:
        get_kb().restore(doc_id)
        return {"ok": True}
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception:
        log.exception("restore failed")
        raise HTTPException(500, "Could not restore this version.")


@app.delete("/api/kb/docs/{doc_id}")
def kb_delete(doc_id: int, x_api_key: str | None = Header(default=None)):
    require_admin(x_api_key)
    get_kb().delete(doc_id)
    return {"ok": True}


@app.get("/api/kb/docs/{doc_id}/versions")
def kb_versions(doc_id: int, x_api_key: str | None = Header(default=None)):
    require_admin(x_api_key)
    doc = get_kb().registry.get(doc_id)
    if not doc:
        raise HTTPException(404, "Document not found.")
    return get_kb().registry.versions_of(doc["name"])


# The original project also contains a small static customer website/widget.
website_dir = Path(__file__).parent / "website"
if website_dir.exists():
    app.mount("/", StaticFiles(directory=str(website_dir), html=True), name="site")
