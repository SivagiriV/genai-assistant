"""Central FastAPI service for the GenAI knowledge assistant.

Run locally:
    uvicorn api:app --reload --port 8000

Architecture:
    Customer Streamlit -> /api/chat -> Chatbot -> Pinecone + Groq
    Backoffice Streamlit -> /api/kb/* -> KnowledgeBase -> Pinecone + SQLite/files

Start-up design (important on small / sleeping hosts such as Render):
    * The server starts listening immediately.
    * The chatbot (embedding model + Pinecone + Groq client) loads in a BACKGROUND thread.
    * /api/health answers instantly and reports `ready: true/false`.
    * /api/chat waits a short time for the bot; if it is still loading it returns HTTP 503
      with Retry-After, so clients can retry instead of hanging until a read-timeout.
"""
import hmac
import logging
import os
import re
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

ENV_FILE = Path(__file__).resolve().parent / ".env"
load_dotenv(ENV_FILE, override=True)

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# uvicorn only configures its own loggers; without this our app logs never reach Render's log view.
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"),
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("chat-api")

SHOW_SOURCES = os.getenv("SHOW_SOURCES", "true").lower() == "true"
API_TOKEN = os.getenv("BO_API_TOKEN", "").strip()
CHAT_WAIT_SECONDS = float(os.getenv("CHAT_WAIT_SECONDS", "25"))  # max time a chat request waits for warm-up
log.info("Backoffice API token configured: %s", bool(API_TOKEN))

# ------------------------------------------------------------------ lazy singletons
_bot = None
_kb = None
_registry = None
_bot_lock = threading.Lock()
_kb_lock = threading.Lock()
sessions: dict[str, list] = {}
_sessions_lock = threading.Lock()

_warm_lock = threading.Lock()
_warm_thread: threading.Thread | None = None
_ready = threading.Event()
_warm_state = {"status": "idle", "error": None, "seconds": None}  # idle | starting | ready | error


def get_bot():
    global _bot
    if _bot is None:
        with _bot_lock:                      # only ONE thread may build the bot
            if _bot is None:
                started = time.time()
                log.info("API: creating chatbot instance")
                from chatbot import Chatbot
                _bot = Chatbot()
                log.info("API: chatbot initialised in %.2fs", time.time() - started)
    return _bot


def get_kb():
    global _kb
    if _kb is None:
        with _kb_lock:
            if _kb is None:
                from kb import KnowledgeBase
                _kb = KnowledgeBase()
    return _kb


def get_registry():
    """Registry only (SQLite). Listing/stats must not need the embedding model or Pinecone."""
    global _registry
    if _registry is None:
        from registry import Registry
        _registry = Registry(os.getenv("KB_DB", "data/kb.db"))
    return _registry


def _warm():
    started = time.time()
    try:
        get_bot()
        _warm_state.update(status="ready", error=None, seconds=round(time.time() - started, 1))
    except Exception as exc:
        log.exception("API: warm-up failed")
        _warm_state.update(status="error", error=type(exc).__name__, seconds=round(time.time() - started, 1))
    finally:
        _ready.set()


def start_warmup():
    """Start loading the bot in the background (no-op if already loaded or loading)."""
    global _warm_thread
    with _warm_lock:
        if _bot is not None or (_warm_thread is not None and _warm_thread.is_alive()):
            return
        _ready.clear()
        _warm_state.update(status="starting", error=None, seconds=None)
        _warm_thread = threading.Thread(target=_warm, name="warmup", daemon=True)
        _warm_thread.start()


def ensure_bot():
    """Return the bot, waiting briefly if it is still loading; otherwise HTTP 503 (retryable)."""
    if _bot is not None:
        return _bot
    start_warmup()
    _ready.wait(CHAT_WAIT_SECONDS)
    if _bot is not None:
        return _bot
    if _warm_state["status"] == "error":
        raise HTTPException(503, "The assistant could not start. Please try again shortly.",
                            headers={"Retry-After": "15"})
    raise HTTPException(503, "The assistant is still starting up. Please try again in a few seconds.",
                        headers={"Retry-After": "5"})


@asynccontextmanager
async def lifespan(app: FastAPI):
    start_warmup()      # returns immediately; the port opens right away
    yield


app = FastAPI(title="Support Chat API", version="1.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[x.strip() for x in os.getenv("ALLOWED_ORIGINS", "*").split(",") if x.strip()],
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


def require_admin(x_api_key: str | None):
    if not API_TOKEN:
        raise HTTPException(503, "Backoffice API token is not configured on the server.")
    if not x_api_key or not hmac.compare_digest(x_api_key, API_TOKEN):
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
    started = time.time()
    bot = ensure_bot()                                   # 503 + Retry-After if still warming up
    with _sessions_lock:
        history = list(sessions.get(body.session_id, []))

    try:
        res = bot.ask(body.message.strip(), history)
    except Exception:
        log.exception("API: chat failed")
        raise HTTPException(500, "Sorry, something went wrong. Please try again.")

    history += [("user", body.message), ("assistant", res["answer"])]
    with _sessions_lock:
        sessions[body.session_id] = history[-12:]
        if len(sessions) > 500:
            sessions.pop(next(iter(sessions)))

    answer = res["answer"]
    sources = []
    if SHOW_SOURCES:
        sources = [{"source": s["source"], "url": s.get("url"), "page": s.get("page")}
                   for s in res["sources"]]
    else:
        answer = re.sub(r"\s\[\d+(?:\s*,\s*\d+)*\]", "", answer)

    log.info("API: chat answered in %.2fs", time.time() - started)
    return {"answer": answer, "sources": sources}


@app.post("/api/reset")
def reset(body: ResetIn):
    with _sessions_lock:
        sessions.pop(body.session_id, None)
    return {"ok": True}


@app.get("/api/health")
def health():
    """Cheap and always fast. `ready` tells clients whether /api/chat can answer right now."""
    start_warmup()          # also re-tries after a failed warm-up
    try:
        kb_version = get_registry().kb_version()
    except Exception:
        log.exception("health: registry unavailable")
        kb_version = None
    return {
        "status": "ok",
        "ready": _bot is not None,
        "warmup": _warm_state["status"],
        "warmup_seconds": _warm_state["seconds"],
        "warmup_error": _warm_state["error"],
        "kb_version": kb_version,
    }


# --------------------------- backoffice API ---------------------------
@app.get("/api/kb/stats")
def kb_stats(x_api_key: str | None = Header(default=None)):
    require_admin(x_api_key)
    return get_registry().stats()


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
    return get_registry().list(doc_types=doc_type or None, status=status, search=search or None)


@app.post("/api/kb/chunking")
def kb_chunking(body: ChunkingIn, x_api_key: str | None = Header(default=None)):
    require_admin(x_api_key)
    get_kb().set_chunking(body.size, body.overlap)
    return {"ok": True, "size": body.size, "overlap": body.overlap}


@app.post("/api/kb/files")
async def kb_upload(
    file: UploadFile = File(...),
    note: str = Form(""),
    x_api_key: str | None = Header(default=None),
):
    require_admin(x_api_key)
    if not file.filename:
        raise HTTPException(400, "Filename is required.")

    suffix = Path(file.filename).suffix
    try:
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
    doc = get_registry().get(doc_id)
    if not doc:
        raise HTTPException(404, "Document not found.")
    return get_registry().versions_of(doc["name"])


# The original project also contains a small static customer website/widget.
website_dir = Path(__file__).parent / "website"
if website_dir.exists():
    app.mount("/", StaticFiles(directory=str(website_dir), html=True), name="site")
