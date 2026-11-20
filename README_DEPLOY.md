# NovaCart RAG: deployment guide

```
Customer Streamlit  -> FastAPI /api/chat   -> Chatbot -> Pinecone + Groq
Backoffice Streamlit -> FastAPI /api/kb/*  -> KnowledgeBase -> Pinecone + SQLite/storage
```
The Streamlit apps never hold Pinecone/Groq keys. Only the FastAPI service (Render) does.

Repo layout: `backend/` (API, Render) · `customer/` (storefront, Streamlit) · `backoffice/` (Streamlit).

## 1. Render: the API

Dashboard settings (or use `render.yaml`):

| Setting | Value |
|---|---|
| Root Directory | `backend` |
| Build Command | `pip install -r requirements.txt && python warmup_models.py` |
| Start Command | `uvicorn api:app --host 0.0.0.0 --port $PORT` |
| Health Check Path | `/api/health` |
| Python | `3.12.8` (env var `PYTHON_VERSION`; `backend/.python-version` matches) |

Environment variables: `GROQ_API_KEY`, `PINECONE_API_KEY`, `BO_API_TOKEN` (long random string), plus
`PINECONE_INDEX`, `PINECONE_NAMESPACE`, `LLM_MODEL`, `VISION_MODEL`, `ALLOWED_ORIGINS` (see `.env.example`).
Run **one** instance/worker (chat sessions live in memory).

`warmup_models.py` downloads the embedding model at *build* time, so it is already on disk when the service starts.

### How start-up works (why the old timeout is gone)
* The server opens its port immediately; the chatbot loads in a **background thread**.
* `GET /api/health` is instant and returns `"ready": true|false`.
* `POST /api/chat` waits up to `CHAT_WAIT_SECONDS` (25 s) for the bot; if still loading it returns **503 + Retry-After** instead of hanging.
* The storefront pings `/api/health` on page load, polls until ready, retries on 503, and shows
  "Waking up the assistant..." rather than an error.
* Embeddings run on ONNX (fastembed) instead of PyTorch: same model (all-MiniLM-L6-v2, 384-d) so
  **existing Pinecone vectors keep working**, with far less RAM and a much faster build/boot.

### Persistent storage / plan
The registry (`kb.db`) and uploaded files live on disk.
* **Paid plan (Starter+)**: attach a disk at `/var/data`, set `KB_DB=/var/data/kb.db`, `KB_STORAGE=/var/data/storage`. Data survives restarts and deploys.
* **Free plan**: no disk is available and the filesystem is wiped on every deploy/spin-down. Vectors stay safe in Pinecone, and the chatbot
  falls back to vector-only search when the registry is empty, but the Backoffice document list/versions and CSV/Excel table answers
  disappear after a restart. Don't set `KB_DB` to `/var/data/...` on free (that path doesn't exist); leave the defaults.
  For a free-tier demo you can re-upload after restarts, or move the registry to a hosted Postgres.
* Free instances also **sleep after ~15 min idle** (first request then takes 1-2 min to boot). The wake-up logic above hides most of that;
  an uptime pinger on `/api/health` every 5 minutes (e.g. UptimeRobot) keeps it awake.

## 2. Streamlit Community Cloud: two apps from the same repo

| App | Main file path | Secrets |
|---|---|---|
| Customer | `customer/storefront.py` | `API_URL = "https://YOUR-API.onrender.com"` |
| Backoffice | `backoffice/backoffice.py` | `API_URL`, `BO_API_TOKEN` (same as Render), `BO_PASSWORD` |

* Under *Advanced settings* choose Python **3.12**.
* Each app folder has its own small `requirements.txt` (Streamlit Cloud uses the one next to the main file, so the heavy backend requirements aren't installed).
* The Backoffice **refuses to open without `BO_PASSWORD`** (the app URL is public). Consider also setting the app to *private* in Streamlit's sharing settings.
* No trailing slash in `API_URL`.

## 3. Local run
```
cd backend && pip install -r requirements.txt
cp ../.env.example .env            # fill in keys
uvicorn api:app --reload --port 8000
# new terminals, with API_URL="http://localhost:8000" in .streamlit/secrets.toml:
streamlit run customer/storefront.py --server.port 8502
streamlit run backoffice/backoffice.py --server.port 8501
```

## 4. Test order
1. `https://YOUR-API.onrender.com/api/health` -> `{"status":"ok","ready":true,...}` (may show `"ready":false,"warmup":"starting"` for the first minute after a boot; that's normal).
2. Backoffice: log in, upload a PDF, confirm document/chunk counts increase.
3. Customer: ask something answered by that PDF.
4. Disable the document in Backoffice; the chatbot should stop using it.

## 5. Troubleshooting
| Symptom | Look at |
|---|---|
| Storefront says "Waking up..." for > 2 min | Render logs: `CHATBOT:` / `EMBEDDINGS:` / `PINECONE:` lines show which step is slow or failing. `/api/health` -> `warmup_error` |
| `/api/health` -> `"warmup":"error"` | Usually a missing/invalid `GROQ_API_KEY` or `PINECONE_API_KEY`, or an index with a dimension other than 384 |
| Out-of-memory restarts on Render | Make sure `sentence-transformers`/`torch` are NOT installed (use the provided requirements) |
| "I couldn't find an answer" for everything | Nothing indexed in this `PINECONE_NAMESPACE`, or `MIN_RELEVANCE` too high |
| Backoffice 401 | `BO_API_TOKEN` differs between Render and Streamlit secrets |

## 6. Secrets hygiene
Never commit `.env`, `.streamlit/secrets.toml`, `data/`, `storage/` (all in `.gitignore`). If real keys were ever pushed to GitHub or shared, rotate them
(Groq, Pinecone, `BO_API_TOKEN`).
