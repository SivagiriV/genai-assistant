# NovaCart RAG deployment

## Architecture

Customer Streamlit -> FastAPI `/api/chat` -> Chatbot -> Pinecone + Groq

Backoffice Streamlit -> FastAPI `/api/kb/*` -> KnowledgeBase -> Pinecone + SQLite/storage

The customer app never contains Pinecone or Groq credentials. The FastAPI service owns the RAG layer.

## Local setup

1. Create a Python 3.12 virtual environment.
2. `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and fill in real keys.
4. Start API: `uvicorn api:app --reload --port 8000`
5. In another terminal start customer: `streamlit run storefront.py --server.port 8502`
6. In another terminal start backoffice: `streamlit run backoffice.py --server.port 8501`
7. For local Streamlit secrets, copy `.streamlit/secrets.example.toml` to `.streamlit/secrets.toml` and set `API_URL = "http://localhost:8000"`.

## GitHub

Do NOT commit `.env`, `.streamlit/secrets.toml`, `data/kb.db`, `storage/`, or `__pycache__/`.

## Render API

Build: `pip install -r requirements.txt`

Start: `uvicorn api:app --host 0.0.0.0 --port $PORT`

Set the RAG/API secrets in Render environment variables. Attach a persistent disk at `/var/data`, and set `KB_DB=/var/data/kb.db` and `KB_STORAGE=/var/data/storage`.

## Streamlit Community Cloud

Create two apps from the same GitHub repository:

- Customer: entrypoint `storefront.py`
- Backoffice: entrypoint `backoffice.py`

Customer secrets:
`API_URL = "https://YOUR-API.onrender.com"`

Backoffice secrets:
`API_URL = "https://YOUR-API.onrender.com"`
`BO_API_TOKEN = "same value as Render"`
`BO_PASSWORD = "your password"`

## Test order

1. Open `https://YOUR-API.onrender.com/api/health`.
2. Open the Backoffice and upload a PDF.
3. Confirm the document/chunk counts increase.
4. Open Customer and ask a question whose answer is in the PDF.
5. Confirm the answer is returned through FastAPI.
6. Disable the document in Backoffice and verify the chatbot no longer uses it.
