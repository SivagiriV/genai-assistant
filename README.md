# Support Platform: Back Office + Customer Chat Widget

Two separate apps that share one knowledge base.

```
 COMPANY SIDE                                   CUSTOMER SIDE
 +-----------------------+                     +-----------------------------+
 | Back Office (Streamlit)|                    | Website + chat bubble       |
 | upload / crawl / list  |                    | website/index.html          |
 | version / disable      |                    | website/widget.js           |
 +-----------+-----------+                     +--------------+--------------+
             | writes                                          | POST /api/chat
             v                                                 v
   Pinecone (vectors)  <------ reads ------  FastAPI (api.py) -> Chatbot (RAG + pandas + Groq)
   SQLite registry (data/kb.db)  <--- reads, refreshes when kb_version changes
   storage/ (original files, kept for every version)
```

## Run (three terminals)
```
cd backend && pip install -r requirements.txt
copy ..\.env.example .env        # fill in GROQ_API_KEY, PINECONE_API_KEY, BO_API_TOKEN
uvicorn api:app --reload --port 8000                          # API

# set API_URL / BO_API_TOKEN / BO_PASSWORD in .streamlit/secrets.toml, then:
streamlit run backoffice/backoffice.py                        # company side  -> http://localhost:8501
streamlit run customer/storefront.py --server.port 8502       # customer side -> http://localhost:8502
```
See `README_DEPLOY.md` for Render + Streamlit Cloud deployment.

### Customer side: two interchangeable options
- **Streamlit storefront** (`storefront.py`): dummy store + floating chat bubble, all in Python. Simplest to run and demo.
- **FastAPI + JavaScript widget** (`api.py`, `website/`): `uvicorn api:app --port 8000`. The widget is one
  `<script>` tag that can be embedded in any real website, which is closer to how production chat widgets work.
Both use the same `Chatbot` engine and the same knowledge base.
Use a fresh `PINECONE_NAMESPACE` (default `kb`); vectors from the earlier single-app version live in
the `default` namespace and are not used here.

## Back office features
- Upload PDF, Word (.docx), Excel (.xlsx), CSV, TXT/MD and images (read with a Groq vision model)
- Add a website URL: crawls the page and same-domain links (max pages configurable)
- Knowledge base table with filters: document type, status (active / archived), name search
- "In use" column shows exactly what the chatbot currently answers from
- Versioning: re-upload the same filename -> new version goes live, old version is archived
  (vectors removed, original kept). Restore any version, disable, download original, or delete.
- Identical re-uploads are detected by SHA-256 and skipped
- Optional password (BO_PASSWORD)

## Customer side features
- Dummy store website with a floating chat bubble (bottom right); the widget is one `<script>` tag
- Multi-turn memory per browser tab, quick-question chips, typing indicator, source list
- Hybrid search (BM25 + vectors), confidence gate ("I couldn't find that"), role-based prompt
- Counts / row lookups over CSV and Excel are computed exactly with pandas
- Picks up back-office changes automatically (no restart)

## Evaluate
`python evaluate.py` indexes `data/sample_docs/` into a separate `eval` namespace and scores retrieval,
faithfulness and out-of-scope refusals using `data/eval_set.json`. Test questions: `docs/test_questions.md`.

## Known limits (good talking points)
- Chat sessions are kept in server memory (use Redis for multiple servers)
- BM25 index is rebuilt in memory when the knowledge base changes (fine for thousands of chunks)
- Scanned PDFs have no text layer: upload their pages as images
- Image/OCR text comes from a vision LLM, so spot-check it
- Each CSV/Excel row becomes one chunk; very large tables take a while to embed
- Only crawl websites you own or have permission to copy
