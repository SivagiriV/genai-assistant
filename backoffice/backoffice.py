"""Company-side Back Office (Streamlit).

The Backoffice is intentionally API-only: it does not import Pinecone or KnowledgeBase.
All knowledge-base operations go through the central FastAPI service.
"""
import hmac
import os
import threading
import time

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Support Back Office", page_icon="🗂️", layout="wide")


def setting(name, default=""):
    try:
        return str(st.secrets.get(name, os.getenv(name, default)))
    except Exception:
        return os.getenv(name, default)


API_URL = setting("API_URL").rstrip("/")
API_TOKEN = setting("BO_API_TOKEN")


def api(method, path, **kwargs):
    headers = kwargs.pop("headers", {})
    if API_TOKEN:
        headers["X-API-Key"] = API_TOKEN
    r = requests.request(method, f"{API_URL}{path}", headers=headers, timeout=(15, 180), **kwargs)
    if not r.ok:
        try:
            detail = r.json().get("detail", "API request failed")
        except Exception:
            detail = r.text or "API request failed"
        raise RuntimeError(detail)
    return r.json()


if not API_URL or not API_TOKEN:
    st.error("API_URL and BO_API_TOKEN must be configured in Streamlit secrets.")
    st.stop()


@st.cache_resource(ttl=900, show_spinner=False)
def wake_api():
    """Ping the API as soon as the page opens so a sleeping Render instance boots while the
    user is still typing the password."""
    def _ping():
        try:
            requests.get(f"{API_URL}/api/health", timeout=120)
        except Exception:
            pass
    threading.Thread(target=_ping, daemon=True).start()
    return True


def wait_for_api(max_wait=150):
    """Block (with a spinner) until the API answers /api/health. Cached per session."""
    if st.session_state.get("api_ready"):
        return True
    with st.spinner("Connecting to the API... if it was asleep this can take up to a minute."):
        deadline = time.time() + max_wait
        while time.time() < deadline:
            try:
                if requests.get(f"{API_URL}/api/health", timeout=10).ok:
                    st.session_state.api_ready = True
                    return True
            except requests.RequestException:
                pass
            time.sleep(3)
    return False


def show_df(df, **kwargs):
    """st.dataframe across Streamlit versions (use_container_width was replaced by width='stretch')."""
    try:
        return st.dataframe(df, width="stretch", **kwargs)
    except Exception:
        return st.dataframe(df, use_container_width=True, **kwargs)


wake_api()

# ---------------- login (required: the app is public on Streamlit Cloud) ----------------
PASSWORD = setting("BO_PASSWORD")
if not PASSWORD:
    st.error("BO_PASSWORD is not set. Add it to this app's Streamlit secrets; "
             "the back office will not open without a password.")
    st.stop()
if not st.session_state.get("auth"):
    st.title("🔐 Support Back Office")
    with st.form("login"):
        entered = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log in", type="primary")
    if submitted:
        if hmac.compare_digest(entered.encode(), PASSWORD.encode()):
            st.session_state.auth = True
            st.rerun()
        time.sleep(1)                       # slow down password guessing
        st.error("Wrong password.")
    st.stop()

if not wait_for_api():
    st.error("The API did not respond in time. Check that the Render service is running, then reload this page.")
    st.stop()


def human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def describe(r):
    if r["status"] == "added":
        return f"✅ **{r['name']}** added as v{r['version']} ({r['chunks']} chunks)"
    if r["status"] == "updated":
        return f"🔄 **{r['name']}**: v{r['version']} is now live, v{r['version'] - 1} was archived ({r['chunks']} chunks)"
    return f"⏭ **{r['name']}**: identical to the active version (v{r['version']}), nothing changed"


with st.sidebar:
    st.header("Indexing settings")
    size = st.slider("Chunk size", 300, 1500, 800, 100)
    overlap = st.slider("Chunk overlap", 0, 300, 120, 20)
    if st.button("Apply chunking settings"):
        try:
            api("POST", "/api/kb/chunking", json={"size": size, "overlap": overlap})
            st.success("Settings updated for future content.")
        except Exception as exc:
            st.error(str(exc))
    st.caption("Applies to content added from now on.")
    st.divider()
    try:
        kb_stats = api("GET", "/api/kb/stats")
        st.markdown(f"**API:** `{API_URL}`  \n**Active docs:** `{kb_stats['active_docs']}`  \n**Active chunks:** `{kb_stats['active_chunks']}`")
    except Exception as exc:
        st.error(f"API unavailable: {exc}")
    if st.button("Log out"):
        st.session_state.auth = False
        st.rerun()

st.title("🗂️ Support Back Office")
tab_add, tab_kb = st.tabs(["➕ Add content", "📚 Knowledge base"])

with tab_add:
    for line in st.session_state.pop("last_results", []):
        st.markdown(line)

    st.subheader("Upload files")
    st.caption("PDF, Word, Excel, CSV, text and images. Uploading a file with the same name creates a new version and retires the old one.")
    up_key = st.session_state.setdefault("up_key", 0)
    files = st.file_uploader("Files", accept_multiple_files=True, key=f"up{up_key}",
                             type=["pdf", "docx", "xlsx", "csv", "txt", "md", "png", "jpg", "jpeg", "webp"])
    note = st.text_input("Version note (optional)", placeholder="e.g. Return window changed to 60 days")
    if st.button("Add files to knowledge base", type="primary", disabled=not files):
        results = []
        with st.status("Processing...", expanded=True) as status:
            for f in files:
                st.write(f"Reading and embedding {f.name} ...")
                try:
                    files_payload = {"file": (f.name, f.getvalue(), f.type or "application/octet-stream")}
                    r = requests.post(f"{API_URL}/api/kb/files", files=files_payload, data={"note": note},
                                      headers={"X-API-Key": API_TOKEN}, timeout=(15, 600))
                    if not r.ok:
                        try:
                            detail = r.json().get("detail", "Upload failed")
                        except ValueError:
                            detail = r.text or "Upload failed"
                        raise RuntimeError(detail)
                    results.append(describe(r.json()))
                except Exception as exc:
                    results.append(f"❌ **{f.name}**: {exc}")
            status.update(label="Done", state="complete")
        st.session_state.last_results = results
        st.session_state.up_key += 1
        st.rerun()

    st.divider()
    st.subheader("Add a website")
    st.caption("Crawls the page and linked pages on the same site. Only add sites you own or have permission to copy.")
    c1, c2 = st.columns([3, 1])
    url = c1.text_input("Website URL", placeholder="https://www.example.com/help")
    pages = c2.number_input("Max pages", 1, 50, 10)
    if st.button("Crawl and add", disabled=not url):
        with st.spinner("Crawling and embedding..."):
            try:
                result = api("POST", "/api/kb/url", json={"url": url, "max_pages": int(pages), "note": note})
                st.session_state.last_results = [describe(result)]
            except Exception as exc:
                st.session_state.last_results = [f"❌ **{url}**: {exc}"]
        st.rerun()

with tab_kb:
    try:
        s = api("GET", "/api/kb/stats")
        m1, m2, m3 = st.columns(3)
        m1.metric("Documents in use", s["active_docs"])
        m2.metric("Searchable chunks in use", s["active_chunks"])
        m3.metric("Archived versions", s["archived"])
    except Exception as exc:
        st.error(f"Could not load knowledge-base stats: {exc}")
        st.stop()

    f1, f2, f3 = st.columns([2, 2, 2])
    types = f1.multiselect("Document type", ["PDF", "Word", "Excel", "CSV", "Text", "Image", "Website"], placeholder="All types")
    status_label = f2.radio("Status", ["Active (in use)", "Archived", "All"], horizontal=True)
    search = f3.text_input("Search by name")
    status = {"Active (in use)": "active", "Archived": "archived", "All": None}[status_label]

    params = []
    if status:
        params.append(("status", status))
    if search:
        params.append(("search", search))
    for t in types:
        params.append(("doc_type", t))
    try:
        docs = api("GET", "/api/kb/docs", params=params)
    except Exception as exc:
        st.error(f"Could not load documents: {exc}")
        st.stop()

    if not docs:
        st.info("No documents match these filters.")
    else:
        table = pd.DataFrame([{
            "In use": "✅ Active" if d["status"] == "active" else "⏸ Archived",
            "Name": d["name"], "Type": d["doc_type"], "Version": f"v{d['version']}",
            "Chunks": d["chunk_count"], "Size": human(d["size_bytes"]),
            "Added": d["created_at"], "Note": d["note"]} for d in docs])
        event = show_df(table, hide_index=True, on_select="rerun", selection_mode="single-row", key="kbtable")
        st.caption("Select a row to manage it.")
        sel = event.selection.rows
        if sel:
            d = docs[sel[0]]
            st.divider()
            st.subheader(f"{d['name']}  ·  v{d['version']}")
            a1, a2, a3 = st.columns([1, 1, 2])
            if d["status"] == "active":
                if a1.button("⏸ Disable (stop using)"):
                    api("POST", f"/api/kb/docs/{d['id']}/disable")
                    st.rerun()
            else:
                if a1.button("↩ Restore this version"):
                    with st.spinner("Re-indexing..."):
                        api("POST", f"/api/kb/docs/{d['id']}/restore")
                    st.rerun()
            confirm = a3.checkbox("Confirm permanent delete", key=f"cf{d['id']}")
            if a2.button("🗑 Delete", disabled=not confirm):
                api("DELETE", f"/api/kb/docs/{d['id']}")
                st.rerun()

            st.write(f"**Status:** {'In use' if d['status'] == 'active' else 'Archived (not used by the chatbot)'}  \n"
                     f"**Source:** {d['source_ref'] if d['source_kind'] == 'url' else 'uploaded file'}  \n"
                     f"**Added:** {d['created_at']}  ·  **Fingerprint:** `{(d['content_hash'] or '')[:12]}`")

            with st.expander("Version history", expanded=True):
                hist = api("GET", f"/api/kb/docs/{d['id']}/versions")
                show_df(pd.DataFrame([{
                    "Version": f"v{h['version']}",
                    "Status": "✅ Active" if h["status"] == "active" else "⏸ Archived",
                    "Chunks": h["chunk_count"], "Size": human(h["size_bytes"]),
                    "Added": h["created_at"], "Note": h["note"]} for h in hist]),
                    hide_index=True)
