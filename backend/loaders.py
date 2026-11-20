"""Turn files / websites into LangChain Documents."""
import base64
import io
import re
from pathlib import Path
from urllib.parse import urldefrag, urljoin, urlparse

import pandas as pd
from langchain_core.documents import Document

TYPE_BY_EXT = {
    ".pdf": "PDF", ".docx": "Word", ".xlsx": "Excel", ".csv": "CSV",
    ".txt": "Text", ".md": "Text",
    ".png": "Image", ".jpg": "Image", ".jpeg": "Image", ".webp": "Image",
}
SUPPORTED_EXTS = sorted(TYPE_BY_EXT)
DOC_TYPES = ["PDF", "Word", "Excel", "CSV", "Text", "Image", "Website"]

IMAGE_PROMPT = (
    "You are preparing a customer-support knowledge base. First transcribe ALL visible text "
    "exactly as written. Then describe what the image shows (screenshots, diagrams, tables, "
    "steps, labels) in clear detail. Do not invent anything that is not visible.")


def doc_type_for(filename):
    ext = Path(filename).suffix.lower()
    if ext not in TYPE_BY_EXT:
        raise ValueError(f"Unsupported file type '{ext}'. Supported: {', '.join(SUPPORTED_EXTS)}")
    return TYPE_BY_EXT[ext]


def _read_text(path):
    raw = Path(path).read_bytes()
    for enc in ("utf-8-sig", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="ignore")


def df_to_docs(df, source, sheet=None):
    """One Document per table row ('column: value' lines)."""
    docs = []
    for i, row in df.iterrows():
        parts = [f"{c}: {v}" for c, v in row.items() if pd.notna(v) and str(v).strip() != ""]
        if not parts:
            continue
        md = {"source": source, "row": int(i) + 1}
        if sheet:
            md["sheet"] = str(sheet)
        docs.append(Document(page_content="\n".join(parts), metadata=md))
    return docs


def _image_b64(path):
    try:
        from PIL import Image
        img = Image.open(path).convert("RGB")
        img.thumbnail((1600, 1600))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return base64.b64encode(Path(path).read_bytes()).decode()


def describe_image(path, vision):
    from langchain_core.messages import HumanMessage
    msg = HumanMessage(content=[
        {"type": "text", "text": IMAGE_PROMPT},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_image_b64(path)}"}},
    ])
    return vision.invoke([msg]).content


def load_file(path, display_name, vision=None):
    ext = Path(display_name).suffix.lower()
    dtype = doc_type_for(display_name)
    if ext == ".pdf":
        from langchain_community.document_loaders import PyPDFLoader
        docs = PyPDFLoader(str(path)).load()
        for d in docs:
            d.metadata = {"source": display_name, "page": d.metadata.get("page")}
        docs = [d for d in docs if d.page_content.strip()]
        if not docs:
            raise ValueError("No readable text in this PDF (scanned?). Upload its pages as images instead.")
        return docs
    if ext in (".txt", ".md"):
        return [Document(page_content=_read_text(path), metadata={"source": display_name})]
    if ext == ".csv":
        try:
            df = pd.read_csv(path)
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding="latin-1")
        return df_to_docs(df, display_name)
    if ext == ".xlsx":
        docs = []
        for sheet, df in pd.read_excel(path, sheet_name=None).items():
            docs += df_to_docs(df, display_name, sheet)
        return docs
    if ext == ".docx":
        from docx import Document as DocxDocument
        d = DocxDocument(str(path))
        parts = [p.text for p in d.paragraphs if p.text.strip()]
        for t in d.tables:
            for row in t.rows:
                parts.append(" | ".join(c.text.strip() for c in row.cells))
        return [Document(page_content="\n".join(parts), metadata={"source": display_name})]
    if dtype == "Image":
        if vision is None:
            raise ValueError("No vision model configured for images.")
        return [Document(page_content=describe_image(path, vision), metadata={"source": display_name})]
    raise ValueError(f"Unsupported file type '{ext}'")


# ---------------------------------------------------------------- websites
SKIP_EXT = (".pdf", ".jpg", ".jpeg", ".png", ".gif", ".svg", ".zip", ".css", ".js",
            ".ico", ".mp4", ".webp", ".xml")


def normalize_url(url):
    url = url.strip()
    if not re.match(r"^https?://", url, re.I):
        url = "https://" + url
    url, _ = urldefrag(url)
    p = urlparse(url)
    if not p.netloc:
        raise ValueError("That doesn't look like a valid URL.")
    return url.rstrip("/") if p.path in ("", "/") else url


def crawl(url, max_pages=10):
    """Breadth-first crawl of one site (same domain only). Returns Documents, one per page."""
    import requests
    from bs4 import BeautifulSoup

    root = normalize_url(url)
    host = urlparse(root).netloc
    queue, seen, docs = [root], {root}, []
    while queue and len(docs) < max_pages:
        page = queue.pop(0)
        try:
            r = requests.get(page, timeout=15, headers={"User-Agent": "SupportBotCrawler/1.0"})
        except Exception:
            continue
        if r.status_code != 200 or "text/html" not in r.headers.get("content-type", ""):
            continue
        soup = BeautifulSoup(r.content, "html.parser")   # bytes: honours the page's own charset
        for a in soup.find_all("a", href=True):       # collect links BEFORE stripping nav/footer
            link, _ = urldefrag(urljoin(page, a["href"]))
            lp = urlparse(link)
            if (lp.scheme in ("http", "https") and lp.netloc == host
                    and not lp.path.lower().endswith(SKIP_EXT) and link not in seen):
                seen.add(link)
                queue.append(link)
        title = soup.title.string.strip() if soup.title and soup.title.string else page
        for tag in soup(["script", "style", "noscript", "nav", "footer", "form", "svg"]):
            tag.decompose()
        text = re.sub(r"\n\s*\n+", "\n", soup.get_text("\n")).strip()
        if len(text) > 40:
            docs.append(Document(page_content=text,
                                 metadata={"source": root, "page_url": page, "title": title}))
    return docs
