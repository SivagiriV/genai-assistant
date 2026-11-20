"""KnowledgeBase: the company-side service (add / version / disable / restore / delete)."""
import hashlib
import os
import re
import shutil
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import loaders
from registry import Registry

load_dotenv()
KEEP_META = ("source", "page", "sheet", "page_url", "title", "row")


def clean_meta(md):
    return {k: v for k, v in md.items() if k in KEEP_META and v is not None}


def _safe(name):
    return re.sub(r"[^\w.\- ]", "_", name)[:80]


def _sha(data: bytes):
    return hashlib.sha256(data).hexdigest()


class KnowledgeBase:
    def __init__(self, namespace=None, db_path=None, storage_dir=None,
                 chunk_size=800, chunk_overlap=120, vectorstore=None, index=None):
        self.namespace = namespace or os.getenv("PINECONE_NAMESPACE", "kb")
        self.registry = Registry(db_path or os.getenv("KB_DB", "data/kb.db"))
        self.storage = Path(storage_dir or os.getenv("KB_STORAGE", "storage"))
        self.storage.mkdir(parents=True, exist_ok=True)
        self.index_name = os.getenv("PINECONE_INDEX", "support-assistant")
        self.set_chunking(chunk_size, chunk_overlap)
        self._vision = None
        if vectorstore is None:                      # (injectable for tests)
            import vectordb
            index, vectorstore = vectordb.connect(self.namespace, vectordb.make_embeddings())
        self.index, self.vectorstore = index, vectorstore

    def set_chunking(self, size, overlap):
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)

    def _vision_llm(self):
        if self._vision is None:
            from langchain_groq import ChatGroq
            self._vision = ChatGroq(model=os.getenv("VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct"),
                                    temperature=0)
        return self._vision

    # ---------- vectors ----------
    def _split(self, docs):
        chunks = [c for c in self.splitter.split_documents(docs) if c.page_content.strip()]
        for c in chunks:
            c.metadata = clean_meta(c.metadata)
        return chunks

    def _index_chunks(self, doc_id, chunks):
        self.vectorstore.add_documents(chunks, ids=[f"{doc_id}::{i}" for i in range(len(chunks))])

    def _remove_vectors(self, doc_id, n):
        for i in range(0, n, 1000):                  # Pinecone: max 1000 ids per delete
            ids = [f"{doc_id}::{j}" for j in range(i, min(i + 1000, n))]
            self.index.delete(ids=ids, namespace=self.namespace)

    # ---------- publishing a new version ----------
    def _publish(self, name, doc_type, kind, ref, size, digest, chunks, note):
        old = self.registry.get_active(name)
        version = self.registry.latest_version(name) + 1
        doc_id = self.registry.add_document(name, doc_type, kind, ref, version,
                                            len(chunks), size, digest, note)
        try:
            for c in chunks:
                c.metadata.update(doc_id=doc_id, version=version, doc_type=doc_type)
            self._index_chunks(doc_id, chunks)
            self.registry.save_chunks(doc_id, [(c.page_content, c.metadata) for c in chunks])
        except Exception:
            self._remove_vectors(doc_id, len(chunks))
            self.registry.delete_document(doc_id)
            raise
        if old:
            self._deactivate(old)                    # old version stops being used
        self.registry.bump()
        return {"status": "added" if version == 1 else "updated", "name": name,
                "version": version, "chunks": len(chunks), "doc_id": doc_id}

    def add_file(self, path, original_name, note=""):
        doc_type = loaders.doc_type_for(original_name)
        data = Path(path).read_bytes()
        digest = _sha(data)
        old = self.registry.get_active(original_name)
        if old and old["content_hash"] == digest:
            return {"status": "unchanged", "name": original_name, "version": old["version"]}
        stored = self.storage / f"{int(time.time() * 1000)}_{_safe(original_name)}"
        stored.write_bytes(data)
        try:
            vision = self._vision_llm() if doc_type == "Image" else None
            docs = loaders.load_file(stored, original_name, vision=vision)
            chunks = self._split(docs)
            if not chunks:
                raise ValueError("No readable content found in this file.")
            return self._publish(original_name, doc_type, "file", stored, len(data), digest, chunks, note)
        except Exception:
            stored.unlink(missing_ok=True)
            raise

    def add_url(self, url, max_pages=10, note=""):
        name = loaders.normalize_url(url)
        docs = loaders.crawl(name, max_pages)
        if not docs:
            raise ValueError("Could not read any pages from that URL.")
        text = "\n".join(d.page_content for d in docs)
        digest = _sha(text.encode("utf-8"))
        old = self.registry.get_active(name)
        if old and old["content_hash"] == digest:
            return {"status": "unchanged", "name": name, "version": old["version"]}
        chunks = self._split(docs)
        return self._publish(name, "Website", "url", name, len(text), digest, chunks,
                             note or f"{len(docs)} page(s) crawled")

    # ---------- lifecycle ----------
    def _deactivate(self, doc):
        self._remove_vectors(doc["id"], doc["chunk_count"])
        self.registry.set_status(doc["id"], "archived")

    def disable(self, doc_id):
        doc = self.registry.get(doc_id)
        if doc and doc["status"] == "active":
            self._deactivate(doc)
            self.registry.bump()

    def restore(self, doc_id):
        """Make an archived version live again (replacing the current one)."""
        doc = self.registry.get(doc_id)
        if not doc or doc["status"] == "active":
            return
        rows = self.registry.get_chunks(doc_id)
        if not rows:
            raise ValueError("Stored chunks for this version are missing; please upload it again.")
        current = self.registry.get_active(doc["name"])
        self._index_chunks(doc_id, [Document(page_content=t, metadata=m) for t, m in rows])
        self.registry.set_status(doc_id, "active")
        if current:
            self._deactivate(current)
        self.registry.bump()

    def delete(self, doc_id):
        doc = self.registry.get(doc_id)
        if not doc:
            return
        if doc["status"] == "active":
            self._remove_vectors(doc_id, doc["chunk_count"])
        self.registry.delete_document(doc_id)
        if doc["source_kind"] == "file":
            shared = any(d["source_ref"] == doc["source_ref"] for d in self.registry.versions_of(doc["name"]))
            if not shared:
                Path(doc["source_ref"]).unlink(missing_ok=True)
        self.registry.bump()

    def reset(self):
        """Wipe everything in this namespace (used by evaluate.py)."""
        try:
            self.index.delete(delete_all=True, namespace=self.namespace)
        except Exception:
            pass
        self.registry.clear_all()
        shutil.rmtree(self.storage, ignore_errors=True)
        self.storage.mkdir(parents=True, exist_ok=True)
        self.registry.bump()

    # ---------- queries ----------
    def list_docs(self, **kw):
        return self.registry.list(**kw)

    def stats(self):
        return self.registry.stats()
