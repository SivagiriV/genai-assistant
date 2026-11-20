"""Pinecone + embedding helpers shared by back office and chat API.

Embeddings
----------
The Pinecone index holds vectors made with sentence-transformers/all-MiniLM-L6-v2
(384 dims, cosine). We keep exactly that model so existing vectors stay valid, but by
default we run it through ONNX (fastembed) instead of PyTorch:

  * ~10x smaller install, no torch            -> fast builds
  * ~250 MB RAM instead of 600 MB+            -> fits small Render instances
  * model is downloaded at BUILD time         -> no slow download on cold start

Set EMBEDDING_BACKEND=huggingface to use the original sentence-transformers path
(you must then add `sentence-transformers` and `langchain-huggingface` to requirements).

A single embeddings object is shared by the chatbot and the knowledge base.
"""

import logging
import os
import threading
import time
from pathlib import Path

from langchain_core.embeddings import Embeddings

log = logging.getLogger("vectordb")

EMBED_DIM = 384  # all-MiniLM-L6-v2
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CACHE_DIR = os.getenv("EMBED_CACHE_DIR") or str(Path(__file__).resolve().parent / ".model_cache")

_embeddings = None
_emb_lock = threading.Lock()


class FastEmbedEmbeddings(Embeddings):
    """LangChain Embeddings wrapper around fastembed (ONNX) with a tiny query cache."""

    def __init__(self, model_name=MODEL_NAME, cache_dir=CACHE_DIR):
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)
        self._qcache = {}
        self._qlock = threading.Lock()

    @staticmethod
    def _normalize(vec):
        import numpy as np

        v = np.asarray(vec, dtype="float32")
        n = float(np.linalg.norm(v))
        return (v / n if n else v).tolist()

    def embed_documents(self, texts):
        texts = list(texts)
        if not texts:
            return []
        return [self._normalize(v) for v in self._model.embed(texts, batch_size=32)]

    def embed_query(self, text):
        with self._qlock:
            hit = self._qcache.get(text)
        if hit is not None:
            return hit
        vec = self.embed_documents([text])[0]
        with self._qlock:
            if len(self._qcache) > 256:
                self._qcache.pop(next(iter(self._qcache)))
            self._qcache[text] = vec
        return vec


def _make_huggingface():
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=MODEL_NAME, encode_kwargs={"normalize_embeddings": True})


def make_embeddings():
    """Return the process-wide embeddings object (created once, thread-safe)."""
    global _embeddings
    if _embeddings is not None:
        return _embeddings
    with _emb_lock:
        if _embeddings is not None:
            return _embeddings
        started = time.time()
        backend = os.getenv("EMBEDDING_BACKEND", "fastembed").lower()
        log.info("EMBEDDINGS: loading backend=%s", backend)
        if backend == "huggingface":
            emb = _make_huggingface()
        else:
            try:
                emb = FastEmbedEmbeddings()
            except ImportError:
                log.warning("EMBEDDINGS: fastembed not installed, falling back to sentence-transformers")
                emb = _make_huggingface()
        log.info("EMBEDDINGS: ready in %.2fs", time.time() - started)
        _embeddings = emb
        return emb


def connect(namespace, embeddings):
    started = time.time()

    from langchain_pinecone import PineconeVectorStore
    from pinecone import Pinecone, ServerlessSpec

    log.info("PINECONE: initializing client")
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    name = os.getenv("PINECONE_INDEX", "support-assistant")

    if name not in pc.list_indexes().names():
        log.info("PINECONE: index %s does not exist, creating it", name)
        pc.create_index(
            name=name,
            dimension=EMBED_DIM,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=os.getenv("PINECONE_CLOUD", "aws"),
                region=os.getenv("PINECONE_REGION", "us-east-1"),
            ),
        )
        deadline = time.time() + 120
        while not pc.describe_index(name).status["ready"]:
            if time.time() > deadline:
                raise TimeoutError("Pinecone index was not ready within 120 seconds")
            time.sleep(2)

    index = pc.Index(name)
    vectorstore = PineconeVectorStore(index=index, embedding=embeddings, namespace=namespace)
    log.info("PINECONE: connected in %.2fs", time.time() - started)
    return index, vectorstore
