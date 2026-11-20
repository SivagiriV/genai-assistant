"""Run at BUILD time (see render.yaml) so the embedding model is already on disk at runtime.

    python warmup_models.py
"""
import logging

logging.basicConfig(level=logging.INFO)

from vectordb import make_embeddings

if __name__ == "__main__":
    vec = make_embeddings().embed_query("warm up")
    print(f"embedding model ready, dim={len(vec)}")
