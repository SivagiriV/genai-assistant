
"""Pinecone + embedding helpers shared by back office and chat API."""

import os
import time
import logging

log = logging.getLogger("vectordb")

EMBED_DIM = 384  # all-MiniLM-L6-v2


def make_embeddings():
    started = time.time()

    log.info("EMBEDDINGS: Starting model initialization")

    from langchain_huggingface import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True}
    )

    log.info(
        "EMBEDDINGS: Model initialized in %.2f seconds",
        time.time() - started
    )

    return embeddings


def connect(namespace, embeddings):
    started = time.time()

    from langchain_pinecone import PineconeVectorStore
    from pinecone import Pinecone, ServerlessSpec

    log.info("PINECONE: Initializing client")

    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

    name = os.getenv("PINECONE_INDEX", "support-assistant")

    log.info("PINECONE: Checking available indexes")

    indexes = pc.list_indexes().names()

    if name not in indexes:
        log.info("PINECONE: Index %s does not exist. Creating it.", name)

        pc.create_index(
            name=name,
            dimension=EMBED_DIM,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=os.getenv("PINECONE_CLOUD", "aws"),
                region=os.getenv("PINECONE_REGION", "us-east-1")
            )
        )

        deadline = time.time() + 120

        while not pc.describe_index(name).status["ready"]:
            if time.time() > deadline:
                raise TimeoutError(
                    "Pinecone index was not ready within 120 seconds"
                )

            time.sleep(2)

    log.info("PINECONE: Index is ready")

    index = pc.Index(name)

    vectorstore = PineconeVectorStore(
        index=index,
        embedding=embeddings,
        namespace=namespace
    )

    log.info(
        "PINECONE: Connected in %.2f seconds",
        time.time() - started
    )

    return index, vectorstore
