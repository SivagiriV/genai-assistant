"""Pinecone + embedding helpers shared by the back office and the chat API."""
import os
import time

EMBED_DIM = 384  # all-MiniLM-L6-v2


def make_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                 encode_kwargs={"normalize_embeddings": True})


def connect(namespace, embeddings):
    """Return (index, vectorstore), creating the Pinecone index on first use."""
    from langchain_pinecone import PineconeVectorStore
    from pinecone import Pinecone, ServerlessSpec

    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    name = os.getenv("PINECONE_INDEX", "support-assistant")
    if name not in pc.list_indexes().names():
        pc.create_index(name=name, dimension=EMBED_DIM, metric="cosine",
                        spec=ServerlessSpec(cloud=os.getenv("PINECONE_CLOUD", "aws"),
                                            region=os.getenv("PINECONE_REGION", "us-east-1")))
        while not pc.describe_index(name).status["ready"]:
            time.sleep(1)
    index = pc.Index(name)
    return index, PineconeVectorStore(index=index, embedding=embeddings, namespace=namespace)
