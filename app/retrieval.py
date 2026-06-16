import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from sentence_transformers import SentenceTransformer

from pinecone import Pinecone, ServerlessSpec

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise ValueError(
        "Missing PINECONE_API_KEY."
    )

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Pinecone setup

index_name = "adaptive-rag"
pc = Pinecone(api_key=PINECONE_API_KEY)

if index_name not in pc.list_indexes().names():
    pc.create_index(
        index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
index = pc.Index(index_name)

# Load documents

def load_docs(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    return documents

def split_docs(documents, chunk_size=500):
    chunks = []
    for doc in documents:
        text = doc.page_content
        for i in range(0, len(text), chunk_size):
            chunk = text[i : i + chunk_size]
            chunks.append(chunk)
    return chunks

# Upload to Pinecone

def upload_chunks(chunks, pdf_name="document"):
    if not chunks:
        raise ValueError(
            "No text chunks generated from the PDF."
        )

    vectors = []
    for i, chunk in enumerate(chunks):
        embedding = embedding_model.encode(chunk).tolist()
        uid = f"{pdf_name}_{i}"
        metadata = {"text": chunk, "source": pdf_name}
        vectors.append((uid, embedding, metadata))

    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i : i + batch_size]
        index.upsert(vectors=batch)

    print(f"Uploaded {len(chunks)} chunks to Pinecone.")

# Retrieval

def retrieve(query, top_k=5):
    query_embedding = embedding_model.encode(query).tolist()
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        include_values=False,
    )

    retrieved_docs = []
    matches = results.get("matches", [])
    for match in matches:
        meta = match.get("metadata", {})
        text = meta.get("text") or meta.get("contents") or ""
        retrieved_docs.append(text)

    return retrieved_docs


def ingest_docs(pdf_path):
    documents = load_docs(pdf_path)
    chunks = split_docs(documents)
    if not chunks:
        raise ValueError(
            "Document contains no text chunks to ingest."
        )
    pdf_name = os.path.basename(pdf_path).split(".")[0]
    upload_chunks(chunks, pdf_name)
    print("Document ingestion complete.")
    return len(chunks)

