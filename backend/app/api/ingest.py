# -------- checking the Loading of folder and meta data-----------------------

# from langchain_community.document_loaders import PyPDFLoader

# def test_loader():
#     pdf_path = "backend/uploads/sample.pdf"  # put any test PDF here
#     loader = PyPDFLoader(pdf_path)

#     pages = loader.load()

#     print("Total pages:", len(pages))
#     print("\nFirst page content preview:\n")
#     print(pages[0].page_content[:500])

#     print("\nMetadata:\n")
#     print(pages[0].metadata)


# if __name__ == "__main__":
#     test_loader()

# ------------------- About the chunking of the document -----------------------

# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter

# def test_chunking():
#     pdf_path = "uploads/sample.pdf"
#     pages = PyPDFLoader(pdf_path).load()

#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=500,      # smaller so you can SEE chunking with a 1-page resume
#         chunk_overlap=80,
#         separators=["\n\n", "\n", " ", ""],
#     )

#     chunks = splitter.split_documents(pages)

#     print("Pages:", len(pages))
#     print("Chunks:", len(chunks))

#     # Add chunk_id and keep useful metadata
#     for i, c in enumerate(chunks):
#         c.metadata["chunk_id"] = i
#         # Friendly page for display (prefer page_label if present)
#         c.metadata["page_display"] = c.metadata.get("page_label") or str(c.metadata.get("page", 0) + 1)

#     # Show first 3 chunks
#     for c in chunks[:3]:
#         print("\n---")
#         print("chunk_id:", c.metadata["chunk_id"])
#         print("source:", c.metadata.get("source"))
#         print("page(stored):", c.metadata.get("page"))
#         print("page(display):", c.metadata.get("page_display"))
#         print("text preview:", c.page_content[:200])

# if __name__ == "__main__":
#     test_chunking()

# ------------------------ to store in Chroma -----------------------

from __future__ import annotations

import os
import glob
import hashlib
from pathlib import Path
from typing import List

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Prefer the new package if you installed it; otherwise fallback works.
try:
    from langchain_chroma import Chroma  # pip install -U langchain-chroma
except Exception:
    from langchain_community.vectorstores import Chroma  # fallback (deprecated but OK)



# 0) Load environment variables

load_dotenv()


CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))

# Use LOCAL embeddings by default (no OpenAI quota needed)
EMBED_PROVIDER = os.getenv("EMBED_PROVIDER", "local").lower()
OPENAI_EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
LOCAL_EMBED_MODEL = os.getenv("LOCAL_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")



# 1) Helper: stable IDs

def stable_chunk_id(source: str, page: int, chunk_id: int) -> str:
    raw = f"{source}|{page}|{chunk_id}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()



# 2) Embeddings factory

# def get_embeddings():  # you can switch to OpenAIEmbeddings if you have an API key and want to use OpenAI's embedding models  
#     """
#     Two modes:
#       - local (default): sentence-transformers (no API key needed)
#       - openai: requires OPENAI_API_KEY and available quota
#     """
#     if EMBED_PROVIDER == "openai":
#         from langchain_openai import OpenAIEmbeddings
#         return OpenAIEmbeddings(model=OPENAI_EMBED_MODEL)

#     # Local embeddings
#     from langchain_community.embeddings import HuggingFaceEmbeddings
#     return HuggingFaceEmbeddings(model_name=LOCAL_EMBED_MODEL)


def get_embeddings():
    # Local embeddings (no OpenAI quota needed)
    from langchain_community.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# 3) Load PDFs

def load_pdf_pages(pdf_path: str):
    """
    PyPDFLoader returns one Document per page, with metadata including:
      - source
      - page (0-based)
      - page_label (often 1-based string if available)
    """
    loader = PyPDFLoader(pdf_path)
    return loader.load()



# 4) Split into chunks + add metadata

def split_and_tag(pages) -> List:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )

    chunks = splitter.split_documents(pages)

    for i, c in enumerate(chunks):
        c.metadata["chunk_id"] = i
        # Use page_label if present, else page+1 for display
        page_label = c.metadata.get("page_label")
        page_num = c.metadata.get("page", 0)
        c.metadata["page_display"] = page_label or str(page_num + 1)

        # Ensure we always have 'source'
        if "source" not in c.metadata:
            c.metadata["source"] = "unknown_source"

    return chunks



# 5) Store in Chroma

def store_in_chroma(chunks, persist_dir: str = CHROMA_DIR):
    from langchain_community.embeddings import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectordb = Chroma(
        collection_name="pdf_docs",
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )

    ids = [
        stable_chunk_id(
            c.metadata.get("source", "unknown_source"),
            int(c.metadata.get("page", 0)),
            int(c.metadata["chunk_id"]),
        )
        for c in chunks
    ]

    vectordb.add_documents(chunks, ids=ids)
    try:
        vectordb.persist()
    except Exception:
        pass

    return vectordb



# 6) Ingest one PDF

def ingest_one_pdf(pdf_path: str):
    pages = load_pdf_pages(pdf_path)
    print("Pages:", len(pages))

    chunks = split_and_tag(pages)
    print("Chunks:", len(chunks))

    store_in_chroma(chunks, persist_dir=CHROMA_DIR)
    print(f"Stored {len(chunks)} chunks in Chroma at {CHROMA_DIR}")



# 7) Ingest a folder (optional)

def ingest_folder(folder_path: str):
    pdfs = sorted(glob.glob(str(Path(folder_path) / "*.pdf")))
    if not pdfs:
        print(f"No PDFs found in: {folder_path}")
        return

    print(f"Found {len(pdfs)} PDFs")

    for p in pdfs:
        print(f"\n--- Ingesting: {p}")
        ingest_one_pdf(p)

# 8) Main function to run when executing this script directly

if __name__ == "__main__":
    from pathlib import Path
    BASE_DIR = Path(__file__).resolve().parents[2]
    pdf_path = BASE_DIR / "uploads" / "sample.pdf"
    ingest_one_pdf(str(pdf_path))