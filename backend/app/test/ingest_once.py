import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings  # ✅ not deprecated

load_dotenv()

CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
PDF_PATH = "uploads/sample.pdf"


def main():
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"PDF not found at: {PDF_PATH}")

    pages = PyPDFLoader(PDF_PATH).load()
    print("Loaded pages:", len(pages))

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(pages)
    print("Total chunks:", len(chunks))
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Create vector database
    vs = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )

    # Store chunks
    vs.add_documents(chunks)

    print("✅ Ingested into:", CHROMA_DIR)
    print("New count:", vs._collection.count())



if __name__ == "__main__":
    main()