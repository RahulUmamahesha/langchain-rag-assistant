import os
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
TOP_K = int(os.getenv("TOP_K", "4"))

def main():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vs = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )

    query = "What is this document about?"
    docs = vs.similarity_search(query, k=TOP_K)

    print("Query:", query)
    print("Retrieved:", len(docs))

    for i, d in enumerate(docs, start=1):
        print(f"\n--- RESULT {i} ---")
        print("source:", d.metadata.get("source"), "page:", d.metadata.get("page"))
        print(d.page_content[:400])

if __name__ == "__main__":
    main()