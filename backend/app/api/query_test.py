import os

try:
    from langchain_chroma import Chroma
except Exception:
    from langchain_community.vectorstores import Chroma

from langchain_community.embeddings import HuggingFaceEmbeddings

   

def main():
    db = Chroma(
        collection_name="pdf_docs",
        persist_directory=os.getenv("CHROMA_DIR", "./chroma_db"),
        embedding_function=HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        ),
    )

    # --- MMR retrieval ---
    # results = db.similarity_search("What is the person's education?", k=6)
    
    # results = db.max_marginal_relevance_search(
    # "What is the person's education?",
    # k=5,
    # fetch_k=15) 

    # results = db.max_marginal_relevance_search(
    # "education University of Florida degree",
    # k=5,
    # fetch_k=20)

    results = db.max_marginal_relevance_search(
        "education University of Florida degree",
        k=5,
        fetch_k=20
    )

    print("\n=== Retrieval Results ===")
    for r in results:
        print("\n---")
        print("chunk_id:", r.metadata.get("chunk_id"))
        print("preview:", r.page_content[:200])

    # --- DEBUG: check if education text exists at all ---
    print("\n=== Checking Stored Documents ===")
    all_docs = db.get()
    texts = all_docs["documents"]

    for i, t in enumerate(texts):
        if "Education" in t or "University of Florida" in t:
            print("\n--- EDUCATION FOUND ---")
            print(t[:300])


if __name__ == "__main__":
    main()

