# app/ingest/chunking.py
from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_with_metadata(pages, chunk_size=1000, chunk_overlap=150, source_name="uploaded.pdf"):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = []
    global_chunk_id = 0

    for page_i, doc in enumerate(pages):
        # doc.page_content, doc.metadata usually include "page"
        page_num = doc.metadata.get("page", page_i)  # fallback if loader differs
        splits = splitter.split_text(doc.page_content)

        for split_i, text in enumerate(splits):
            meta = dict(doc.metadata)
            meta.update({
                "source": meta.get("source", source_name),
                "page": page_num,
                "chunk_id": f"p{page_num}_c{split_i}_g{global_chunk_id}"
            })
            # rebuild as Document if needed
            doc.__class__  # (just to show we're using Document objects)
            from langchain_core.documents import Document
            chunks.append(Document(page_content=text, metadata=meta))
            global_chunk_id += 1

    return chunks