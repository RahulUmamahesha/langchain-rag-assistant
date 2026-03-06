# app/rag/citations.py
def build_citations(docs, max_sources=4, snippet_chars=220):
    citations = []
    seen = set()

    for d in docs:
        m = d.metadata or {}
        source = m.get("source", "unknown")
        page = m.get("page", "unknown")
        chunk_id = m.get("chunk_id", "unknown")

        key = (source, page, chunk_id)
        if key in seen:
            continue
        seen.add(key)

        citations.append({
            "source": source,
            "page": page,
            "chunk_id": chunk_id,
            "snippet": (d.page_content[:snippet_chars].replace("\n", " ").strip() + "…")
                      if d.page_content else ""
        })

        if len(citations) >= max_sources:
            break

    return citations