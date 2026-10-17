
def select_context(
    chunks: list[dict],
    max_chunks: int = 10
) -> list[dict]:
    """
    Select relevant, non-duplicate chunks.

    Prioritizes chunks with reranker scores, then preserves
    the order of expanded neighboring chunks.
    """

    if not chunks or max_chunks <= 0:
        return []

    # Remove empty and duplicate chunks.
    unique_chunks = []
    seen = set()

    for chunk in chunks:
        content = chunk.get("content", "").strip()

        if not content:
            continue

        key = (
            chunk.get("strategy", ""),
            content
        )

        if key in seen:
            continue

        seen.add(key)
        unique_chunks.append(chunk)

    # Separate scored chunks from neighboring chunks.
    ranked_chunks = [
        chunk for chunk in unique_chunks
        if chunk.get("rerank_score") is not None
    ]

    neighboring_chunks = [
        chunk for chunk in unique_chunks
        if chunk.get("rerank_score") is None
    ]

    # Highest reranker scores first.
    ranked_chunks.sort(
        key=lambda chunk: chunk["rerank_score"],
        reverse=True
    )

    # Keep the strongest results, then fill remaining slots
    # with neighboring context in its original order.
    selected = ranked_chunks[:max_chunks]

    remaining = max_chunks - len(selected)

    if remaining > 0:
        selected.extend(neighboring_chunks[:remaining])

    return selected