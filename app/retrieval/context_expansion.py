from sqlalchemy import and_, or_, select

from app.database.connection import engine
from app.database.models import KnowledgeChunk


async def expand_context(
    retrieved_chunks,
    window: int = 2
):
    """
    Expand the context around chunks selected by retrieval/reranking.

    This function is document-independent.

    Parameters
    ----------
    retrieved_chunks : list
        Chunks returned by the retrieval/reranking stage.

    window : int
        Number of neighboring chunks to retrieve on each side.

    Returns
    -------
    list
        Expanded chunks ordered by database position.
    """

    if not retrieved_chunks:
        return []

    expanded_chunk_ids = set()
    original_chunks = {}

    # Determine neighboring chunks dynamically from the
    # chunk IDs returned by retrieval.
    for chunk in retrieved_chunks:

        chunk_id = chunk["chunk_id"]
        strategy = chunk["strategy"]
        document_id = chunk.get("document_id")

        original_key = (document_id, strategy, chunk_id)
        original_chunks[original_key] = chunk
        expanded_chunk_ids.add(original_key)

        try:
            prefix, number = chunk_id.rsplit("_", 1)
            target_index = int(number)
        except (AttributeError, ValueError):
            continue

        start_index = max(0, target_index - window)
        end_index = target_index + window

        for index in range(start_index, end_index + 1):

            expanded_chunk_ids.add(
                (
                    document_id,
                    strategy,
                    f"{prefix}_{index}"
                )
            )

    # Group IDs by strategy so that we can retrieve them
    # without making assumptions about the document.
    conditions = []

    for document_id, strategy, chunk_id in expanded_chunk_ids:

        conditions_for_chunk = [
            KnowledgeChunk.strategy == strategy,
            KnowledgeChunk.chunk_id == chunk_id,
        ]

        if document_id is not None:
            conditions_for_chunk.append(
                KnowledgeChunk.document_id == document_id
            )

        conditions.append(
            and_(*conditions_for_chunk)
        )

    if not conditions:
        return []

    async with engine.connect() as connection:

        query = (
            select(
                KnowledgeChunk.id,
                KnowledgeChunk.document_id,
                KnowledgeChunk.document_name,
                KnowledgeChunk.chunk_id,
                KnowledgeChunk.parent_chunk_id,
                KnowledgeChunk.strategy,
                KnowledgeChunk.content,
                KnowledgeChunk.summary,
                KnowledgeChunk.page_number,
                KnowledgeChunk.section
            )
            .where(
                or_(*conditions)
            )
        )

        result = await connection.execute(query)

        rows = result.all()

    expanded_chunks = []

    for row in rows:

        original_key = (row[1], row[5], row[3])
        original = original_chunks.get(original_key, {})

        expanded_chunks.append(
            {
                "id": row[0],
                "document_id": row[1],
                "document_name": row[2],
                "chunk_id": row[3],
                "parent_chunk_id": row[4],
                "strategy": row[5],
                "content": row[6],
                "summary": row[7],
                "page_number": row[8],
                "section": row[9],
                **{
                    key: value
                    for key, value in original.items()
                    if key not in {
                        "id", "document_id", "document_name", "chunk_id",
                        "parent_chunk_id", "strategy", "content", "summary",
                        "page_number", "section",
                    }
                },
            }
        )

    def chunk_order(chunk):
        chunk_id = chunk["chunk_id"]
        prefix, separator, suffix = chunk_id.rpartition("_")

        if separator and suffix.isdigit():
            return (
                chunk["document_id"],
                chunk["strategy"],
                prefix,
                int(suffix),
            )

        return (chunk["document_id"], chunk["strategy"], chunk_id, -1)

    expanded_chunks.sort(key=chunk_order)

    return expanded_chunks


def prepare_retrieval_chunks(results):
    """
    Convert retrieval/reranking results into the standard format
    expected by context expansion.

    This keeps context expansion independent from the exact
    implementation of semantic search or reranking.
    """

    chunks = []

    for result in results:

        if isinstance(result, dict):

            chunks.append(
                {
                    "chunk_id": result["chunk_id"],
                    "strategy": result["strategy"]
                }
            )

        else:

            chunks.append(
                {
                    "chunk_id": result.chunk_id,
                    "strategy": result.strategy
                }
            )

    return chunks