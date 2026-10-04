from sqlalchemy import select

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

    # Determine neighboring chunks dynamically from the
    # chunk IDs returned by retrieval.
    for chunk in retrieved_chunks:

        chunk_id = chunk["chunk_id"]
        strategy = chunk["strategy"]

        prefix, number = chunk_id.rsplit("_", 1)
        target_index = int(number)

        start_index = max(0, target_index - window)
        end_index = target_index + window

        for index in range(start_index, end_index + 1):

            expanded_chunk_ids.add(
                (
                    strategy,
                    f"{prefix}_{index}"
                )
            )

    # Group IDs by strategy so that we can retrieve them
    # without making assumptions about the document.
    conditions = []

    for strategy, chunk_id in expanded_chunk_ids:

        conditions.append(
            (
                KnowledgeChunk.strategy == strategy,
                KnowledgeChunk.chunk_id == chunk_id
            )
        )

    if not conditions:
        return []

    async with engine.connect() as connection:

        from sqlalchemy import or_, and_

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
                or_(
                    and_(*condition)
                    for condition in conditions
                )
            )
            .order_by(KnowledgeChunk.id)
        )

        result = await connection.execute(query)

        rows = result.all()

    expanded_chunks = []

    for row in rows:

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
                "section": row[9]
            }
        )

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