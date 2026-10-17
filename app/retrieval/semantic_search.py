import asyncio
import time

from sqlalchemy import select

from app.database.connection import engine
from app.database.models import KnowledgeChunk
from app.database.session import SessionLocal
from app.embeddings.model import create_embedding


async def semantic_search(
    query: str,
    top_k: int = 15,
    strategy: str | None = None,
    return_metadata: bool = False,
):
    """
    Convert the query into an embedding and retrieve
    the most semantically similar chunks from PostgreSQL.

    Parameters:
        query: User's natural-language question.
        top_k: Number of candidate chunks to retrieve.
        strategy: Optional chunking strategy filter.
                  Examples: recursive, structure_aware, contextual.
        return_metadata: When True, return (results, metadata) instead
                         of only the retrieved list.
    """

    embedding_start = time.perf_counter()
    query_embedding = create_embedding(query)
    embedding_elapsed_ms = (time.perf_counter() - embedding_start) * 1000.0

    search_start = time.perf_counter()
    async with SessionLocal() as session:
        distance = KnowledgeChunk.embedding.cosine_distance(
            query_embedding
        )

        statement = (
            select(
                KnowledgeChunk,
                distance.label("distance")
            )
            .order_by(distance)
            .limit(top_k)
        )

        if strategy:
            statement = statement.where(
                KnowledgeChunk.strategy == strategy
            )

        result = await session.execute(statement)
        rows = result.all()

    search_elapsed_ms = (time.perf_counter() - search_start) * 1000.0

    results = []

    for chunk, distance_value in rows:
        distance_value = float(distance_value)
        results.append({
            "id": chunk.id,
            "document_id": chunk.document_id,
            "document_name": chunk.document_name,
            "chunk_id": chunk.chunk_id,
            "parent_chunk_id": chunk.parent_chunk_id,
            "strategy": chunk.strategy,
            "section": chunk.section,
            "page_number": chunk.page_number,
            "summary": chunk.summary,
            "content": chunk.content,
            "distance": distance_value,
            "similarity": 1 - distance_value,
        })

    metadata = {
        "query_embedding_time_ms": embedding_elapsed_ms,
        "semantic_search_time_ms": search_elapsed_ms,
        "total_retrieval_time_ms": embedding_elapsed_ms + search_elapsed_ms,
        "top_k": top_k,
        "strategy": strategy,
    }

    if return_metadata:
        return results, metadata

    return results


async def main():

    # ---------------------------------------------------------
    # Test query
    # ---------------------------------------------------------
    query = "What was Zen Technologies EBITDA in FY 2023-24?"

    # We currently want a larger candidate pool.
    # Later, the reranker will select the best chunks from these.
    top_k = 15
    # Test all three chunking strategies separately.
    strategies = [
        "recursive",
        "structure_aware",
        "contextual",
    ]

    print("\n" + "=" * 80)
    print("SEMANTIC RETRIEVAL COMPARISON")
    print("=" * 80)

    print(f"\nQuery: {query}")
    print(f"Candidate count: Top {top_k}")

    # ---------------------------------------------------------
    # Run semantic search for each strategy
    # ---------------------------------------------------------
    for strategy in strategies:

        print("\n" + "=" * 80)
        print(f"STRATEGY: {strategy}")
        print("=" * 80)

        results = await semantic_search(
            query=query,
            top_k=top_k,
            strategy=strategy
        )

        if not results:
            print("\nNo results found.")
            continue

        # -----------------------------------------------------
        # Display retrieved chunks
        # -----------------------------------------------------
        for i, result in enumerate(results, start=1):

            print(f"\nRESULT {i}")
            print("-" * 80)

            print(f"Chunk ID   : {result['chunk_id']}")
            print(f"Strategy   : {result['strategy']}")
            print(f"Section    : {result['section']}")
            print(f"Similarity : {result['similarity']:.4f}")
            print(f"Distance   : {result['distance']:.4f}")

            print("\nContent:")
            print(result["content"])


if __name__ == "__main__":
    asyncio.run(main())
