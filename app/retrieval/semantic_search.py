import asyncio

from sqlalchemy import select

from app.database.connection import engine
from app.database.models import KnowledgeChunk
from app.database.session import SessionLocal
from app.embeddings.model import create_embedding


async def semantic_search(
    query: str,
    top_k: int = 15,
    strategy: str | None = None
):
    """
    Convert the query into an embedding and retrieve
    the most semantically similar chunks from PostgreSQL.

    Parameters:
        query: User's natural-language question.
        top_k: Number of candidate chunks to retrieve.
        strategy: Optional chunking strategy filter.
                  Examples: recursive, structure_aware, contextual.
    """

    # ---------------------------------------------------------
    # 1. Convert the user query into a 384-dimensional embedding
    # ---------------------------------------------------------
    query_embedding = create_embedding(query)

    async with SessionLocal() as session:

        # -----------------------------------------------------
        # 2. Calculate cosine distance using pgvector
        # -----------------------------------------------------
        distance = KnowledgeChunk.embedding.cosine_distance(
            query_embedding
        )

        # -----------------------------------------------------
        # 3. Build the semantic search query
        # -----------------------------------------------------
        statement = (
            select(
                KnowledgeChunk,
                distance.label("distance")
            )
            .order_by(distance)
            .limit(top_k)
        )

        # -----------------------------------------------------
        # 4. Optionally search only one chunking strategy
        # -----------------------------------------------------
        if strategy:
            statement = statement.where(
                KnowledgeChunk.strategy == strategy
            )

        # -----------------------------------------------------
        # 5. Execute search
        # -----------------------------------------------------
        result = await session.execute(statement)

        rows = result.all()

    # ---------------------------------------------------------
    # 6. Format the retrieved chunks
    # ---------------------------------------------------------
    results = []

    for chunk, distance_value in rows:

        distance_value = float(distance_value)

        results.append({
            "chunk_id": chunk.chunk_id,
            "strategy": chunk.strategy,
            "section": chunk.section,
            "content": chunk.content,
            "distance": distance_value,
            "similarity": 1 - distance_value,
        })

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
