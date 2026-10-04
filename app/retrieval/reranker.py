import asyncio

from sentence_transformers import CrossEncoder

from app.retrieval.semantic_search import semantic_search


# Cross-encoder reranking model
MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

reranker = CrossEncoder(MODEL_NAME)


def rerank_results(
    query: str,
    results: list[dict],
    top_k: int = 5
):
    """
    Rerank semantic-search results using a cross-encoder.

    The cross-encoder directly evaluates:
        query + chunk

    and produces a relevance score.
    """

    if not results:
        return []

    pairs = [
        (query, result["content"])
        for result in results
    ]

    scores = reranker.predict(pairs)

    reranked_results = []

    for result, score in zip(results, scores):
        result_copy = result.copy()
        result_copy["rerank_score"] = float(score)
        reranked_results.append(result_copy)

    # Highest score = most relevant
    reranked_results.sort(
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    return reranked_results[:top_k]


async def main():

    query = "What was Zen Technologies EBITDA in FY 2023-24?"

    print("\n" + "=" * 80)
    print("SEMANTIC RETRIEVAL + RERANKING")
    print("=" * 80)

    print(f"\nQuery: {query}")

    # Test each chunking strategy separately
    strategies = [
        "recursive",
        "structure_aware",
        "contextual",
    ]

    for strategy in strategies:

        print("\n" + "=" * 80)
        print(f"STRATEGY: {strategy}")
        print("=" * 80)

        # Step 1: Retrieve top 15 candidates using semantic search
        results = await semantic_search(
            query=query,
            top_k=15,
            strategy=strategy
        )

        print(
            f"\nSemantic candidates retrieved: {len(results)}"
        )

        # Step 2: Rerank the 15 candidates
        reranked = rerank_results(
            query=query,
            results=results,
            top_k=5
        )

        print("\nTOP 5 RERANKED RESULTS")
        print("=" * 80)

        # Step 3: Display the top 5 results
        for i, result in enumerate(reranked, start=1):

            print(f"\nRESULT {i}")
            print("-" * 80)

            print(f"Chunk ID       : {result['chunk_id']}")
            print(f"Strategy       : {result['strategy']}")
            print(f"Semantic Score : {result['similarity']:.4f}")
            print(f"Rerank Score   : {result['rerank_score']:.4f}")
            print(f"Section        : {result['section']}")

            print("\nContent:")
            print(result["content"])


if __name__ == "__main__":
    asyncio.run(main())