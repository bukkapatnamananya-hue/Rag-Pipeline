import asyncio
from sentence_transformers import CrossEncoder

from app.retrieval.semantic_search import semantic_search
from app.retrieval.context_expansion import expand_context


MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
reranker = CrossEncoder(MODEL_NAME)


def rerank_results(query: str, results: list[dict], top_k: int = 5):

    if not results:
        return []

    pairs = [(query, result["content"]) for result in results]

    scores = reranker.predict(pairs)

    reranked_results = []

    for result, score in zip(results, scores):
        result_copy = result.copy()
        result_copy["rerank_score"] = float(score)
        reranked_results.append(result_copy)

    reranked_results.sort(
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    return reranked_results[:top_k]


async def main():

    query = input("`nEnter your question: ").strip()

    strategies = [
        "recursive",
        "structure_aware",
        "contextual",
    ]

    for strategy in strategies:

        print("`n" + "=" * 80)
        print(f"STRATEGY: {strategy}")
        print("=" * 80)

        results = await semantic_search(
            query=query,
            top_k=15,
            strategy=strategy
        )

        print(f"`nSemantic candidates retrieved: {len(results)}")

        reranked = rerank_results(
            query=query,
            results=results,
            top_k=5
        )

        print("`nTOP 5 RERANKED RESULTS")
        print("=" * 80)

        for i, result in enumerate(reranked, start=1):

            print(f"`nRESULT {i}")
            print("-" * 80)
            print(f"Chunk ID       : {result['chunk_id']}")
            print(f"Strategy       : {result['strategy']}")
            print(f"Semantic Score : {result['similarity']:.4f}")
            print(f"Rerank Score   : {result['rerank_score']:.4f}")
            print(f"Section        : {result['section']}")
            print("`nContent:")
            print(result["content"])

        expanded_context = await expand_context(
            reranked,
            window=2
        )

        print("`nEXPANDED CONTEXT")
        print("=" * 80)
        print(f"`nExpanded chunks retrieved: {len(expanded_context)}")

        for i, chunk in enumerate(expanded_context, start=1):

            print(f"`nCONTEXT {i}")
            print("-" * 80)
            print(f"Chunk ID : {chunk['chunk_id']}")
            print(f"Strategy : {chunk['strategy']}")
            print(f"Section  : {chunk['section']}")
            print("`nContent:")
            print(chunk["content"])


if __name__ == "__main__":
    asyncio.run(main())
