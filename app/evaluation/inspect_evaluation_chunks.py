import asyncio
import json
from pathlib import Path

from app.retrieval.semantic_search import semantic_search


QUERIES_PATH = Path(
    "data/evaluation/evaluation_queries.json"
)

STRATEGIES = [
    "recursive",
    "structure_aware",
    "contextual",
]

TOP_K = 10


def load_queries():
    """Load evaluation queries from JSON."""

    with open(
        QUERIES_PATH,
        "r",
        encoding="utf-8"
    ) as file:
        data = json.load(file)

    return data["queries"]


async def inspect_chunks():

    queries = load_queries()

    print()
    print("=" * 80)
    print("EVALUATION CHUNK INSPECTION")
    print("=" * 80)

    for query_data in queries:

        query_id = query_data["id"]
        query = query_data["query"]

        print()
        print("=" * 80)
        print(f"{query_id}: {query}")
        print("=" * 80)

        for strategy in STRATEGIES:

            print()
            print("-" * 80)
            print(f"STRATEGY: {strategy}")
            print("-" * 80)

            results = await semantic_search(
                query=query,
                strategy=strategy,
                top_k=TOP_K
            )

            for rank, result in enumerate(
                results,
                start=1
            ):

                chunk_id = result["chunk_id"]
                score = result.get(
                    "score",
                    result.get("similarity")
                )
                content = result["content"]

                print()
                print(
                    f"Rank {rank} | "
                    f"Chunk: {chunk_id} | "
                    f"Score: {score}"
                )

                print(
                    f"Content: {content}"
                )

    print()
    print("=" * 80)
    print("INSPECTION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(inspect_chunks())