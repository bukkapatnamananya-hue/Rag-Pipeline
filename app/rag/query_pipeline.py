from __future__ import annotations

from typing import Any

from app.retrieval.reranker import rerank_results
from app.retrieval.semantic_search import semantic_search


async def run_query_pipeline(query: str, strategy: str, top_k: int = 15, rerank_top_k: int = 5):
    results = await semantic_search(query=query, top_k=top_k, strategy=strategy)
    reranked = rerank_results(query=query, results=results, top_k=rerank_top_k)
    return {"semantic_results": results, "reranked_results": reranked}
