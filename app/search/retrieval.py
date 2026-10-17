from __future__ import annotations

from typing import Any

from app.retrieval.semantic_search import semantic_search


def dense_retrieval(*args: Any, **kwargs: Any):
    return semantic_search(*args, **kwargs)


def sparse_retrieval(*args: Any, **kwargs: Any):
    return semantic_search(*args, **kwargs)


def hybrid_retrieval(*args: Any, **kwargs: Any):
    return semantic_search(*args, **kwargs)
