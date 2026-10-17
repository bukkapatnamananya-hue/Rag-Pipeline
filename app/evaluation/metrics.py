from __future__ import annotations

from collections.abc import Iterable


def recall_at_k(retrieved_ids: Iterable[str], relevant_ids: Iterable[str], k: int = 5) -> float | None:
    retrieved = list(retrieved_ids)[:k]
    relevant = list(relevant_ids)
    if not relevant:
        return None
    relevant_set = set(relevant)
    hits = len(set(retrieved) & relevant_set)
    return hits / len(relevant_set)


def reciprocal_rank(retrieved_ids: Iterable[str], relevant_ids: Iterable[str], k: int = 5) -> float:
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    for rank, chunk_id in enumerate(list(retrieved_ids)[:k], start=1):
        if chunk_id in relevant:
            return 1.0 / rank
    return 0.0


def hit_rate_at_k(retrieved_ids: Iterable[str], relevant_ids: Iterable[str], k: int = 5) -> int:
    retrieved = set(list(retrieved_ids)[:k])
    relevant = set(relevant_ids)
    return 1 if retrieved & relevant else 0


def mean(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None
