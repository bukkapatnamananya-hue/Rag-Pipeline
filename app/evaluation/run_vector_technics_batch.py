import argparse
import asyncio
import csv
import json
import re
import time
from datetime import datetime
from pathlib import Path

from app.rag_pipeline import GROQ_MODEL, build_context, generate_answer
from app.retrieval.context_expansion import expand_context
from app.retrieval.reranker import rerank_results
from app.retrieval.semantic_search import semantic_search


PROJECT_ROOT = Path(__file__).resolve().parents[2]
QUERY_PATH = PROJECT_ROOT / "data" / "evaluation" / "evaluation_queries.json"
OUTPUT_DIR = PROJECT_ROOT / "data" / "evaluation"
JSON_PATH = OUTPUT_DIR / "vector_technics_answers.json"
CSV_PATH = OUTPUT_DIR / "vector_technics_answers.csv"
STRATEGIES = ["recursive", "structure_aware", "contextual"]
SEMANTIC_TOP_K = 15
RERANK_TOP_K = 5
CONTEXT_WINDOW = 2
MAX_CONTEXT_CHUNKS = 25
VALIDATION_CASES = (
    ("q2", "contextual", "descriptive"),
    ("q6", "recursive", "numerical specification"),
    ("q14", "recursive", "technical test data"),
)
REQUIRED_RESPONSE_FIELDS = {
    "answer",
    "product_or_model",
    "supporting_chunk_ids",
    "insufficient_evidence",
    "supporting_evidence",
}
ANSWER_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "by", "for", "from", "in",
    "is", "it", "of", "on", "or", "the", "to", "was", "were", "with",
}
CSV_FIELDS = [
    "query_id",
    "question",
    "strategy",
    "generated_answer",
    "status",
    "product_or_model",
    "supporting_chunk_ids",
    "insufficient_evidence",
    "retrieved_chunk_count",
    "reranked_chunk_count",
    "expanded_context_chunk_ids",
    "evidence_check",
    "retrieval_time_seconds",
    "reranking_time_seconds",
    "context_expansion_time_seconds",
    "llm_generation_time_seconds",
    "total_processing_time_seconds",
    "errors",
]


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def result_key(result):
    return result["query_id"], result["strategy"]


def load_existing_results():
    if not JSON_PATH.exists():
        return {}
    try:
        with JSON_PATH.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        return {
            result_key(result): result
            for result in payload.get("results", [])
            if isinstance(result, dict)
            and result.get("query_id")
            and result.get("strategy")
        }
    except (OSError, json.JSONDecodeError, TypeError):
        return {}


def save_results(results, validation):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ordered_results = sorted(
        results.values(),
        key=lambda result: (
            result.get("query_id", ""),
            STRATEGIES.index(result["strategy"])
            if result.get("strategy") in STRATEGIES
            else len(STRATEGIES),
        ),
    )
    payload = {
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "document": "VectorTechnics Catalogue 2026",
        "model": GROQ_MODEL,
        "configuration": {
            "semantic_top_k": SEMANTIC_TOP_K,
            "rerank_top_k": RERANK_TOP_K,
            "context_window": CONTEXT_WINDOW,
            "max_context_chunks": MAX_CONTEXT_CHUNKS,
        },
        "validation": json_safe(validation),
        "completed_runs": len(ordered_results),
        "results": json_safe(ordered_results),
    }

    temporary_json = JSON_PATH.with_suffix(".json.tmp")
    with temporary_json.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
    temporary_json.replace(JSON_PATH)

    temporary_csv = CSV_PATH.with_suffix(".csv.tmp")
    with temporary_csv.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for result in ordered_results:
            writer.writerow({
                "query_id": result.get("query_id", ""),
                "question": result.get("question", ""),
                "strategy": result.get("strategy", ""),
                "generated_answer": result.get("generated_answer", ""),
                "status": result.get("status", ""),
                "evidence_check": json.dumps(result.get("evidence_check", {}), ensure_ascii=False),
                "product_or_model": result.get("product_or_model") or "",
                "supporting_chunk_ids": "; ".join(
                    result.get("supporting_chunk_ids", [])
                ),
                "insufficient_evidence": result.get("insufficient_evidence", ""),
                "retrieved_chunk_count": len(
                    result.get("top_15_retrieved_chunks", [])
                ),
                "reranked_chunk_count": len(
                    result.get("top_5_reranked_chunks", [])
                ),
                "expanded_context_chunk_ids": "; ".join(
                    result.get("expanded_context_chunk_ids", [])
                ),
                "retrieval_time_seconds": result.get("retrieval_time_seconds", ""),
                "reranking_time_seconds": result.get("reranking_time_seconds", ""),
                "context_expansion_time_seconds": result.get(
                    "context_expansion_time_seconds", ""
                ),
                "llm_generation_time_seconds": result.get(
                    "llm_generation_time_seconds", ""
                ),
                "total_processing_time_seconds": result.get(
                    "total_processing_time_seconds", ""
                ),
                "errors": "; ".join(result.get("errors", [])),
            })
    temporary_csv.replace(CSV_PATH)


def normalize_text(value):
    return " ".join(str(value).casefold().split())


def answer_terms(value):
    return {
        token.casefold()
        for token in re.findall(r"[\w.%-]+", str(value))
        if token.casefold() not in ANSWER_STOP_WORDS
    }


def extract_json_object(raw_answer):
    answer_text = str(raw_answer or "").strip()
    answer_text = re.sub(r"^\s*```(?:json)?\s*", "", answer_text, flags=re.I)
    answer_text = re.sub(r"\s*```\s*$", "", answer_text)
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", answer_text):
        try:
            payload, _ = decoder.raw_decode(answer_text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise ValueError("No valid JSON object found in the LLM response")


def parse_answer(raw_answer, evidence_chunks):
    insufficient_answer = "Insufficient evidence to answer from the retrieved context."
    try:
        payload = extract_json_object(raw_answer)
    except (TypeError, ValueError) as error:
        return insufficient_answer, None, [], True, [], str(error)

    missing_fields = REQUIRED_RESPONSE_FIELDS - payload.keys()
    if missing_fields:
        return (
            insufficient_answer,
            None,
            [],
            True,
            [],
            f"LLM response is missing required fields: {', '.join(sorted(missing_fields))}",
        )

    answer = payload["answer"].strip() if isinstance(payload["answer"], str) else ""
    is_insufficient = payload["insufficient_evidence"] is True
    if not answer:
        return insufficient_answer, None, [], True, [], "LLM response has an empty answer"
    if is_insufficient:
        if "insufficient evidence" not in answer.casefold():
            answer = f"Insufficient evidence: {answer}"
        return answer, None, [], True, [], None

    available_chunks = {
        str(chunk["chunk_id"]): chunk
        for chunk in evidence_chunks
        if chunk.get("chunk_id") is not None
    }
    requested_ids = payload["supporting_chunk_ids"]
    if not isinstance(requested_ids, list):
        return insufficient_answer, None, [], True, [], "supporting_chunk_ids must be a list"

    quote_records = payload["supporting_evidence"]
    if not isinstance(quote_records, list):
        return insufficient_answer, None, [], True, [], "supporting_evidence must be a list"

    verified_quotes = []
    verified_ids = set()
    for record in quote_records:
        if not isinstance(record, dict):
            continue
        chunk_id = str(record.get("chunk_id", ""))
        quote = record.get("quote")
        chunk = available_chunks.get(chunk_id)
        if not chunk or not isinstance(quote, str) or not quote.strip():
            continue
        if normalize_text(quote) not in normalize_text(chunk.get("content", "")):
            continue
        verified_ids.add(chunk_id)
        verified_quotes.append({"chunk_id": chunk_id, "quote": quote.strip()})

    supporting_ids = list(dict.fromkeys(
        str(chunk_id)
        for chunk_id in requested_ids
        if str(chunk_id) in verified_ids and str(chunk_id) in available_chunks
    ))
    quote_text = " ".join(
        quote["quote"] for quote in verified_quotes
        if quote["chunk_id"] in supporting_ids
    )
    supported_terms = answer_terms(quote_text)
    answer_claim_terms = answer_terms(answer)
    product_or_model = payload["product_or_model"]
    if product_or_model is not None:
        if not isinstance(product_or_model, str):
            return insufficient_answer, None, [], True, [], "product_or_model must be a string or null"
        product_or_model = product_or_model.strip() or None
        if product_or_model and not answer_terms(product_or_model).issubset(supported_terms):
            return insufficient_answer, None, [], True, [], "product_or_model is not present in verified evidence"

    if not supporting_ids:
        return insufficient_answer, None, [], True, [], "No supporting chunk IDs have a verifiable evidence quote"
    unsupported_terms = answer_claim_terms - supported_terms
    if unsupported_terms:
        return (
            insufficient_answer,
            None,
            [],
            True,
            [],
            "Answer contains terms not present in its cited evidence: "
            + ", ".join(sorted(unsupported_terms)),
        )

    return answer, product_or_model, supporting_ids, False, verified_quotes, None


async def run_case(query_record, strategy):
    query_id = query_record["id"]
    question = query_record["query"]
    result = {
        "query_id": query_id,
        "question": question,
        "strategy": strategy,
        "top_15_retrieved_chunks": [],
        "top_5_reranked_chunks": [],
        "expanded_context_chunk_ids": [],
        "expanded_context_chunks": [],
        "generated_answer": "",
        "llm_response_raw": "",
        "supporting_evidence": [],
        "product_or_model": None,
        "supporting_chunk_ids": [],
        "retrieval_time_seconds": 0.0,
        "reranking_time_seconds": 0.0,
        "context_expansion_time_seconds": 0.0,
        "llm_generation_time_seconds": 0.0,
        "total_processing_time_seconds": 0.0,
        "insufficient_evidence": False,
        "status": "pending",
        "evidence_check": {"passed": False, "reason": "Not validated yet"},
        "errors": [],
    }
    total_start = time.perf_counter()
    print(f"[{query_id} | {strategy}] retrieval", flush=True)

    stage_start = time.perf_counter()
    try:
        retrieved = await semantic_search(
            question,
            top_k=SEMANTIC_TOP_K,
            strategy=strategy,
        )
        result["top_15_retrieved_chunks"] = retrieved
    except Exception as error:
        result["errors"].append(f"semantic_search: {type(error).__name__}: {error}")
        retrieved = []
    result["retrieval_time_seconds"] = round(
        time.perf_counter() - stage_start, 3
    )

    print(f"[{query_id} | {strategy}] reranking", flush=True)
    stage_start = time.perf_counter()
    try:
        reranked = rerank_results(
            query=question,
            results=retrieved,
            top_k=RERANK_TOP_K,
        )
        result["top_5_reranked_chunks"] = reranked
    except Exception as error:
        result["errors"].append(f"reranking: {type(error).__name__}: {error}")
        reranked = []
    result["reranking_time_seconds"] = round(
        time.perf_counter() - stage_start, 3
    )

    print(f"[{query_id} | {strategy}] context expansion", flush=True)
    stage_start = time.perf_counter()
    try:
        expanded = await expand_context(reranked, window=CONTEXT_WINDOW)
        result["expanded_context_chunks"] = expanded
        result["expanded_context_chunk_ids"] = [
            chunk["chunk_id"] for chunk in expanded
        ]
    except Exception as error:
        result["errors"].append(
            f"context_expansion: {type(error).__name__}: {error}"
        )
        expanded = []
    result["context_expansion_time_seconds"] = round(
        time.perf_counter() - stage_start, 3
    )

    reranked_ids = {
        (chunk.get("document_id"), chunk["strategy"], chunk["chunk_id"])
        for chunk in reranked
    }
    supporting_neighbors = [
        chunk
        for chunk in expanded
        if (chunk.get("document_id"), chunk["strategy"], chunk["chunk_id"])
        not in reranked_ids
    ]
    evidence_chunks = []
    seen_evidence = set()
    for chunk in reranked[:RERANK_TOP_K] + supporting_neighbors[
        :max(0, MAX_CONTEXT_CHUNKS - len(reranked[:RERANK_TOP_K]))
    ]:
        key = (chunk.get("document_id"), chunk["strategy"], chunk["chunk_id"])
        if key not in seen_evidence:
            evidence_chunks.append(chunk)
            seen_evidence.add(key)
    evidence_ids = [chunk["chunk_id"] for chunk in evidence_chunks]
    context = (
        "TOP RERANKED CHUNKS\n"
        + build_context(reranked[:RERANK_TOP_K])
        + "\n\nEXPANDED SUPPORTING CONTEXT\n"
        + build_context(supporting_neighbors[
            :max(0, MAX_CONTEXT_CHUNKS - len(reranked[:RERANK_TOP_K]))
        ])
    )

    print(f"[{query_id} | {strategy}] answer generation", flush=True)
    stage_start = time.perf_counter()
    if not evidence_chunks:
        result["generated_answer"] = (
            "Insufficient evidence: no supporting context was retrieved."
        )
        result["insufficient_evidence"] = True
        result["status"] = "insufficient_evidence"
        result["evidence_check"] = {
            "passed": False,
            "reason": "No retrieved or expanded evidence was available",
        }
    else:
        retry_feedback = None
        parsed = None
        try:
            for attempt in range(2):
                raw_answer = await generate_answer(
                    question,
                    context,
                    structured=True,
                    retry_feedback=retry_feedback,
                )
                result["llm_response_raw"] = raw_answer
                parsed = parse_answer(raw_answer, evidence_chunks)
                if parsed[5] is None:
                    break
                retry_feedback = (
                    f"The previous response failed evidence validation: {parsed[5]}. "
                    "Return the required JSON fields, quote exact text from supplied "
                    "chunks, and do not include any unsupported answer claims."
                )

            (
                answer,
                product_or_model,
                supporting_ids,
                insufficient,
                supporting_evidence,
                response_error,
            ) = parsed
            result["generated_answer"] = answer
            result["product_or_model"] = product_or_model
            result["supporting_chunk_ids"] = supporting_ids
            result["supporting_evidence"] = supporting_evidence
            result["insufficient_evidence"] = insufficient
            if response_error:
                result["errors"].append(f"answer_validation: {response_error}")
                result["status"] = "failed"
                result["evidence_check"] = {
                    "passed": False,
                    "reason": response_error,
                }
            elif insufficient:
                result["status"] = "insufficient_evidence"
                result["evidence_check"] = {
                    "passed": False,
                    "reason": "The model explicitly abstained",
                }
            else:
                result["status"] = "success"
                result["evidence_check"] = {
                    "passed": verify_result_evidence(result),
                    "method": "exact source quote plus answer-term coverage",
                }
                if not result["evidence_check"]["passed"]:
                    result["errors"].append(
                        "answer_validation: saved answer failed evidence verification"
                    )
                    result["status"] = "failed"
        except Exception as error:
            result["errors"].append(
                f"llm_generation: {type(error).__name__}: {error}"
            )
            result["generated_answer"] = (
                "Answer unavailable because LLM generation failed."
            )
            result["status"] = "failed"
    result["llm_generation_time_seconds"] = round(
        time.perf_counter() - stage_start, 3
    )
    result["total_processing_time_seconds"] = round(
        time.perf_counter() - total_start, 3
    )
    print(
        f"[{query_id} | {strategy}] done in "
        f"{result['total_processing_time_seconds']:.2f}s; "
        f"retrieved={len(retrieved)}, reranked={len(reranked)}, "
        f"expanded={len(expanded)}, errors={len(result['errors'])}",
        flush=True,
    )
    return result


async def validate_missing_evidence():
    test_question = "What is the maximum continuous power of the 9999 ZZ 999KV motor?"
    raw_answer = await generate_answer(
        test_question,
        "No retrieved chunks contain evidence relevant to this question.",
        structured=True,
    )
    answer, _, supporting_ids, insufficient, _, response_error = parse_answer(
        raw_answer,
        [],
    )
    if response_error:
        raise RuntimeError(f"Empty-evidence response was invalid: {response_error}")
    if not insufficient or "insufficient evidence" not in answer.casefold():
        raise RuntimeError(
            "Empty-evidence validation failed: the LLM did not explicitly "
            "report insufficient evidence."
        )
    if supporting_ids:
        raise RuntimeError(
            "Empty-evidence validation failed: the LLM cited unavailable chunks."
        )
    print("[validation] empty evidence correctly produced an insufficient-evidence answer")
    return {
        "passed": True,
        "question": test_question,
        "answer": answer,
        "supporting_chunk_ids": supporting_ids,
        "insufficient_evidence": insufficient,
    }


def verify_result_evidence(result):
    source_chunks = (
        result.get("top_5_reranked_chunks", [])
        + result.get("expanded_context_chunks", [])
    )
    source_by_id = {
        str(chunk["chunk_id"]): chunk
        for chunk in source_chunks
        if chunk.get("chunk_id") is not None
    }
    supporting_ids = result.get("supporting_chunk_ids", [])
    if not supporting_ids or not set(supporting_ids).issubset(source_by_id):
        return False

    verified_quotes = []
    verified_ids = set()
    for record in result.get("supporting_evidence", []):
        if not isinstance(record, dict):
            continue
        chunk_id = str(record.get("chunk_id", ""))
        quote = record.get("quote")
        chunk = source_by_id.get(chunk_id)
        if (
            chunk
            and isinstance(quote, str)
            and normalize_text(quote) in normalize_text(chunk.get("content", ""))
        ):
            verified_ids.add(chunk_id)
            verified_quotes.append(quote)

    if not set(supporting_ids).issubset(verified_ids):
        return False
    evidence_terms = answer_terms(" ".join(verified_quotes))
    answer_and_model_terms = answer_terms(result.get("generated_answer", ""))
    if result.get("product_or_model"):
        answer_and_model_terms.update(answer_terms(result["product_or_model"]))
    return answer_and_model_terms.issubset(evidence_terms)


def retain_previous_attempt(result, previous):
    if not previous:
        return result
    history = list(previous.get("previous_attempts", []))
    history.append({
        "status": previous.get("status"),
        "generated_answer": previous.get("generated_answer", ""),
        "errors": previous.get("errors", []),
        "top_15_retrieved_chunks": previous.get("top_15_retrieved_chunks", []),
        "top_5_reranked_chunks": previous.get("top_5_reranked_chunks", []),
        "expanded_context_chunks": previous.get("expanded_context_chunks", []),
    })
    result["previous_attempts"] = history
    return result


async def main(validation_only=False):
    with QUERY_PATH.open("r", encoding="utf-8") as file:
        query_records = json.load(file).get("queries", [])
    if len(query_records) != 14:
        raise ValueError(f"Expected 14 VectorTechnics queries, found {len(query_records)}")

    results = load_existing_results()
    validation = {"representative_cases": [], "empty_evidence": None}
    started_at = time.perf_counter()
    print(
        f"VectorTechnics batch | model={GROQ_MODEL} | "
        f"queries={len(query_records)} | strategies={len(STRATEGIES)}",
        flush=True,
    )

    if validation_only:
        # Validation is informational and never blocks a full batch.
        for query_id, strategy, case_type in VALIDATION_CASES:
            query_record = next(item for item in query_records if item["id"] == query_id)
            key = (query_id, strategy)
            result = results.get(key)
            if result is None:
                result = await run_case(query_record, strategy)
                results[key] = result
                save_results(results, validation)

            passed = (
                result.get("status") == "success"
                and not result.get("errors")
                and bool(result.get("top_15_retrieved_chunks"))
                and bool(result.get("top_5_reranked_chunks"))
                and verify_result_evidence(result)
            )
            validation["representative_cases"].append({
                "query_id": query_id,
                "strategy": strategy,
                "case_type": case_type,
                "passed": passed,
                "status": result.get("status"),
                "supporting_chunk_ids": result.get("supporting_chunk_ids", []),
                "evidence_check": result.get("evidence_check"),
                "answer": result.get("generated_answer", ""),
            })
            print(f"[validation] {query_id}/{strategy}: "
                  f"{'passed' if passed else 'FAILED'}", flush=True)
            save_results(results, validation)

        try:
            validation["empty_evidence"] = await validate_missing_evidence()
        except Exception as error:
            validation["empty_evidence"] = {
                "passed": False,
                "error": f"{type(error).__name__}: {error}",
            }
        save_results(results, validation)
        print("Validation-only mode complete; failures were recorded.")
        return

    # Full batch: do not rerun any existing case, including failed ones.
    # Failed API calls and insufficient-evidence answers remain in the report.
    for query_record in query_records:
        for strategy in STRATEGIES:
            key = (query_record["id"], strategy)
            if key in results:
                print(f"[skip] {key[0]}/{strategy}: "
                      f"already saved ({results[key].get('status')})", flush=True)
                continue
            try:
                result = await run_case(query_record, strategy)
            except Exception as error:
                result = {
                    "query_id": query_record["id"],
                    "question": query_record["query"],
                    "strategy": strategy,
                    "status": "failed",
                    "generated_answer": "",
                    "insufficient_evidence": False,
                    "errors": [f"unhandled: {type(error).__name__}: {error}"],
                }
                print(f"[error] {key[0]}/{strategy}: {error}", flush=True)
            results[key] = result
            save_results(results, validation)

    elapsed = time.perf_counter() - started_at
    successful = sum(item.get("status") == "success" for item in results.values())
    insufficient = sum(
        item.get("status") == "insufficient_evidence" for item in results.values()
    )
    failed = len(results) - successful - insufficient
    print("\nVectorTechnics batch summary")
    print(f"Runs saved: {len(results)} / {len(query_records) * len(STRATEGIES)}")
    print(f"Successful runs: {successful}; insufficient evidence: {insufficient}; "
          f"failed runs: {failed}")
    print(f"Wall time this execution: {elapsed:.2f} seconds")
    for field in (
        "retrieval_time_seconds", "reranking_time_seconds",
        "context_expansion_time_seconds", "llm_generation_time_seconds",
        "total_processing_time_seconds",
    ):
        print(f"Total {field}: "
              f"{sum(item.get(field, 0) or 0 for item in results.values()):.2f}s")
    print(f"JSON: {JSON_PATH}")
    print(f"CSV:  {CSV_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run VectorTechnics retrieval, reranking, context expansion, and answer generation."
    )
    parser.add_argument(
        "--validation-only",
        action="store_true",
        help="Run representative retrieval and empty-evidence checks without the full batch.",
    )
    args = parser.parse_args()
    asyncio.run(main(validation_only=args.validation_only))
