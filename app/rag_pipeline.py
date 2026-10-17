
import asyncio
import os
import json
import csv
import time
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
from groq import Groq

from app.retrieval.semantic_search import semantic_search
from app.retrieval.reranker import rerank_results
from app.retrieval.context_expansion import expand_context
from app.retrieval.context_selection import select_context


# ============================================================
# ENVIRONMENT AND GROQ CONFIGURATION
# ============================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY was not found. "
        "Make sure it is present in your .env file."
    )

GROQ_MODEL = "openai/gpt-oss-20b"
groq_client = Groq(api_key=GROQ_API_KEY)


# ============================================================
# RAG CONFIGURATION
# ============================================================

STRATEGIES = [
    "recursive",
    "structure_aware",
    "contextual",
]

SEMANTIC_TOP_K = 15
RERANK_TOP_K = 5
CONTEXT_WINDOW = 2
MAX_CONTEXT_CHUNKS = 10

# Always save relative to the project root, not the terminal folder.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "evaluation"

JSON_PATH = OUTPUT_DIR / "batch_results.json"
CSV_PATH = OUTPUT_DIR / "batch_results.csv"


# ============================================================
# HELPERS FOR SAVING RESULTS
# ============================================================

def make_json_safe(value):
    """Convert nested results into values JSON can serialize."""

    if isinstance(value, dict):
        return {
            str(key): make_json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]

    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    return str(value)


def save_results(all_results: dict, errors: list) -> None:
    """Save all completed results to JSON and CSV."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # JSON: retain detailed results, including context and chunk metadata.
    json_data = {
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "model": GROQ_MODEL,
        "strategies": STRATEGIES,
        "configuration": {
            "semantic_top_k": SEMANTIC_TOP_K,
            "rerank_top_k": RERANK_TOP_K,
            "context_window": CONTEXT_WINDOW,
            "max_context_chunks": MAX_CONTEXT_CHUNKS,
        },
        "completed_questions": len(all_results),
        "results": make_json_safe(all_results),
        "errors": errors,
    }

    # Write to a temporary file first, then replace the previous file.
    temp_json_path = JSON_PATH.with_suffix(".json.tmp")

    with open(temp_json_path, "w", encoding="utf-8") as file:
        json.dump(
            json_data,
            file,
            indent=2,
            ensure_ascii=False,
        )

    temp_json_path.replace(JSON_PATH)

    # CSV: one row per completed question and strategy.
    csv_rows = []

    for query, strategy_results in all_results.items():
        for strategy in STRATEGIES:
            result = strategy_results.get(strategy)

            if not result:
                continue

            csv_rows.append({
                "question": query,
                "strategy": strategy,
                "answer": result.get("answer", ""),
                "duration_seconds": result.get("duration_seconds", ""),
                "semantic_results_count": len(
                    result.get("semantic_results", [])
                ),
                "reranked_results_count": len(
                    result.get("reranked_results", [])
                ),
                "expanded_chunks_count": len(
                    result.get("expanded_chunks", [])
                ),
                "selected_chunks_count": len(
                    result.get("selected_chunks", [])
                ),
            })

    csv_fields = [
        "question",
        "strategy",
        "answer",
        "duration_seconds",
        "semantic_results_count",
        "reranked_results_count",
        "expanded_chunks_count",
        "selected_chunks_count",
    ]

    temp_csv_path = CSV_PATH.with_suffix(".csv.tmp")

    with open(
        temp_csv_path,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(csv_rows)

    temp_csv_path.replace(CSV_PATH)

    print("\n[SAVED] Results updated:")
    print(f"        JSON: {JSON_PATH}")
    print(f"        CSV : {CSV_PATH}")


# ============================================================
# BUILD CONTEXT
# ============================================================

def build_context(chunks: list[dict]) -> str:
    """Convert selected chunks into an LLM-ready context."""

    if not chunks:
        return "No relevant context was retrieved."

    context_parts = []

    for index, chunk in enumerate(chunks, start=1):
        metadata = [
            f"Chunk: {chunk.get('chunk_id', 'unknown')}",
            f"Strategy: {chunk.get('strategy', 'unknown')}",
        ]

        section = chunk.get("section")
        page_number = chunk.get("page_number")

        if section:
            metadata.append(f"Section: {section}")

        if page_number:
            metadata.append(f"Page: {page_number}")

        context_parts.append(
            f"[CONTEXT {index}]\n"
            f"{' | '.join(metadata)}\n"
            f"{chunk.get('content', '')}"
        )

    return "\n\n".join(context_parts)


# ============================================================
# GENERATE ANSWER USING GROQ
# ============================================================

async def generate_answer(
    query: str,
    context: str,
    structured: bool = False,
    retry_feedback: str | None = None,
) -> str:
    """Generate a grounded answer, optionally as evidence-citing JSON."""

    rules = [
        "Answer using ONLY the supplied document context.",
        "Do not use outside knowledge or invent facts.",
        "Preserve numerical values and units exactly.",
        "If evidence is insufficient, explicitly say so.",
    ]
    if structured:
        rules.extend([
            "Return ONLY a valid JSON object, without markdown fences.",
            "Include these five fields exactly: answer, product_or_model, "
            "supporting_chunk_ids, insufficient_evidence, supporting_evidence.",
            "answer must be a string; product_or_model must be a string or null.",
            "supporting_chunk_ids must be an array of IDs actually shown in the context.",
            "insufficient_evidence must be a JSON boolean.",
            "supporting_evidence must be an array of objects with chunk_id and quote.",
            "Every quote must be an exact substring of the corresponding chunk content.",
            "If the context is insufficient, use an answer starting with "
            "'Insufficient evidence', set insufficient_evidence to true, "
            "and use empty arrays for supporting_chunk_ids and supporting_evidence.",
            "Keep the answer concise and avoid unsupported numerical claims.",
        ])
    if retry_feedback:
        rules.append(f"Correct the previous response: {retry_feedback}")

    prompt = (
        "INSTRUCTIONS:\n- " + "\n- ".join(rules)
        + f"\n\nUSER QUESTION:\n{query}"
        + f"\n\nRETRIEVED CONTEXT:\n{context}"
        + "\n\nANSWER:\n"
    )
    completion_args = {
        "model": GROQ_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "You are a document question-answering assistant. "
                           "Answer strictly using the supplied context.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
        "max_completion_tokens": 2048,
        "include_reasoning": False,
    }

    for attempt in range(3):
        try:
            response = await asyncio.to_thread(
                groq_client.chat.completions.create, **completion_args
            )
            if not response.choices:
                raise RuntimeError("The LLM returned no choices")
            choice = response.choices[0]
            answer = choice.message.content
            if answer and answer.strip():
                return answer.strip()
            error = RuntimeError(
                "The LLM did not return an answer. "
                f"finish_reason={choice.finish_reason}"
            )
            if attempt == 2:
                raise error
        except Exception as error:
            status_code = getattr(error, "status_code", None)
            transient = (
                status_code == 429
                or (isinstance(status_code, int) and status_code >= 500)
                or type(error).__name__ in {
                    "APITimeoutError", "APIConnectionError", "RateLimitError",
                    "InternalServerError", "TimeoutException",
                }
            )
            if not transient or attempt == 2:
                raise
        await asyncio.sleep(0.5 * (attempt + 1))

    raise RuntimeError("The LLM did not return an answer after retries")


# ============================================================
# RUN ONE CHUNKING STRATEGY
# ============================================================

async def run_strategy(
    query: str,
    strategy: str,
    semantic_top_k: int = SEMANTIC_TOP_K,
    rerank_top_k: int = RERANK_TOP_K,
    context_window: int = CONTEXT_WINDOW,
) -> dict:
    """Execute the complete RAG pipeline for one strategy."""

    print("\n" + "=" * 80)
    print(f"RUNNING STRATEGY: {strategy}")
    print("=" * 80)

    start_time = time.perf_counter()

    # 1. Semantic Search
    results = await semantic_search(
        query=query,
        top_k=semantic_top_k,
        strategy=strategy,
    )

    print(f"Semantic candidates retrieved: {len(results)}")

    # 2. Cross-Encoder Reranking
    reranked_results = rerank_results(
        query=query,
        results=results,
        top_k=rerank_top_k,
    )

    print(f"Reranked candidates: {len(reranked_results)}")

    # 3. Context Expansion
    expanded_chunks = await expand_context(
        retrieved_chunks=reranked_results,
        window=context_window,
    )

    print(f"Expanded context chunks: {len(expanded_chunks)}")

    # 4. Context Selection
    selected_chunks = select_context(
        chunks=expanded_chunks,
        max_chunks=MAX_CONTEXT_CHUNKS,
    )

    print(f"Selected context chunks: {len(selected_chunks)}")

    # 5. Build Context
    context = build_context(selected_chunks)

    # 6. Generate Answer
    answer = await generate_answer(
        query=query,
        context=context,
    )

    duration = time.perf_counter() - start_time

    print(f"Strategy duration: {duration:.2f} seconds")

    return {
        "strategy": strategy,
        "query": query,
        "semantic_results": results,
        "reranked_results": reranked_results,
        "expanded_chunks": expanded_chunks,
        "selected_chunks": selected_chunks,
        "context": context,
        "answer": answer,
        "duration_seconds": round(duration, 3),
    }


# ============================================================
# RUN ALL THREE STRATEGIES
# ============================================================

async def run_all_strategies(
    query: str,
    semantic_top_k: int = SEMANTIC_TOP_K,
    rerank_top_k: int = RERANK_TOP_K,
    context_window: int = CONTEXT_WINDOW,
) -> dict:
    """Run the pipeline independently for every chunking strategy."""

    all_results = {}

    for strategy in STRATEGIES:
        result = await run_strategy(
            query=query,
            strategy=strategy,
            semantic_top_k=semantic_top_k,
            rerank_top_k=rerank_top_k,
            context_window=context_window,
        )

        all_results[strategy] = result

    return all_results


# ============================================================
# DISPLAY FINAL RESULTS
# ============================================================

def display_results(results: dict) -> None:
    """Display retrieval counts and final answers."""

    print("\n\n" + "=" * 80)
    print("FINAL RAG RESULTS")
    print("=" * 80)

    for strategy in STRATEGIES:
        result = results.get(strategy)

        if not result:
            continue

        print("\n" + "=" * 80)
        print(f"STRATEGY: {strategy}")
        print("=" * 80)

        print(f"\nQuery: {result['query']}")
        print(f"Semantic candidates: {len(result['semantic_results'])}")
        print(f"Reranked candidates: {len(result['reranked_results'])}")
        print(f"Expanded chunks: {len(result['expanded_chunks'])}")
        print(f"Selected chunks: {len(result['selected_chunks'])}")

        print("\nFINAL ANSWER")
        print("-" * 80)
        print(result["answer"])
        print("-" * 80)


# ============================================================
# BATCH TESTING
# ============================================================

async def main():
    print("=" * 80)
    print("DOCUMENT RAG PIPELINE - BATCH TESTING")
    print("=" * 80)

    print(f"LLM: Groq - {GROQ_MODEL}")
    print("Strategies: " + ", ".join(STRATEGIES))
    print(f"Output folder: {OUTPUT_DIR}")

    print("\nEnter your questions, one per line.")
    print("Press Enter on an empty line to start testing.\n")

    queries = []

    while True:
        query = input(f"Question {len(queries) + 1}: ").strip()

        if not query:
            break

        queries.append(query)

    if not queries:
        print("No questions entered. Exiting.")
        return

    print(f"\nTotal questions to test: {len(queries)}")
    print(f"Maximum strategy runs: {len(queries) * len(STRATEGIES)}")

    all_results = {}
    errors = []

    # Create output files immediately, before the lengthy batch begins.
    save_results(all_results, errors)

    for index, query in enumerate(queries, start=1):
        print("\n\n" + "#" * 80)
        print(f"QUESTION {index}/{len(queries)}: {query}")
        print("#" * 80)

        all_results[query] = {}

        for strategy in STRATEGIES:
            try:
                result = await run_strategy(
                    query=query,
                    strategy=strategy,
                )

                all_results[query][strategy] = result

                # Save immediately after each successful strategy.
                save_results(all_results, errors)

            except Exception as error:
                error_message = (
                    f"Question {index}, strategy '{strategy}': "
                    f"{type(error).__name__}: {error}"
                )

                print(f"\n[ERROR] {error_message}")

                errors.append({
                    "question": query,
                    "strategy": strategy,
                    "error": error_message,
                })

                # Persist the error and any completed answers.
                try:
                    save_results(all_results, errors)
                except Exception as save_error:
                    print(f"[SAVE ERROR] {save_error}")

                print("Continuing with the next strategy.")

        # Show whichever strategies completed successfully.
        display_results(all_results[query])

        # Save again after finishing this question.
        try:
            save_results(all_results, errors)
        except Exception as save_error:
            print(f"[SAVE ERROR] {save_error}")

    print("\n" + "=" * 80)
    print("BATCH TESTING COMPLETED")
    print(f"Questions submitted: {len(queries)}")
    print(f"Questions with saved results: {len(all_results)}")
    print(f"Errors recorded: {len(errors)}")
    print(f"JSON file: {JSON_PATH}")
    print(f"CSV file: {CSV_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())