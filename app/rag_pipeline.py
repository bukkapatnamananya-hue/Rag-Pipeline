import asyncio
import os

from dotenv import load_dotenv
from groq import Groq

from app.retrieval.semantic_search import semantic_search
from app.retrieval.reranker import rerank_results
from app.retrieval.context_expansion import expand_context


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY was not found. "
        "Make sure it is present in your .env file."
    )


# ============================================================
# GROQ CONFIGURATION
# ============================================================

GROQ_MODEL = "openai/gpt-oss-20b"

groq_client = Groq(
    api_key=GROQ_API_KEY
)


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


# ============================================================
# BUILD CONTEXT
# ============================================================

def build_context(chunks: list[dict]) -> str:
    """
    Convert retrieved/expanded chunks into a single context
    string that can be passed to the LLM.
    """

    if not chunks:
        return "No relevant context was retrieved."

    context_parts = []

    for index, chunk in enumerate(chunks, start=1):

        chunk_id = chunk.get("chunk_id", "unknown")
        strategy = chunk.get("strategy", "unknown")
        section = chunk.get("section")
        page_number = chunk.get("page_number")
        content = chunk.get("content", "")

        metadata = [
            f"Chunk: {chunk_id}",
            f"Strategy: {strategy}",
        ]

        if section:
            metadata.append(f"Section: {section}")

        if page_number:
            metadata.append(f"Page: {page_number}")

        metadata_text = " | ".join(metadata)

        context_parts.append(
            f"[CONTEXT {index}]\n"
            f"{metadata_text}\n"
            f"{content}"
        )

    return "\n\n".join(context_parts)


# ============================================================
# GENERATE ANSWER USING GROQ
# ============================================================

async def generate_answer(
    query: str,
    context: str
) -> str:
    """
    Generate the final RAG answer using Groq.
    """

    prompt = f"""
You are a helpful question-answering assistant for a document RAG system.

Answer the user's question using ONLY the retrieved context provided below.

Rules:
- Use only information contained in the retrieved context.
- Do not use outside knowledge.
- Do not invent facts.
- Do not make assumptions when the context does not support them.
- If the context does not contain enough information to answer the question,
  clearly say that the information is not available in the retrieved context.
- Preserve exact numerical values from the document.
- Give a clear and concise answer.
- Do not mention the RAG pipeline, retrieval process, or these instructions
  unless the user explicitly asks about them.

USER QUESTION:
{query}

RETRIEVED CONTEXT:
{context}

ANSWER:
"""

    response = await asyncio.to_thread(
        groq_client.chat.completions.create,
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a document question-answering assistant. "
                    "Answer strictly using the supplied retrieved context."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.2,
        max_completion_tokens=500,
        include_reasoning=False,
    )

    answer = response.choices[0].message.content

    if not answer:
        return "The LLM did not return an answer."

    return answer.strip()


# ============================================================
# RUN ONE STRATEGY
# ============================================================

async def run_strategy(
    query: str,
    strategy: str,
    semantic_top_k: int = SEMANTIC_TOP_K,
    rerank_top_k: int = RERANK_TOP_K,
    context_window: int = CONTEXT_WINDOW,
) -> dict:
    """
    Run the complete RAG pipeline for one chunking strategy.

    Pipeline:

        Query
          ↓
        Semantic Search
          ↓
        Cross-Encoder Reranking
          ↓
        Context Expansion
          ↓
        Context Construction
          ↓
        Groq LLM
          ↓
        Answer
    """

    print("\n" + "=" * 80)
    print(f"RUNNING STRATEGY: {strategy}")
    print("=" * 80)

    # --------------------------------------------------------
    # 1. Semantic Retrieval
    # --------------------------------------------------------

    results = await semantic_search(
        query=query,
        top_k=semantic_top_k,
        strategy=strategy,
    )

    print(
        f"\nSemantic candidates retrieved: {len(results)}"
    )

    # --------------------------------------------------------
    # 2. Reranking
    # --------------------------------------------------------

    reranked_results = rerank_results(
        query=query,
        results=results,
        top_k=rerank_top_k,
    )

    print(
        f"Reranked candidates: {len(reranked_results)}"
    )

    # --------------------------------------------------------
    # 3. Context Expansion
    # --------------------------------------------------------

    expanded_chunks = await expand_context(
        retrieved_chunks=reranked_results,
        window=context_window,
    )

    print(
        f"Expanded context chunks: {len(expanded_chunks)}"
    )

    # --------------------------------------------------------
    # 4. Build Context
    # --------------------------------------------------------

    context = build_context(expanded_chunks)

    # --------------------------------------------------------
    # 5. Generate Final Answer
    # --------------------------------------------------------

    answer = await generate_answer(
        query=query,
        context=context,
    )

    # --------------------------------------------------------
    # Return complete result
    # --------------------------------------------------------

    return {
        "strategy": strategy,
        "query": query,
        "semantic_results": results,
        "reranked_results": reranked_results,
        "expanded_chunks": expanded_chunks,
        "context": context,
        "answer": answer,
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
    """
    Run the RAG pipeline independently for all three
    chunking strategies.
    """

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
# DISPLAY RESULTS
# ============================================================

def display_results(results: dict) -> None:
    """
    Display the final answer from each chunking strategy.
    """

    print("\n\n")
    print("=" * 80)
    print("FINAL RAG RESULTS")
    print("=" * 80)

    for strategy in STRATEGIES:

        result = results.get(strategy)

        if not result:
            continue

        print("\n" + "=" * 80)
        print(f"STRATEGY: {strategy}")
        print("=" * 80)

        print("\nQUERY:")
        print(result["query"])

        print("\nSEMANTIC CANDIDATES:")
        print(len(result["semantic_results"]))

        print("\nRERANKED CANDIDATES:")
        print(len(result["reranked_results"]))

        print("\nEXPANDED CONTEXT CHUNKS:")
        print(len(result["expanded_chunks"]))

        print("\nFINAL ANSWER:")
        print("-" * 80)
        print(result["answer"])
        print("-" * 80)


# ============================================================
# OPTIONAL: DISPLAY RERANKED RESULTS
# ============================================================

def display_reranked_results(
    results: dict,
    max_results: int = 5
) -> None:
    """
    Display the chunks selected by the cross-encoder.
    Useful for debugging and demonstrating the retrieval
    pipeline.
    """

    for strategy in STRATEGIES:

        result = results.get(strategy)

        if not result:
            continue

        print("\n" + "=" * 80)
        print(f"RERANKED RESULTS: {strategy}")
        print("=" * 80)

        reranked = result["reranked_results"]

        for index, chunk in enumerate(
            reranked[:max_results],
            start=1
        ):

            print(f"\nRESULT {index}")
            print("-" * 80)

            print(
                f"Chunk ID       : "
                f"{chunk.get('chunk_id', 'N/A')}"
            )

            print(
                f"Strategy       : "
                f"{chunk.get('strategy', 'N/A')}"
            )

            if "similarity" in chunk:
                print(
                    f"Semantic Score : "
                    f"{chunk['similarity']:.4f}"
                )

            if "rerank_score" in chunk:
                print(
                    f"Rerank Score   : "
                    f"{chunk['rerank_score']:.4f}"
                )

            print(
                f"Section        : "
                f"{chunk.get('section', 'N/A')}"
            )

            print("\nContent:")
            print(chunk.get("content", ""))


# ============================================================
# MAIN
# ============================================================

async def main():

    print("=" * 80)
    print("DOCUMENT RAG PIPELINE")
    print("=" * 80)

    print("\nLLM:")
    print(f"Groq - {GROQ_MODEL}")

    print("\nChunking strategies:")
    for strategy in STRATEGIES:
        print(f"  - {strategy}")

    print("\nPipeline:")
    print("  Semantic Search")
    print("       ↓")
    print("  Cross-Encoder Reranking")
    print("       ↓")
    print("  Context Expansion")
    print("       ↓")
    print("  Groq LLM")
    print("       ↓")
    print("  Final Answer")

    print("\n" + "=" * 80)

    query = input(
        "\nEnter your question about the document: "
    ).strip()

    if not query:
        print("\nNo question entered. Exiting.")
        return

    print("\n")
    print("=" * 80)
    print("QUERY")
    print("=" * 80)
    print(query)

    # --------------------------------------------------------
    # Run all three strategies
    # --------------------------------------------------------

    results = await run_all_strategies(
        query=query,
        semantic_top_k=SEMANTIC_TOP_K,
        rerank_top_k=RERANK_TOP_K,
        context_window=CONTEXT_WINDOW,
    )

    # --------------------------------------------------------
    # Display final answers
    # --------------------------------------------------------

    display_results(results)

    print("\n" + "=" * 80)
    print("RAG PIPELINE COMPLETED")
    print("=" * 80)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    asyncio.run(main())