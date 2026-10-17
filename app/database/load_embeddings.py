import argparse
import asyncio
import json
from pathlib import Path

from app.database.session import SessionLocal
from app.database.models import KnowledgeChunk


BASE_DIR = Path(__file__).resolve().parents[2]

EMBEDDING_FILES = [
    BASE_DIR / "data" / "embeddings" / "recursive_embeddings.json",
    BASE_DIR / "data" / "embeddings" / "structure_aware_embeddings.json",
    BASE_DIR / "data" / "embeddings" / "contextual_embeddings.json",
]


def normalize_document_name(value):
    text = str(value or "document").strip()
    text = text.replace("\\", "/").split("/")[-1]
    text = text.rsplit(".", 1)[0]
    text = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text)
    return text or "document"


def resolve_document_name(file_path: Path, override: str | None = None) -> str:
    if override:
        return override

    stem = file_path.stem.replace("_embeddings", "")
    return normalize_document_name(stem)


async def load_embeddings(document_name: str | None = None):
    total_loaded = 0

    async with SessionLocal() as session:
        for file_path in EMBEDDING_FILES:
            actual_document_name = resolve_document_name(file_path, document_name)

            print(f"\nLoading: {file_path.name}")
            with open(file_path, "r", encoding="utf-8") as f:
                chunks = json.load(f)

            print(f"Found {len(chunks)} chunks")

            for chunk in chunks:
                knowledge_chunk = KnowledgeChunk(
                    document_id=chunk.get("document_id", actual_document_name),
                    document_name=actual_document_name,
                    chunk_id=chunk["chunk_id"],
                    parent_chunk_id=None,
                    content=chunk["text"],
                    summary=None,
                    page_number=None,
                    section=chunk.get("section"),
                    strategy=chunk["strategy"],
                    embedding=chunk["embedding"],
                )

                session.add(knowledge_chunk)
                total_loaded += 1

        await session.commit()

    print("\n" + "=" * 60)
    print(f"TOTAL CHUNKS LOADED: {total_loaded}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load chunk embeddings into PostgreSQL.")
    parser.add_argument(
        "--document-name",
        type=str,
        default=None,
        help="Optional document name to label chunks with. Defaults to the embedding file name.",
    )
    args = parser.parse_args()

    asyncio.run(load_embeddings(args.document_name))