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

DOCUMENT_NAME = "annual_report_zen_technologies_fy24.pdf"


async def load_embeddings():
    total_loaded = 0

    async with SessionLocal() as session:
        for file_path in EMBEDDING_FILES:
            print(f"\nLoading: {file_path.name}")

            with open(file_path, "r", encoding="utf-8") as f:
                chunks = json.load(f)

            print(f"Found {len(chunks)} chunks")

            for chunk in chunks:
                knowledge_chunk = KnowledgeChunk(
                    document_id=chunk["document_id"],
                    document_name=DOCUMENT_NAME,
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
    asyncio.run(load_embeddings())