import asyncio

from app.database.session import SessionLocal
from app.database.models import KnowledgeChunk
from app.embeddings.model import create_embedding


async def store_test_chunk():
    content = "The standard warranty covers manufacturing defects."

    embedding = create_embedding(content)

    async with SessionLocal() as session:
        chunk = KnowledgeChunk(
            document_id="test_document",
            document_name="test_document.txt",
            chunk_id="chunk_0001",
            parent_chunk_id=None,
            content=content,
            summary=None,
            page_number=1,
            section="Test Section",
            embedding=embedding,
        )

        session.add(chunk)
        await session.commit()

        print("Test chunk stored successfully!")
        print("Embedding dimension:", len(embedding))


if __name__ == "__main__":
    asyncio.run(store_test_chunk())
    