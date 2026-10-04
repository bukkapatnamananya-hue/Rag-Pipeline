import asyncio

from sqlalchemy import text

from app.database.connection import engine


async def main():
    async with engine.connect() as conn:
        result = await conn.execute(
            text("""
                SELECT
                    id,
                    document_id,
                    document_name,
                    chunk_id,
                    section,
                    LENGTH(content) AS content_length,
                    vector_dims(embedding) AS embedding_dimension
                FROM knowledge_chunks
                ORDER BY id;
            """)
        )

        print("\nEXISTING KNOWLEDGE CHUNKS")
        print("=" * 80)

        for row in result:
            print(f"ID:                  {row.id}")
            print(f"Document ID:         {row.document_id}")
            print(f"Document name:       {row.document_name}")
            print(f"Chunk ID:            {row.chunk_id}")
            print(f"Section:             {row.section}")
            print(f"Content length:      {row.content_length}")
            print(f"Embedding dimension: {row.embedding_dimension}")
            print("-" * 80)


if __name__ == "__main__":
    asyncio.run(main())