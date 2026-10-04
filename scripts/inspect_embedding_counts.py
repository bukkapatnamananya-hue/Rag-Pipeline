import asyncio
from sqlalchemy import select, func

from app.database.connection import engine
from app.database.session import SessionLocal
from app.database.models import KnowledgeChunk


async def main():
    async with SessionLocal() as session:
        result = await session.execute(
            select(
                KnowledgeChunk.strategy,
                func.count(KnowledgeChunk.id)
            ).group_by(KnowledgeChunk.strategy)
        )

        rows = result.all()

        print("\nEmbedding counts by strategy:")
        print("-" * 40)

        total = 0

        for strategy, count in rows:
            print(f"{strategy}: {count}")
            total += count

        print("-" * 40)
        print(f"TOTAL: {total}")

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())