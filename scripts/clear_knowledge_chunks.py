import asyncio

from sqlalchemy import text

from app.database.connection import engine


async def main():
    async with engine.begin() as conn:
        result = await conn.execute(
            text("DELETE FROM knowledge_chunks")
        )
        print(f"Deleted {result.rowcount} existing rows.")

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())