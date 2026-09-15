from sqlalchemy.ext.asyncio import async_sessionmaker

from app.database.connection import engine


SessionLocal = async_sessionmaker(
    bind=engine,
    expire_on_commit=False
)