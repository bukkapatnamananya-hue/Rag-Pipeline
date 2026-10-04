from datetime import datetime
from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class KnowledgeChunk(Base):
    embedding: Mapped[list[float] | None] = mapped_column(
    Vector(384),
    nullable=True
)

    __tablename__ = "knowledge_chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    document_id: Mapped[str] = mapped_column(String(255), nullable=False)
    document_name: Mapped[str] = mapped_column(String(255), nullable=False)

    chunk_id: Mapped[str] = mapped_column(String(255), nullable=False)
    parent_chunk_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True
    )
    strategy: Mapped[str] = mapped_column(
    String(50),
    nullable=False
)

    content: Mapped[str] = mapped_column(Text, nullable=False)

    summary: Mapped[str | None] = mapped_column(
        Text,
        nullable=True
    )

    page_number: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True
    )

    section: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True
    )
    strategy: Mapped[str] = mapped_column(
    String(50),
    nullable=False
)

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False
    )