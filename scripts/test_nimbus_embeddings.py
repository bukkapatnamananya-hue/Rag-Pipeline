from ingestion import elements
from chunking import custom_rule_chunking
from app.embeddings.model import create_embedding


print("Running chunking...")

chunks = custom_rule_chunking(elements)

print(f"Total chunks: {len(chunks)}")

print("\nCreating embeddings...")

for i, chunk in enumerate(chunks, start=1):
    embedding = create_embedding(chunk.page_content)

    print(f"Chunk {i}:")
    print(f"  Text length: {len(chunk.page_content)}")
    print(f"  Embedding dimension: {len(embedding)}")