from chunking import custom_rule_chunking
from ingestion import elements
from app.embeddings.model import create_embedding


print("Running chunking...")

chunks = custom_rule_chunking(elements)

print(f"Total chunks: {len(chunks)}")


first_chunk = chunks[0]

print("\nFirst chunk:")
print(first_chunk.page_content[:500])

print("\nCreating embedding...")

embedding = create_embedding(first_chunk.page_content)

print("Embedding created successfully!")
print("Embedding dimension:", len(embedding))
print("First 10 values:", embedding[:10])