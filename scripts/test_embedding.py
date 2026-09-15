from app.embeddings.model import create_embedding


text = "The standard warranty covers manufacturing defects."

embedding = create_embedding(text)

print("Embedding created successfully!")
print("Embedding dimension:", len(embedding))
print("First 10 values:", embedding[:10])