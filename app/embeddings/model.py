from sentence_transformers import SentenceTransformer


MODEL_NAME = "BAAI/bge-small-en-v1.5"

model = SentenceTransformer(MODEL_NAME)


def create_embedding(text: str) -> list[float]:
    """
    Convert text into a numerical embedding vector.
    """
    embedding = model.encode(text)

    return embedding.tolist()