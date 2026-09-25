import json
from pathlib import Path

from app.embeddings.model import create_embedding


CHUNK_DIR = Path("data/chunks")
OUTPUT_DIR = Path("data/embeddings")


FILES = {
    "recursive": CHUNK_DIR / "recursive_chunks.json",
    "structure_aware": CHUNK_DIR / "structure_aware_chunks.json",
    "contextual": CHUNK_DIR / "contextual_chunks.json",
}


def load_chunks(path):
    """Load chunks from a JSON file."""

    if not path.exists():
        raise FileNotFoundError(
            f"Chunk file not found: {path.resolve()}"
        )

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def embed_chunks(chunks):
    """Create an embedding for every chunk."""

    embedded_chunks = []

    for index, chunk in enumerate(chunks):

        print(
            f"Embedding chunk {index + 1}/{len(chunks)}..."
        )

        embedding = create_embedding(
            chunk["text"]
        )

        embedded_chunk = {
            **chunk,
            "embedding": embedding,
            "embedding_dimension": len(embedding)
        }

        embedded_chunks.append(
            embedded_chunk
        )

    return embedded_chunks


def save_embeddings(chunks, filename):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = OUTPUT_DIR / filename

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            chunks,
            file,
            indent=2,
            ensure_ascii=False
        )

    print(
        f"Saved embeddings → {output_path}"
    )


def main():

    print("=" * 60)
    print("EMBEDDING GENERATION")
    print("=" * 60)

    for strategy, path in FILES.items():

        print(
            f"\nProcessing {strategy} chunks..."
        )

        chunks = load_chunks(path)

        print(
            f"Loaded {len(chunks)} chunks"
        )

        embedded_chunks = embed_chunks(
            chunks
        )

        output_filename = (
            f"{strategy}_embeddings.json"
        )

        save_embeddings(
            embedded_chunks,
            output_filename
        )

    print("\n" + "=" * 60)
    print("EMBEDDING GENERATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()