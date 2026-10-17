import hashlib
import json
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter


INPUT_PATH = Path("data/processed/cleaned_elements.json")
OUTPUT_DIR = Path("data/chunks")


def normalize_document_name(value):
    """Convert a file name or title to a safe, reusable document identifier."""
    text = str(value or "document").strip()
    text = text.replace("\\", "/").split("/")[-1]
    text = text.rsplit(".", 1)[0]
    text = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text)
    return text or "document"


def infer_document_metadata(elements):
    """
    Infer a document id and title from the actual extracted content.
    This removes all hardcoded annual-report assumptions.
    """
    source_name = "document"
    title = "Uploaded Document"

    for element in elements:
        metadata = element.get("metadata") or {}

        for key in ("filename", "file_name", "source_name", "source"):
            candidate = metadata.get(key)
            if candidate:
                source_name = candidate
                break

        element_type = element.get("element_type", "")
        text = str(element.get("text", "")).strip()

        if element_type in {"Title", "Header", "SectionHeader"} and text:
            title = text
            break

        if text and len(text) > 20 and element_type in {"NarrativeText", "ListItem", "TableRow"}:
            title = text[:80]
            break

    safe_name = normalize_document_name(source_name or title)
    doc_hash = hashlib.sha1(f"{safe_name}:{title}".encode("utf-8")).hexdigest()[:12]
    document_id = f"doc_{safe_name}_{doc_hash}"

    return document_id, title


def load_elements():
    """Load cleaned elements from the JSON file."""
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH.resolve()}")

    with open(INPUT_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


# ============================================================
# Strategy 1: Recursive Chunking
# ============================================================

def recursive_chunking(elements):
    document_id, _ = infer_document_metadata(elements)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    chunk_index = 0

    current_block = []
    current_section = "General"

    for element in elements:
        text = str(element.get("text", "")).strip()
        if not text:
            continue

        element_type = element.get("element_type", "NarrativeText")

        if element_type == "SectionHeader":
            if current_block:
                block_text = "\n".join(current_block)
                split_texts = splitter.split_text(block_text)

                for text_chunk in split_texts:
                    chunks.append(
                        {
                            "chunk_id": f"recursive_{chunk_index}",
                            "document_id": document_id,
                            "strategy": "recursive",
                            "section": current_section,
                            "source_element_type": "CombinedText",
                            "chunk_index": chunk_index,
                            "text": text_chunk,
                            "character_count": len(text_chunk),
                            "word_count": len(text_chunk.split()),
                        }
                    )
                    chunk_index += 1

            current_block = []
            current_section = element.get("section") or text
            continue

        current_block.append(text)

    if current_block:
        block_text = "\n".join(current_block)
        split_texts = splitter.split_text(block_text)

        for text_chunk in split_texts:
            chunks.append(
                {
                    "chunk_id": f"recursive_{chunk_index}",
                    "document_id": document_id,
                    "strategy": "recursive",
                    "section": current_section,
                    "source_element_type": "CombinedText",
                    "chunk_index": chunk_index,
                    "text": text_chunk,
                    "character_count": len(text_chunk),
                    "word_count": len(text_chunk.split()),
                }
            )
            chunk_index += 1

    return chunks


# ============================================================
# Strategy 2: Structure-Aware Chunking
# ============================================================

def structure_aware_chunking(elements):
    document_id, _ = infer_document_metadata(elements)

    chunks = []
    chunk_index = 0
    current_section = "General"

    for element in elements:
        text = str(element.get("text", "")).strip()
        if not text:
            continue

        element_type = element.get("element_type", "NarrativeText")

        if element_type == "SectionHeader":
            current_section = element.get("section") or text
            continue

        chunks.append(
            {
                "chunk_id": f"structure_{chunk_index}",
                "document_id": document_id,
                "strategy": "structure_aware",
                "section": current_section,
                "source_element_type": element_type,
                "chunk_index": chunk_index,
                "text": text,
                "character_count": len(text),
                "word_count": len(text.split()),
            }
        )
        chunk_index += 1

    return chunks


# ============================================================
# Strategy 3: Contextual Chunking
# ============================================================

def contextual_chunking(elements):
    document_id, title = infer_document_metadata(elements)

    chunks = []
    chunk_index = 0

    document_context = f"Document: {title}. This is a generic document chunk for retrieval and indexing."

    current_section = "General"

    for element in elements:
        text = str(element.get("text", "")).strip()
        if not text:
            continue

        element_type = element.get("element_type", "NarrativeText")

        if element_type == "SectionHeader":
            current_section = element.get("section") or text
            continue

        contextual_text = (
            f"{document_context}\n"
            f"Section: {current_section}\n"
            f"Content:\n{text}"
        )

        chunks.append(
            {
                "chunk_id": f"contextual_{chunk_index}",
                "document_id": document_id,
                "strategy": "contextual",
                "section": current_section,
                "source_element_type": element_type,
                "chunk_index": chunk_index,
                "text": contextual_text,
                "original_text": text,
                "character_count": len(contextual_text),
                "word_count": len(contextual_text.split()),
            }
        )

        chunk_index += 1

    return chunks


# ============================================================
# Save Chunks
# ============================================================

def save_chunks(chunks, filename):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / filename

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(chunks, file, indent=2, ensure_ascii=False)

    print(f"Saved {len(chunks)} chunks → {output_path}")


# ============================================================
# Main
# ============================================================

def main():
    print(f"Reading cleaned elements from: {INPUT_PATH}")
    elements = load_elements()
    print(f"Loaded {len(elements)} cleaned elements")

    print("\nCreating recursive chunks...")
    recursive_chunks = recursive_chunking(elements)
    save_chunks(recursive_chunks, "recursive_chunks.json")

    print("\nCreating structure-aware chunks...")
    structure_chunks = structure_aware_chunking(elements)
    save_chunks(structure_chunks, "structure_aware_chunks.json")

    print("\nCreating contextual chunks...")
    contextual_chunks = contextual_chunking(elements)
    save_chunks(contextual_chunks, "contextual_chunks.json")

    print("\n" + "=" * 60)
    print("CHUNKING COMPLETE")
    print("=" * 60)

    print(f"Recursive chunks:        {len(recursive_chunks)}")
    print(f"Structure-aware chunks:  {len(structure_chunks)}")
    print(f"Contextual chunks:       {len(contextual_chunks)}")
    print("=" * 60)


if __name__ == "__main__":
    main()