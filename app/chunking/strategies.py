import json
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter


INPUT_PATH = Path(
    "data/processed/cleaned_elements.json"
)

OUTPUT_DIR = Path(
    "data/chunks"
)


def load_elements():
    """Load cleaned elements from the JSON file."""

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_PATH.resolve()}"
        )

    with open(INPUT_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


# ============================================================
# Strategy 1: Recursive Chunking
# ============================================================

def recursive_chunking(elements):
    """
    Recursive chunking using LangChain's
    RecursiveCharacterTextSplitter.

    Related cleaned elements are first combined into larger
    text blocks, then recursively split into manageable chunks.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    chunks = []
    chunk_index = 0

    # Combine related elements into larger blocks
    current_block = []
    current_section = "General"

    for element in elements:

        text = element["text"].strip()

        if not text:
            continue

        element_type = element["element_type"]

        # Section headers start a new logical block
        if element_type == "SectionHeader":

            if current_block:
                block_text = "\n".join(current_block)

                split_texts = splitter.split_text(block_text)

                for text_chunk in split_texts:

                    chunks.append({
                        "chunk_id": f"recursive_{chunk_index}",
                        "document_id": "zen_annual_report_fy24",
                        "strategy": "recursive",
                        "section": current_section,
                        "source_element_type": "CombinedText",
                        "chunk_index": chunk_index,
                        "text": text_chunk,
                        "character_count": len(text_chunk),
                        "word_count": len(text_chunk.split())
                    })

                    chunk_index += 1

            current_block = []
            current_section = element.get(
                "section",
                text
            )

            continue

        current_block.append(text)

    # Process final block
    if current_block:

        block_text = "\n".join(current_block)

        split_texts = splitter.split_text(block_text)

        for text_chunk in split_texts:

            chunks.append({
                "chunk_id": f"recursive_{chunk_index}",
                "document_id": "zen_annual_report_fy24",
                "strategy": "recursive",
                "section": current_section,
                "source_element_type": "CombinedText",
                "chunk_index": chunk_index,
                "text": text_chunk,
                "character_count": len(text_chunk),
                "word_count": len(text_chunk.split())
            })

            chunk_index += 1

    return chunks
# ============================================================
# Strategy 2: Structure-Aware Chunking
# ============================================================

def structure_aware_chunking(elements):
    """
    Structure-aware chunking.

    Uses the structure created during the cleaning stage.
    Section headers establish context, while TableRows are
    preserved as complete semantic units.
    """

    chunks = []
    chunk_index = 0

    current_section = "General"

    for element in elements:

        text = element["text"].strip()

        if not text:
            continue

        element_type = element["element_type"]

        # Section headers establish the current section
        if element_type == "SectionHeader":

            current_section = element.get(
                "section",
                text
            )

            continue

        chunks.append({
            "chunk_id": f"structure_{chunk_index}",
            "document_id": "zen_annual_report_fy24",
            "strategy": "structure_aware",
            "section": current_section,
            "source_element_type": element_type,
            "chunk_index": chunk_index,
            "text": text,
            "character_count": len(text),
            "word_count": len(text.split())
        })

        chunk_index += 1

    return chunks


# ============================================================
# Strategy 3: Contextual Chunking
# ============================================================

def contextual_chunking(elements):
    """
    Contextual chunking.

    Adds document-level and section-level context to every
    semantic chunk before embedding.

    This version does not require an external LLM/API.
    """

    chunks = []
    chunk_index = 0

    document_context = (
        "Document: Zen Technologies Limited Annual Report 2023-24. "
        "The document contains corporate overview, financial performance, "
        "business segment revenue, and research and development information."
    )

    current_section = "General"

    for element in elements:

        text = element["text"].strip()

        if not text:
            continue

        element_type = element["element_type"]

        # Update section context
        if element_type == "SectionHeader":

            current_section = element.get(
                "section",
                text
            )

            continue

        contextual_text = (
            f"{document_context}\n"
            f"Section: {current_section}\n"
            f"Content:\n{text}"
        )

        chunks.append({
            "chunk_id": f"contextual_{chunk_index}",
            "document_id": "zen_annual_report_fy24",
            "strategy": "contextual",
            "section": current_section,
            "source_element_type": element_type,
            "chunk_index": chunk_index,
            "text": contextual_text,
            "original_text": text,
            "character_count": len(contextual_text),
            "word_count": len(contextual_text.split())
        })

        chunk_index += 1

    return chunks


# ============================================================
# Save Chunks
# ============================================================

def save_chunks(chunks, filename):

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
        f"Saved {len(chunks)} chunks → {output_path}"
    )


# ============================================================
# Main
# ============================================================

def main():

    print(
        f"Reading cleaned elements from: {INPUT_PATH}"
    )

    elements = load_elements()

    print(
        f"Loaded {len(elements)} cleaned elements"
    )

    # --------------------------------------------------------
    # Strategy 1
    # --------------------------------------------------------

    print("\nCreating recursive chunks...")

    recursive_chunks = recursive_chunking(elements)

    save_chunks(
        recursive_chunks,
        "recursive_chunks.json"
    )

    # --------------------------------------------------------
    # Strategy 2
    # --------------------------------------------------------

    print("\nCreating structure-aware chunks...")

    structure_chunks = structure_aware_chunking(elements)

    save_chunks(
        structure_chunks,
        "structure_aware_chunks.json"
    )

    # --------------------------------------------------------
    # Strategy 3
    # --------------------------------------------------------

    print("\nCreating contextual chunks...")

    contextual_chunks = contextual_chunking(elements)

    save_chunks(
        contextual_chunks,
        "contextual_chunks.json"
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("CHUNKING COMPLETE")
    print("=" * 60)

    print(
        f"Recursive chunks:        {len(recursive_chunks)}"
    )

    print(
        f"Structure-aware chunks:  {len(structure_chunks)}"
    )

    print(
        f"Contextual chunks:       {len(contextual_chunks)}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()