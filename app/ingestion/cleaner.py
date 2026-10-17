import json
from pathlib import Path


INPUT_PATH = Path("data/processed/unstructured_elements.json")
OUTPUT_PATH = Path("data/processed/cleaned_elements.json")


def load_elements():
    """Load elements extracted by Unstructured."""
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH.resolve()}")

    with open(INPUT_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def clean_text(text):
    """Normalize text from any document type."""
    if not isinstance(text, str):
        return ""

    cleaned = text.replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ")
    cleaned = " ".join(cleaned.split())
    return cleaned.strip()


def build_cleaned_elements(elements):
    """
    Keep the extracted structure generic so it works for PDFs of any kind.
    """
    cleaned = []

    for index, element in enumerate(elements):
        if not isinstance(element, dict):
            continue

        element_type = element.get("element_type") or "NarrativeText"
        text = clean_text(element.get("text", ""))

        if not text:
            continue

        cleaned.append(
            {
                "element_id": element.get("element_id", index),
                "element_type": element_type,
                "section": element.get("section"),
                "text": text,
                "metadata": element.get("metadata", {}),
            }
        )

    return cleaned


def main():
    print(f"Reading: {INPUT_PATH}")

    elements = load_elements()
    print(f"Loaded {len(elements)} extracted elements")

    cleaned = build_cleaned_elements(elements)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(cleaned, file, indent=2, ensure_ascii=False)

    print()
    print("Cleaning complete.")
    print(f"Created {len(cleaned)} cleaned elements")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()