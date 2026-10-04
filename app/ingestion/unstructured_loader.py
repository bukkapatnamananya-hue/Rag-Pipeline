from pathlib import Path
import json
import argparse

from unstructured.partition.pdf import partition_pdf


DEFAULT_PDF_PATH = Path(
    "data/raw/annual_report_zen_technologies_fy24.pdf"
)

OUTPUT_PATH = Path(
    "data/processed/unstructured_elements.json"
)


def extract_pdf(pdf_path: Path):
    """
    Extract a PDF using Unstructured and return
    structured document elements.
    """

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path.resolve()}"
        )

    elements = partition_pdf(
        filename=str(pdf_path),
        strategy="auto",
    )

    return elements


def element_to_dict(element, index: int):
    """
    Convert an Unstructured element into a JSON-safe dictionary.
    """

    metadata = element.metadata.to_dict()

    return {
        "element_id": index,
        "element_type": element.category,
        "text": str(element),
        "metadata": metadata,
    }


def main():

    parser = argparse.ArgumentParser(
        description="Extract elements from a PDF using Unstructured."
    )

    parser.add_argument(
        "--file",
        type=str,
        default=str(DEFAULT_PDF_PATH),
        help="Path to the PDF file to process."
    )

    args = parser.parse_args()

    pdf_path = Path(args.file)

    print(f"Reading PDF: {pdf_path}")

    elements = extract_pdf(pdf_path)

    print(f"Extracted {len(elements)} elements")

    output = []

    for index, element in enumerate(elements):

        output.append(
            element_to_dict(element, index)
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False
        )

    print()
    print("Extraction complete.")
    print(f"Output saved to: {OUTPUT_PATH}")
    print()

    # Show a small summary
    element_types = {}

    for element in output:

        element_type = element["element_type"]

        element_types[element_type] = (
            element_types.get(element_type, 0) + 1
        )

    print("Element type summary:")

    for element_type, count in element_types.items():

        print(
            f"  {element_type}: {count}"
        )


if __name__ == "__main__":
    main()