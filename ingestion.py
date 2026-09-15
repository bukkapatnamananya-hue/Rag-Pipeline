from unstructured.partition.pdf import partition_pdf


PDF_FILE = "annual_report_ideaforge_technology_limited_fy2026.pdf"


elements = partition_pdf(
    filename=PDF_FILE,
    strategy="fast",
)


if __name__ == "__main__":
    print(f"Loaded {len(elements)} elements from PDF.")

    for i, element in enumerate(elements[:5]):
        print(f"\n--- Element {i + 1} ---")
        print(f"Category: {getattr(element, 'category', 'Uncategorized')}")
        print(f"Page: {getattr(getattr(element, 'metadata', None), 'page_number', 'N/A')}")
        print(f"Text: {getattr(element, 'text', '')[:300]}")