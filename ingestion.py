from unstructured.partition.pdf import partition_pdf

# 1. Partition the PDF globally so it can be imported by other files like demo.py
elements = partition_pdf(
    filename="annual_report_ideaforge_technology_limited_fy2026.pdf",
    strategy="hi_res",
    infer_table_structure=True,
)

if __name__ == "__main__":
    # Inspect raw elements locally when running ingestion.py directly
    for i, chunk in enumerate(elements[:5]):
        print(f"--- Element {i+1} ---")
        print(f"Category: {getattr(chunk, 'category', 'Uncategorized')}")
        print(f"Page Number: {getattr(getattr(chunk, 'metadata', None), 'page_number', 'N/A')}")
        print(f"Text Preview: {getattr(chunk, 'text', '')[:200]}\n")