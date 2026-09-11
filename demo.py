from chunking import custom_rule_chunking
from ingestion import elements

# Step 2: Run your custom chunking logic on unstructured elements
chunked_docs = custom_rule_chunking(elements)

# Step 3: Show a few chunks for demo
print(f"Total chunks created: {len(chunked_docs)}\n")
for i, doc in enumerate(chunked_docs[:3]):  # show first 3 chunks
    print(f"--- Chunk {i+1} ---")
    print(doc.page_content[:500])  # preview first 500 chars
    print("Metadata:", doc.metadata)
    print()