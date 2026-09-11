from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def custom_rule_chunking(elements):
    """Executes chunking separately using LangChain and custom rules
    (e.g., preserving tables, custom section boundaries).
    """
    custom_docs = []
    current_chunk = ""
    current_metadata = {}

    # Initialize standard LangChain splitter for body text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    for element in elements:
        text = getattr(element, "text", "")
        category = getattr(element, "category", "Uncategorized")

        # Custom Rule 1: Keep tables strictly intact as standalone documents
        if category == "Table":
            if current_chunk:
                sub_docs = text_splitter.create_documents(
                    [current_chunk], metadatas=[current_metadata]
                )
                custom_docs.extend(sub_docs)
                current_chunk = ""
                current_metadata = {}
            
            table_metadata = {
                "category": "Table", 
                "page_number": getattr(getattr(element, "metadata", None), "page_number", None)
            }
            custom_docs.append(Document(page_content=text, metadata=table_metadata))
            continue

        # Custom Rule 2: Handle Titles / Section Headings
        if category == "Title":
            if current_chunk:
                sub_docs = text_splitter.create_documents(
                    [current_chunk], metadatas=[current_metadata]
                )
                custom_docs.extend(sub_docs)
                current_chunk = ""
            current_metadata = {"category": "Title", "section_heading": text}

        current_chunk += "\n" + text

    # Flush the remaining buffer
    if current_chunk:
        sub_docs = text_splitter.create_documents(
            [current_chunk], metadatas=[current_metadata]
        )
        custom_docs.extend(sub_docs)

    return custom_docs

if __name__ == "__main__":
    from ingestion import elements
    
    print(f"Loaded {len(elements)} raw elements from PDF.")
    
    # Run your custom chunking logic
    chunked_docs = custom_rule_chunking(elements)
    
    print(f"Successfully generated {len(chunked_docs)} chunks!")
    
    # Print first 2 chunks to verify output
    for i, doc in enumerate(chunked_docs[:2]):
        print(f"\n--- Chunk {i+1} ---")
        print(f"Metadata: {doc.metadata}")
        print(f"Content Preview: {doc.page_content[:300]}...")