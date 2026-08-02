"""This module standardizes metadata for document chunks."""

import os
from langchain_core.documents import Document

class MetadataManager:
    """Cleans and standardizes metadata for document chunks."""
    def process_metadata(self, documents: list[Document]) -> list[Document]:
        """
        Standardize metadata for all chunks.
        Args:
            documents: List of chunked LangChain Document objects.
        Returns:
            List of Document objects with standardized metadata.
        """
        for chunk_id, document in enumerate(documents, start=1):
            source = document.metadata.get("source", "")
            source = os.path.basename(source)

            document.metadata = {
                "chunk_id": chunk_id,
                "page": document.metadata.get("page"),
                "source": source
            }

        print("=" * 50)
        print("Metadata Processing Completed")
        print(f"Processed Chunks: {len(documents)}")
        print("=" * 50)

        return documents