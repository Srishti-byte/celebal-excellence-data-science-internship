"""Test file for the MetadataManager module."""
from modules.loader import DocumentLoader
from modules.chunker import TextChunker
from modules.metadata_manager import MetadataManager

def main():
    loader = DocumentLoader("data/rag_paper.pdf")
    documents = loader.load_document()

    chunker = TextChunker()
    chunks = chunker.split_documents(documents)

    metadata_manager = MetadataManager()
    processed_chunks = metadata_manager.process_metadata(chunks)

    print("\nFirst Chunk Metadata:\n")
    print(processed_chunks[0].metadata)
    print("\nSecond Chunk Metadata:\n")
    print(processed_chunks[1].metadata)

if __name__ == "__main__":
    main()