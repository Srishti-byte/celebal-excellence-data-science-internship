"""
Test file for the TextChunker module.
"""
from modules.loader import DocumentLoader
from modules.chunker import TextChunker

def main():
    loader = DocumentLoader("data/rag_paper.pdf")
    documents = loader.load_document()

    # Split the document into chunks
    chunker = TextChunker()
    chunks = chunker.split_documents(documents)

    print(f"\nTotal Chunks: {len(chunks)}")

    print("\nFirst Chunk Metadata:\n")
    print(chunks[0].metadata)

    print("\nFirst Chunk:\n")
    print(chunks[0].page_content)
    print("\nSecond Chunk:\n")
    print(chunks[1].page_content)

if __name__ == "__main__":
    main()