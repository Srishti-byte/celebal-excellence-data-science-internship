from src.pdf_loader import PDFLoader
from src.text_cleaner import TextCleaner
from src.chunker import StructuredChunker
from src.embeddings import EmbeddingManager
from src.vector_store import VectorStore


def main():
    print("\nBuilding production vector store...")
    print("=" * 60)

    print("\nLoading documents...")

    loader = PDFLoader()
    documents = loader.load_documents()

    print(f"Documents Loaded: {len(documents)}")

    print("\nCleaning documents...")

    cleaner = TextCleaner()
    documents = cleaner.clean_documents(documents)

    print(f"Documents After Cleaning: {len(documents)}")

    print("\nCreating structured chunks...")

    chunker = StructuredChunker()
    chunks = chunker.chunk_documents(documents)

    print(f"Total Chunks: {len(chunks)}")

    if not chunks:
        raise ValueError("No chunks were generated.")

    print("\nGenerating embeddings...")

    embedding_manager = EmbeddingManager()
    embeddings = embedding_manager.embed_documents(chunks)

    print(f"Embeddings Generated: {len(embeddings)}")

    if len(embeddings) != len(chunks):
        raise ValueError(
            "Number of embeddings does not match "
            "number of chunks."
        )

    print("\nInitializing production vector store...")

    vector_store = VectorStore()

    print(
        f"Collection: "
        f"{vector_store.get_collection_name()}"
    )

    print(
        f"Existing Documents: "
        f"{vector_store.count()}"
    )

    print("\nStoring documents and embeddings...")

    vector_store.add_documents(
        documents=chunks,
        embeddings=embeddings,
    )

    print(
        f"Documents Stored: "
        f"{vector_store.count()}"
    )

    print("\nProduction vector store ready.")
    print("=" * 60)


if __name__ == "__main__":
    main()