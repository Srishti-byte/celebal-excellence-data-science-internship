from src.pdf_loader import PDFLoader
from src.text_cleaner import TextCleaner
from src.chunker import StructuredChunker
from src.embeddings import EmbeddingManager
from src.config import EMBEDDING_DIMENSION


def main():
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

    print("\nInitializing embedding model...")

    embedding_manager = EmbeddingManager()

    print(
        f"Embedding Model: "
        f"{embedding_manager.get_model_name()}"
    )

    print(
        f"Expected Embedding Dimension: "
        f"{embedding_manager.get_embedding_dimension()}"
    )

    sample_size = min(5, len(chunks))
    sample_chunks = chunks[:sample_size]

    print(
        f"\nGenerating embeddings for "
        f"{sample_size} sample chunks..."
    )

    vectors = embedding_manager.embed_documents(
        sample_chunks
    )

    if not vectors:
        raise ValueError(
            "No embeddings were generated."
        )

    print("\nEmbedding Results")
    print("=" * 80)

    print(f"Number of vectors: {len(vectors)}")
    print(f"Vector dimension: {len(vectors[0])}")

    for index, vector in enumerate(vectors):
        if len(vector) != EMBEDDING_DIMENSION:
            raise ValueError(
                f"Incorrect embedding dimension for "
                f"vector {index}: {len(vector)}"
            )

        chunk = sample_chunks[index]

        print(
            f"\nChunk {index}"
            f"\nBrand: {chunk.metadata.get('brand')}"
            f"\nModel: {chunk.metadata.get('model')}"
            f"\nSection: "
            f"{chunk.metadata.get('brochure_section')}"
            f"\nChunk ID: "
            f"{chunk.metadata.get('chunk_id')}"
        )

        print(f"Vector dimension: {len(vector)}")
        print(f"First 5 values: {vector[:5]}")

    print("\nTesting query embedding...")

    query = "What safety features does this car have?"

    query_vector = embedding_manager.embed_query(query)

    if len(query_vector) != EMBEDDING_DIMENSION:
        raise ValueError(
            "Incorrect query embedding dimension."
        )

    print(f"Query: {query}")
    print(
        f"Query vector dimension: "
        f"{len(query_vector)}"
    )
    print(
        f"First 5 query values: "
        f"{query_vector[:5]}"
    )

    print("\nValidation")
    print("=" * 80)

    assert len(vectors) == sample_size

    assert all(
        len(vector) == EMBEDDING_DIMENSION
        for vector in vectors
    )

    assert len(query_vector) == EMBEDDING_DIMENSION

    assert all(
        isinstance(value, float)
        for vector in vectors
        for value in vector
    )

    assert all(
        isinstance(value, float)
        for value in query_vector
    )

    print("All embedding validation checks passed.")


if __name__ == "__main__":
    main()