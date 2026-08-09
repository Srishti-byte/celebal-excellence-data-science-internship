from src.pdf_loader import PDFLoader
from src.text_cleaner import TextCleaner
from src.chunker import StructuredChunker
from src.embeddings import EmbeddingManager
from src.vector_store import VectorStore

from src.config import (
    CHROMA_PERSIST_DIRECTORY,
    CHROMA_TEST_COLLECTION_NAME,
    EMBEDDING_DIMENSION,
    RETRIEVAL_K,
)


def prepare_vector_store(embedding_manager):
    print("\nLoading documents...")

    loader = PDFLoader()
    documents = loader.load_documents()

    print(
        f"Documents Loaded: {len(documents)}"
    )

    print("\nCleaning documents...")

    cleaner = TextCleaner()
    documents = cleaner.clean_documents(documents)

    print(
        f"Documents After Cleaning: "
        f"{len(documents)}"
    )

    print("\nCreating structured chunks...")

    chunker = StructuredChunker()
    chunks = chunker.chunk_documents(documents)

    print(
        f"Total Chunks: {len(chunks)}"
    )

    if not chunks:
        raise ValueError(
            "No chunks were generated."
        )

    print("\nGenerating embeddings...")

    embeddings = embedding_manager.embed_documents(
        chunks
    )

    if len(embeddings) != len(chunks):
        raise ValueError(
            "Number of embeddings does not match "
            "number of chunks."
        )

    print(
        f"Embeddings Generated: "
        f"{len(embeddings)}"
    )

    print("\nInitializing ChromaDB...")

    vector_store = VectorStore(
        persist_directory=CHROMA_PERSIST_DIRECTORY,
        collection_name=CHROMA_TEST_COLLECTION_NAME,
    )

    vector_store.clear()

    vector_store.add_documents(
        documents=chunks,
        embeddings=embeddings,
    )

    print(
        f"Documents Stored: "
        f"{vector_store.count()}"
    )

    return vector_store


def display_results(results):
    print("\nRetrieved Results")
    print("=" * 80)

    for index, (document, score) in enumerate(
        results,
        start=1,
    ):
        print(
            f"\nResult {index}"
        )
        print("-" * 80)

        print(
            f"Distance: {score:.6f}"
        )

        print(
            f"Brand: "
            f"{document.metadata.get('brand')}"
        )

        print(
            f"Model: "
            f"{document.metadata.get('model')}"
        )

        print(
            f"Section: "
            f"{document.metadata.get('brochure_section')}"
        )

        print(
            f"Source: "
            f"{document.metadata.get('source_file')}"
        )

        print(
            f"Page: "
            f"{document.metadata.get('page')}"
        )

        print(
            f"Chunk ID: "
            f"{document.metadata.get('chunk_id')}"
        )

        print("\nContent:")
        print(
            document.page_content[:500]
        )


def validate_results(results):
    assert len(results) == RETRIEVAL_K

    for document, score in results:
        assert document.page_content.strip()

        assert document.metadata.get(
            "chunk_id"
        )

        assert document.metadata.get(
            "source_file"
        )

        assert document.metadata.get(
            "model"
        )

        assert isinstance(
            score,
            float,
        )


def run_similarity_search_test(
    vector_store,
    embedding_manager,
):
    query = (
        "What safety features does the car have?"
    )

    print("\nTesting similarity search...")

    query_embedding = (
        embedding_manager.embed_query(query)
    )

    assert len(query_embedding) == (
        EMBEDDING_DIMENSION
    )

    results = (
        vector_store.similarity_search_with_scores(
            query_embedding=query_embedding,
            k=RETRIEVAL_K,
        )
    )

    if not results:
        raise ValueError(
            "No documents were retrieved."
        )

    print(
        f"\nQuery: {query}"
    )

    display_results(results)
    validate_results(results)

    print(
        "\nSimilarity Search Validation Passed."
    )


def run_metadata_filter_test(
    vector_store,
    embedding_manager,
):
    brand = "Toyota"
    model = "Urban Cruiser Hyryder"

    query = (
        "What safety features does the car have?"
    )

    print("\nTesting metadata filtering...")

    query_embedding = (
        embedding_manager.embed_query(query)
    )

    results = (
        vector_store.similarity_search_with_scores(
            query_embedding=query_embedding,
            k=RETRIEVAL_K,
            brand=brand,
            model=model,
        )
    )

    if not results:
        raise ValueError(
            "No documents were retrieved "
            "for the selected brand and model."
        )

    print(
        f"\nBrand: {brand}"
    )

    print(
        f"Model: {model}"
    )

    display_results(results)

    # Every result must belong to the selected car.
    for document, score in results:
        assert (
            document.metadata.get("brand")
            == brand
        )

        assert (
            document.metadata.get("model")
            == model
        )

        assert isinstance(
            score,
            float,
        )

    print(
        "\nMetadata Filter Validation Passed."
    )


def main():
    print(
        "\nInitializing embedding manager..."
    )

    embedding_manager = EmbeddingManager()

    print(
        f"Embedding Model: "
        f"{embedding_manager.get_model_name()}"
    )

    vector_store = prepare_vector_store(
        embedding_manager
    )

    print(
        f"\nCollection: "
        f"{vector_store.get_collection_name()}"
    )

    print(
        f"Documents in collection: "
        f"{vector_store.count()}"
    )

    run_similarity_search_test(
        vector_store,
        embedding_manager,
    )

    run_metadata_filter_test(
        vector_store,
        embedding_manager,
    )

    print("\nValidation")
    print("=" * 80)

    print(
        "All vector store validation checks passed."
    )


if __name__ == "__main__":
    main()