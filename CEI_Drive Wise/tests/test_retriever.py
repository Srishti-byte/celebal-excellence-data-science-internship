from src.pdf_loader import PDFLoader
from src.text_cleaner import TextCleaner
from src.chunker import StructuredChunker
from src.embeddings import EmbeddingManager
from src.vector_store import VectorStore
from src.retriever import Retriever

from src.config import (
    CHROMA_PERSIST_DIRECTORY,
    CHROMA_TEST_COLLECTION_NAME,
    RETRIEVAL_K,
)


def prepare_test_vector_store(
    embedding_manager,
):
    vector_store = VectorStore(
        persist_directory=CHROMA_PERSIST_DIRECTORY,
        collection_name=CHROMA_TEST_COLLECTION_NAME,
    )

    if vector_store.count() > 0:
        print(
            f"Existing test collection found: "
            f"{vector_store.count()} documents"
        )
        return vector_store

    print("\nPreparing test vector store...")

    print("\nLoading documents...")

    loader = PDFLoader()
    documents = loader.load_documents()

    print(
        f"Documents Loaded: "
        f"{len(documents)}"
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
        f"Total Chunks: "
        f"{len(chunks)}"
    )

    print("\nGenerating embeddings...")

    embeddings = embedding_manager.embed_documents(
        chunks
    )

    print(
        f"Embeddings Generated: "
        f"{len(embeddings)}"
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
            f"Distance: "
            f"{score:.6f}"
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
        print(document.page_content)


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

        assert isinstance(score, float)


def run_automated_test(retriever):
    query = "What safety features does the car have?"

    print("\nAutomated Retrieval Test")
    print("=" * 80)

    print(
        f"Query: {query}"
    )

    results = retriever.retrieve_with_scores(
        query=query
    )

    if not results:
        raise ValueError(
            "Retriever returned no results."
        )

    display_results(results)

    validate_results(results)

    print("\nAutomated Validation")
    print("=" * 80)

    print(
        "All retriever validation checks passed."
    )


def run_metadata_filter_test(retriever):
    brand = "Toyota"
    model = "Urban Cruiser Hyryder"
    query = "What safety features does the car have?"

    print("\nMetadata Filtering Test")
    print("=" * 80)

    print(
        f"Brand: {brand}"
    )

    print(
        f"Model: {model}"
    )

    print(
        f"Query: {query}"
    )

    results = retriever.retrieve_with_scores(
        query=query,
        brand=brand,
        model=model,
    )

    if not results:
        raise ValueError(
            "Metadata-filtered retrieval returned no results."
        )

    display_results(results)

    for document, score in results:
        assert (
            document.metadata.get("brand")
            == brand
        )

        assert (
            document.metadata.get("model")
            == model
        )

        assert document.page_content.strip()

        assert document.metadata.get(
            "chunk_id"
        )

        assert isinstance(score, float)

    print("\nMetadata Filter Validation")
    print("=" * 80)

    print(
        "All retrieved documents belong to "
        "the selected brand and model."
    )


def run_interactive_test(retriever):
    print("\nInteractive Retrieval Test")
    print("=" * 80)

    print(
        "Enter questions to test retrieval."
    )

    print(
        "Enter 'q' to quit."
    )

    while True:
        query = input(
            "\nEnter your query: "
        ).strip()

        if query.lower() == "q":
            break

        if not query:
            print(
                "Please enter a query."
            )
            continue

        results = retriever.retrieve_with_scores(
            query=query
        )

        if not results:
            print(
                "\nNo results found."
            )
            continue

        display_results(results)


def main():
    print(
        "\nInitializing embedding manager..."
    )

    embedding_manager = EmbeddingManager()

    print(
        f"Embedding Model: "
        f"{embedding_manager.get_model_name()}"
    )

    print("\nInitializing vector store...")

    vector_store = prepare_test_vector_store(
        embedding_manager
    )

    print(
        f"Collection: "
        f"{vector_store.get_collection_name()}"
    )

    print(
        f"Documents in collection: "
        f"{vector_store.count()}"
    )

    print("\nInitializing retriever...")

    retriever = Retriever(
        vector_store=vector_store,
        embedding_manager=embedding_manager,
    )

    print(
        f"Top-K: "
        f"{retriever.get_top_k()}"
    )

    run_automated_test(
        retriever
    )

    run_metadata_filter_test(
        retriever
    )

    run_interactive_test(
        retriever
    )


if __name__ == "__main__":
    main()