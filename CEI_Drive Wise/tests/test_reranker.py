import os

from dotenv import load_dotenv

from src.embeddings import EmbeddingManager
from src.vector_store import VectorStore
from src.retriever import Retriever
from src.reranker import Reranker

from src.config import (
    CHROMA_PERSIST_DIRECTORY,
    CHROMA_TEST_COLLECTION_NAME,
    RETRIEVAL_K,
    RERANK_TOP_N,
)


def display_results(results, reranked=False):
    title = (
        "Cohere Reranked Results"
        if reranked
        else "Vector Retrieval Results"
    )

    print(f"\n{title}")
    print("=" * 80)

    for index, (document, score) in enumerate(
        results,
        start=1,
    ):
        print(f"\nResult {index}")
        print("-" * 80)

        if reranked:
            print(
                f"Cohere Relevance Score: "
                f"{score:.6f}"
            )
        else:
            print(
                f"Vector Distance: "
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
            f"Chunk ID: "
            f"{document.metadata.get('chunk_id')}"
        )

        print("\nContent:")
        print(document.page_content)


def validate_results(
    retrieved_results,
    reranked_results,
):
    assert len(retrieved_results) == RETRIEVAL_K

    expected_count = min(
        RERANK_TOP_N,
        len(retrieved_results),
    )

    assert len(reranked_results) == expected_count

    for document, score in reranked_results:
        assert document.page_content.strip()
        assert document.metadata.get("chunk_id")
        assert document.metadata.get("source_file")
        assert document.metadata.get("model")
        assert isinstance(score, float)

    scores = [
        score
        for _, score in reranked_results
    ]

    assert scores == sorted(
        scores,
        reverse=True,
    )


def run_test(
    query,
    retriever,
    reranker,
):
    print("\nQuery")
    print("=" * 80)
    print(query)

    retrieved_results = (
        retriever.retrieve_with_scores(
            query=query
        )
    )

    if not retrieved_results:
        raise ValueError(
            "Retriever returned no results."
        )

    display_results(
        retrieved_results
    )

    documents = [
        document
        for document, _ in retrieved_results
    ]

    reranked_results = reranker.rerank(
        query=query,
        documents=documents,
    )

    if not reranked_results:
        raise ValueError(
            "Reranker returned no results."
        )

    display_results(
        reranked_results,
        reranked=True,
    )

    validate_results(
        retrieved_results,
        reranked_results,
    )

    print("\nValidation")
    print("=" * 80)

    print(
        "Reranker validation checks passed."
    )


def main():
    load_dotenv()

    api_key = os.getenv(
        "COHERE_API_KEY"
    )

    if not api_key:
        raise ValueError(
            "COHERE_API_KEY was not found "
            "in the environment."
        )

    print("\nInitializing embedding manager...")

    embedding_manager = EmbeddingManager()

    print(
        f"Embedding Model: "
        f"{embedding_manager.get_model_name()}"
    )

    print("\nInitializing vector store...")

    vector_store = VectorStore(
        persist_directory=CHROMA_PERSIST_DIRECTORY,
        collection_name=CHROMA_TEST_COLLECTION_NAME,
    )

    if vector_store.count() == 0:
        raise ValueError(
            "Vector store is empty. "
            "Run test_vector_store.py first."
        )

    print(
        f"Documents in collection: "
        f"{vector_store.count()}"
    )

    print("\nInitializing retriever...")

    retriever = Retriever(
        vector_store=vector_store,
        embedding_manager=embedding_manager,
        top_k=RETRIEVAL_K,
    )

    print(
        f"Retriever Top-K: "
        f"{retriever.get_top_k()}"
    )

    print("\nInitializing reranker...")

    reranker = Reranker(
        api_key=api_key,
    )

    print(
        f"Reranker Model: "
        f"{reranker.get_model_name()}"
    )

    print(
        f"Reranker Top-N: "
        f"{reranker.get_top_n()}"
    )

    query = (
        "Does Toyota Urban Cruiser Hyryder "
        "have ADAS?"
    )

    run_test(
        query=query,
        retriever=retriever,
        reranker=reranker,
    )

    print("\nInteractive Reranking Test")
    print("=" * 80)

    while True:
        query = input(
            "\nEnter your query "
            "(or 'q' to quit): "
        ).strip()

        if query.lower() == "q":
            break

        if not query:
            print("Please enter a query.")
            continue

        run_test(
            query=query,
            retriever=retriever,
            reranker=reranker,
        )


if __name__ == "__main__":
    main()