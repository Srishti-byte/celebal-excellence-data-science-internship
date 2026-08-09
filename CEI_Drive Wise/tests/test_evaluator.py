from src.config import (
    CHROMA_PERSIST_DIRECTORY,
    CHROMA_TEST_COLLECTION_NAME,
    COHERE_API_KEY,
    RETRIEVAL_K,
)
from src.embeddings import EmbeddingManager
from src.evaluator import Evaluator
from src.reranker import Reranker
from src.retriever import Retriever
from src.vector_store import VectorStore


def main():
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

    print(
        f"Documents in collection: "
        f"{vector_store.count()}"
    )

    if vector_store.count() == 0:
        raise ValueError(
            "Test vector store is empty."
        )

    print("\nInitializing retriever...")

    retriever = Retriever(
        vector_store=vector_store,
        embedding_manager=embedding_manager,
        top_k=RETRIEVAL_K,
    )

    print("\nInitializing reranker...")

    reranker = Reranker(
        api_key=COHERE_API_KEY
    )

    print("\nInitializing evaluator...")

    evaluator = Evaluator(
        embedding_manager=embedding_manager
    )

    query = (
    "What safety features does the "
    "Toyota Urban Cruiser Hyryder have?"
)
    answer = (
    "The Toyota Urban Cruiser Hyryder has "
    "six airbags and ABS with EBD."
)

    print("\nQuery")
    print("=" * 80)
    print(query)

    print("\nRetrieving candidates...")

    retrieved_results = (
        retriever.retrieve_with_scores(
            query=query
        )
    )

    documents = [
        document
        for document, _ in retrieved_results
    ]

    print(
        f"Retrieved documents: "
        f"{len(documents)}"
    )

    if not documents:
        raise ValueError(
            "No documents were retrieved."
        )

    print("\nReranking candidates...")

    reranked_results = reranker.rerank(
        query=query,
        documents=documents,
    )

    reranked_documents = [
        document
        for document, _ in reranked_results
    ]

    print(
        f"Reranked documents: "
        f"{len(reranked_documents)}"
    )

    print("\nReranked Context")
    print("=" * 80)
    
    for index, document in enumerate(
    reranked_documents,
    start=1,
    ):
        print(f"\nContext {index}")
        print("-" * 80)
        print(document.page_content)

    if not reranked_documents:
        raise ValueError(
            "No documents remained after reranking."
        )

    print("\nGenerated Answer")
    print("=" * 80)
    print(answer)

    print("\nRunning evaluation...")

    metrics = evaluator.evaluate(
        query=query,
        answer=answer,
        documents=reranked_documents,
    )

    print("\nEvaluation Results")
    print("=" * 80)

    for metric, score in metrics.items():
        print(
            f"{metric}: {score:.4f}"
        )

    print("\nValidation")
    print("=" * 80)

    assert set(metrics) == {
        "answer_correctness",
        "faithfulness",
        "context_relevance",
    }

    for score in metrics.values():
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    print(
        "All evaluator validation checks passed."
    )


if __name__ == "__main__":
    main()