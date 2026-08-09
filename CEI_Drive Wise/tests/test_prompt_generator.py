import os

from dotenv import load_dotenv

from src.embeddings import EmbeddingManager
from src.vector_store import VectorStore
from src.retriever import Retriever
from src.reranker import Reranker
from src.prompt_generator import PromptGenerator

from src.config import (
    CHROMA_PERSIST_DIRECTORY,
    CHROMA_TEST_COLLECTION_NAME,
    RETRIEVAL_K,
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

    print(
        "\nInitializing embedding manager..."
    )

    embedding_manager = EmbeddingManager()

    print(
        f"Embedding Model: "
        f"{embedding_manager.get_model_name()}"
    )

    print(
        "\nInitializing vector store..."
    )

    vector_store = VectorStore(
        persist_directory=CHROMA_PERSIST_DIRECTORY,
        collection_name=CHROMA_TEST_COLLECTION_NAME,
    )

    if vector_store.count() == 0:
        raise ValueError(
            "Vector store is empty."
        )

    print(
        f"Documents in collection: "
        f"{vector_store.count()}"
    )

    print(
        "\nInitializing retriever..."
    )

    retriever = Retriever(
        vector_store=vector_store,
        embedding_manager=embedding_manager,
        top_k=RETRIEVAL_K,
    )

    print(
        "\nInitializing reranker..."
    )

    reranker = Reranker(
        api_key=api_key,
    )

    print(
        "\nInitializing prompt generator..."
    )

    prompt_generator = PromptGenerator()

    query = (
        "Does Toyota Urban Cruiser Hyryder "
        "have ADAS?"
    )

    print("\nQuery")
    print("=" * 80)
    print(query)

    print(
        "\nRetrieving candidates..."
    )

    retrieved_results = (
        retriever.retrieve_with_scores(
            query=query
        )
    )

    if not retrieved_results:
        raise ValueError(
            "Retriever returned no results."
        )

    documents = [
        document
        for document, _ in retrieved_results
    ]

    print(
        f"Retrieved documents: "
        f"{len(documents)}"
    )

    print(
        "\nReranking candidates..."
    )

    reranked_results = reranker.rerank(
        query=query,
        documents=documents,
    )

    if not reranked_results:
        raise ValueError(
            "Reranker returned no results."
        )

    reranked_documents = [
        document
        for document, _ in reranked_results
    ]

    print(
        f"Reranked documents: "
        f"{len(reranked_documents)}"
    )

    print(
        "\nGenerating prompt..."
    )

    prompt = prompt_generator.generate_prompt(
        query=query,
        documents=reranked_documents,
    )

    print("\nGenerated Prompt")
    print("=" * 80)
    print(prompt)

    print("\nValidation")
    print("=" * 80)

    assert prompt.strip()

    assert query in prompt

    assert "CONTEXT" in prompt

    assert "QUESTION" in prompt

    assert "ANSWER" in prompt

    assert (
        "provided brochure context"
        in prompt
    )

    for document in reranked_documents:
        assert (
            document.page_content.strip()
            in prompt
        )

    print(
        "All prompt generator validation "
        "checks passed."
    )


if __name__ == "__main__":
    main()