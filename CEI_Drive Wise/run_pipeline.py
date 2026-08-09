import time

from src.config import COHERE_API_KEY
from src.embeddings import EmbeddingManager
from src.evaluator import Evaluator
from src.gemini_generator import GeminiGenerator
from src.logger import DriveWiseLogger
from src.prompt_generator import PromptGenerator
from src.reranker import Reranker
from src.retriever import Retriever
from src.vector_store import VectorStore


def initialize_pipeline():
    print("\nInitializing Drive Wise...")
    print("=" * 60)

    print("\nInitializing embedding manager...")
    embedding_manager = EmbeddingManager()

    print(
        f"Embedding Model: "
        f"{embedding_manager.get_model_name()}"
    )

    print("\nInitializing vector store...")
    vector_store = VectorStore()

    print(
        f"Collection: "
        f"{vector_store.get_collection_name()}"
    )
    print(f"Documents: {vector_store.count()}")

    if vector_store.count() == 0:
        raise ValueError(
            "Production vector store is empty. "
            "Run build_vector_store.py first."
        )

    print("\nInitializing retriever...")
    retriever = Retriever(
        vector_store=vector_store,
        embedding_manager=embedding_manager,
    )

    print(
        f"Retrieval Top-K: "
        f"{retriever.get_top_k()}"
    )

    print("\nInitializing reranker...")
    reranker = Reranker(
        api_key=COHERE_API_KEY
    )

    print(
        f"Reranker Model: "
        f"{reranker.get_model_name()}"
    )
    print(
        f"Reranker Top-N: "
        f"{reranker.get_top_n()}"
    )

    print("\nInitializing prompt generator...")
    prompt_generator = PromptGenerator()

    print("\nInitializing Gemini generator...")
    gemini_generator = GeminiGenerator()

    print(
        f"Gemini Model: "
        f"{gemini_generator.get_model_name()}"
    )

    print("\nInitializing evaluator...")
    evaluator = Evaluator(
        embedding_manager=embedding_manager
    )

    print("\nInitializing logger...")
    logger = DriveWiseLogger()

    print("\nDrive Wise initialized successfully.")
    print("=" * 60)

    return (
        retriever,
        reranker,
        prompt_generator,
        gemini_generator,
        evaluator,
        logger,
    )


def build_sources(documents):
    sources = []

    for document in documents:
        metadata = document.metadata

        sources.append(
            {
                "brand": metadata.get(
                    "brand", "Unknown"
                ),
                "model": metadata.get(
                    "model", "Unknown"
                ),
                "section": metadata.get(
                    "brochure_section", "Unknown"
                ),
                "source_file": metadata.get(
                    "source_file", "Unknown"
                ),
                "page": metadata.get(
                    "page_label",
                    metadata.get("page", "Unknown"),
                ),
                "chunk_id": metadata.get(
                    "chunk_id", "Unknown"
                ),
            }
        )

    return sources


def display_sources(sources):
    print("\n" + "=" * 60)
    print("SOURCES")
    print("=" * 60)

    for index, source in enumerate(
        sources,
        start=1,
    ):
        print(f"\nSource {index}")
        print(f"Brand: {source['brand']}")
        print(f"Model: {source['model']}")
        print(f"Section: {source['section']}")
        print(f"Source File: {source['source_file']}")
        print(f"Page: {source['page']}")
        print(f"Chunk ID: {source['chunk_id']}")


def display_evaluation(metrics):
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)

    for metric, score in metrics.items():
        print(f"{metric}: {score:.4f}")


def run_query(
    query,
    brand,
    model,
    retriever,
    reranker,
    prompt_generator,
    gemini_generator,
    evaluator,
    logger,
):
    start_time = time.perf_counter()

    try:
        logger.log_query(query)

        print("\nRetrieving documents...")

        retrieved_results = (
            retriever.retrieve_with_scores(
                query=query,
                brand=brand,
                model=model,
            )
        )

        if not retrieved_results:
            raise ValueError(
                "No documents were retrieved."
            )

        retrieved_documents = [
            document
            for document, _ in retrieved_results
        ]

        print(
            f"Retrieved documents: "
            f"{len(retrieved_documents)}"
        )

        print("\nReranking documents...")

        reranked_results = reranker.rerank(
            query=query,
            documents=retrieved_documents,
        )

        if not reranked_results:
            raise ValueError(
                "No documents remained after reranking."
            )

        reranked_documents = [
            document
            for document, _ in reranked_results
        ]

        print(
            f"Reranked documents: "
            f"{len(reranked_documents)}"
        )

        print("\nGenerating grounded prompt...")

        prompt = prompt_generator.generate_prompt(
            query=query,
            documents=reranked_documents,
        )

        print("\nGenerating answer...")

        answer = gemini_generator.generate_answer(
            prompt
        )

        print("\nEvaluating answer...")

        metrics = evaluator.evaluate(
            query=query,
            answer=answer,
            documents=reranked_documents,
        )

        response_time = (
            time.perf_counter() - start_time
        )

        sources = build_sources(
            reranked_documents
        )

        # Record the complete successful request.
        logger.log_response_time(
            response_time
        )

        logger.log_retrieval_results(
            f"{len(retrieved_documents)} documents "
            f"retrieved; "
            f"{len(reranked_documents)} documents "
            f"after reranking"
        )

        logger.log_generation_status(
            "success"
        )

        logger.log_evaluation_metrics(
            metrics
        )

        print("\n" + "=" * 60)
        print("ANSWER")
        print("=" * 60)
        print(answer)

        display_sources(sources)
        display_evaluation(metrics)

        print(
            f"\nResponse Time: "
            f"{response_time:.4f}s"
        )

        print("\nQuery completed successfully.")

        return {
            "answer": answer,
            "sources": sources,
            "evaluation": metrics,
            "response_time": response_time,
        }

    except Exception as error:
        response_time = (
            time.perf_counter() - start_time
        )

        logger.log_response_time(
            response_time
        )
        logger.log_failed_query(
            str(error)
        )

        print("\n" + "=" * 60)
        print("QUERY FAILED")
        print("=" * 60)
        print(error)

        return None


def main():
    (
        retriever,
        reranker,
        prompt_generator,
        gemini_generator,
        evaluator,
        logger,
    ) = initialize_pipeline()

    while True:
        print("\n" + "=" * 60)
        print("DRIVE WISE")
        print("=" * 60)

        brand = input(
            "\nEnter brand (or 'q' to quit): "
        ).strip()

        if brand.lower() == "q":
            break

        if not brand:
            print("Brand cannot be empty.")
            continue

        model = input(
            "Enter model: "
        ).strip()

        if not model:
            print("Model cannot be empty.")
            continue

        query = input(
            "Enter your question: "
        ).strip()

        if not query:
            print("Question cannot be empty.")
            continue

        run_query(
            query=query,
            brand=brand,
            model=model,
            retriever=retriever,
            reranker=reranker,
            prompt_generator=prompt_generator,
            gemini_generator=gemini_generator,
            evaluator=evaluator,
            logger=logger,
        )

        choice = input(
            "\nAsk another question? (y/n): "
        ).strip().lower()

        if choice != "y":
            break

    print("\nDrive Wise session ended.")


if __name__ == "__main__":
    main()