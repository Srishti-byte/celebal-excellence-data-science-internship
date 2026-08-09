from langchain_core.documents import Document

from src.rag_pipeline import RAGPipeline


class FakeRetriever:
    def retrieve_with_scores(
        self,
        query,
        brand,
        model,
    ):
        document = Document(
            page_content=(
                "The vehicle has six airbags "
                "and ABS with EBD."
            ),
            metadata={
                "brand": brand,
                "model": model,
                "brochure_section": "Safety",
                "source_file": "test_brochure.pdf",
                "page": 12,
                "chunk_id": "test_chunk_1",
            },
        )

        return [
            (document, 0.25),
        ]


class FakeReranker:
    def rerank(
        self,
        query,
        documents,
    ):
        return [
            (
                documents[0],
                0.95,
            )
        ]


class FakePromptGenerator:
    def generate_prompt(
        self,
        query,
        documents,
    ):
        return (
            "Answer using the provided context."
        )


class FakeGeminiGenerator:
    def generate_answer(
        self,
        prompt,
    ):
        return (
            "The vehicle has six airbags "
            "and ABS with EBD."
        )


class FakeEvaluator:
    def evaluate(
        self,
        query,
        answer,
        documents,
        reference_answer=None,
    ):
        return {
            "answer_correctness": None,
            "faithfulness": 0.90,
            "context_relevance": 0.92,
        }


class FakeLogger:
    def __init__(self):
        self.events = []

    def log_query(self, query):
        self.events.append(
            ("query", query)
        )

    def log_response_time(self, response_time):
        self.events.append(
            ("response_time", response_time)
        )

    def log_failed_query(self, error):
        self.events.append(
            ("failed_query", error)
        )

    def log_retrieval_results(self, results):
        self.events.append(
            ("retrieval", results)
        )

    def log_generation_status(self, status):
        self.events.append(
            ("generation", status)
        )

    def log_evaluation_metrics(self, metrics):
        self.events.append(
            ("evaluation", metrics)
        )

    def log_request(
        self,
        query,
        response_time,
        retrieval_results,
        generation_status,
    ):
        self.log_query(query)
        self.log_response_time(
            response_time
        )
        self.log_retrieval_results(
            retrieval_results
        )
        self.log_generation_status(
            generation_status
        )


def main():
    print("\nInitializing test pipeline...")

    logger = FakeLogger()

    pipeline = RAGPipeline(
        retriever=FakeRetriever(),
        reranker=FakeReranker(),
        prompt_generator=FakePromptGenerator(),
        gemini_generator=FakeGeminiGenerator(),
        evaluator=FakeEvaluator(),
        logger=logger,
    )

    query = (
        "What safety features does it have?"
    )

    brand = "Toyota"
    model = "Urban Cruiser Hyryder"

    print("\nRunning RAG pipeline...")

    result = pipeline.run(
        query=query,
        brand=brand,
        model=model,
    )

    print("\nAnswer")
    print("=" * 80)
    print(result["answer"])

    print("\nSources")
    print("=" * 80)

    for source in result["sources"]:
        print(source)

    print("\nEvaluation")
    print("=" * 80)

    for metric, score in result[
        "evaluation"
    ].items():
        print(
            f"{metric}: {score}"
        )

    print("\nValidation")
    print("=" * 80)

    assert result["answer"]

    assert len(
        result["sources"]
    ) == 1

    source = result["sources"][0]

    assert source["brand"] == brand
    assert source["model"] == model
    assert source["section"] == "Safety"
    assert source["source_file"] == (
        "test_brochure.pdf"
    )
    assert source["page"] == 12
    assert source["chunk_id"] == (
        "test_chunk_1"
    )

    assert (
        result["evaluation"][
            "faithfulness"
        ]
        == 0.90
    )

    assert (
        result["evaluation"][
            "context_relevance"
        ]
        == 0.92
    )

    assert any(
        event[0] == "generation"
        and event[1] == "success"
        for event in logger.events
    )

    assert any(
        event[0] == "evaluation"
        for event in logger.events
    )

    print(
        "All RAG pipeline validation checks passed."
    )


if __name__ == "__main__":
    main()