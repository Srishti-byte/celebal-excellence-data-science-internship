import time
from typing import Optional

from langchain_core.documents import Document

from src.evaluator import Evaluator
from src.logger import DriveWiseLogger
from src.prompt_generator import PromptGenerator
from src.reranker import Reranker
from src.retriever import Retriever
from src.gemini_generator import GeminiGenerator


class RAGPipeline:
    def __init__(
        self,
        retriever: Retriever,
        reranker: Reranker,
        prompt_generator: PromptGenerator,
        gemini_generator: GeminiGenerator,
        evaluator: Evaluator,
        logger: DriveWiseLogger,
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.prompt_generator = prompt_generator
        self.gemini_generator = gemini_generator
        self.evaluator = evaluator
        self.logger = logger

    def run(
        self,
        query: str,
        brand: str,
        model: str,
        reference_answer: Optional[str] = None,
    ) -> dict:
        if not query or not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        if not brand or not brand.strip():
            raise ValueError(
                "Brand cannot be empty."
            )

        if not model or not model.strip():
            raise ValueError(
                "Model cannot be empty."
            )

        start_time = time.perf_counter()

        try:
            self.logger.log_query(
                query.strip()
            )

            retrieved_results = (
                self.retriever.retrieve_with_scores(
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

            reranked_results = (
                self.reranker.rerank(
                    query=query,
                    documents=retrieved_documents,
                )
            )

            if not reranked_results:
                raise ValueError(
                    "No documents remained after reranking."
                )

            reranked_documents = [
                document
                for document, _ in reranked_results
            ]

            prompt = (
                self.prompt_generator.generate_prompt(
                    query=query,
                    documents=reranked_documents,
                )
            )

            answer = (
                self.gemini_generator.generate_answer(
                    prompt
                )
            )

            metrics = self.evaluator.evaluate(
                query=query,
                answer=answer,
                documents=reranked_documents,
                reference_answer=reference_answer,
            )

            response_time = (
                time.perf_counter()
                - start_time
            )

            self.logger.log_request(
                query=query,
                response_time=response_time,
                retrieval_results=(
                    f"{len(retrieved_documents)} "
                    f"documents retrieved; "
                    f"{len(reranked_documents)} "
                    f"documents after reranking"
                ),
                generation_status="success",
            )

            self.logger.log_evaluation_metrics(
                metrics
            )

            return {
                "answer": answer,
                "sources": self._build_sources(
                    reranked_documents
                ),
                "evaluation": metrics,
                "response_time": response_time,
            }

        except Exception as error:
            response_time = (
                time.perf_counter()
                - start_time
            )

            self.logger.log_response_time(
                response_time
            )

            self.logger.log_failed_query(
                str(error)
            )

            raise

    def _build_sources(
        self,
        documents: list[Document],
    ) -> list[dict]:
        sources = []

        for document in documents:
            metadata = document.metadata

            sources.append(
                {
                    "brand": metadata.get(
                        "brand",
                        "Unknown",
                    ),
                    "model": metadata.get(
                        "model",
                        "Unknown",
                    ),
                    "section": metadata.get(
                        "brochure_section",
                        "Unknown",
                    ),
                    "source_file": metadata.get(
                        "source_file",
                        "Unknown",
                    ),
                    "page": metadata.get(
                        "page_label",
                        metadata.get(
                            "page",
                            "Unknown",
                        ),
                    ),
                    "chunk_id": metadata.get(
                        "chunk_id",
                        "Unknown",
                    ),
                }
            )

        return sources