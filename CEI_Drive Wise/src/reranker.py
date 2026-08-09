from typing import List, Tuple

import cohere
from langchain_core.documents import Document

from src.config import (
    RERANK_MODEL,
    RERANK_TOP_N,
)


class Reranker:
    def __init__(
        self,
        api_key: str,
        model_name: str = RERANK_MODEL,
        top_n: int = RERANK_TOP_N,
    ):
        if not api_key or not api_key.strip():
            raise ValueError(
                "Cohere API key cannot be empty."
            )

        if top_n <= 0:
            raise ValueError(
                "top_n must be greater than zero."
            )

        self.api_key = api_key
        self.model_name = model_name
        self.top_n = top_n

        self.client = cohere.ClientV2(
            api_key=self.api_key
        )

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_n: int | None = None,
    ) -> List[Tuple[Document, float]]:
        if not query or not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        if not documents:
            return []

        requested_top_n = (
            top_n
            if top_n is not None
            else self.top_n
        )

        if requested_top_n <= 0:
            raise ValueError(
                "top_n must be greater than zero."
            )

        requested_top_n = min(
            requested_top_n,
            len(documents),
        )

        rerank_documents = [
            self._format_document(document)
            for document in documents
        ]

        response = self.client.rerank(
            model=self.model_name,
            query=query.strip(),
            documents=rerank_documents,
            top_n=requested_top_n,
        )

        reranked_documents = []

        for result in response.results:
            original_document = documents[result.index]

            reranked_documents.append(
                (
                    original_document,
                    float(result.relevance_score),
                )
            )

        return reranked_documents

    def _format_document(
        self,
        document: Document,
    ) -> str:
        metadata = document.metadata

        brand = metadata.get("brand", "Unknown")
        model = metadata.get("model", "Unknown")
        section = metadata.get(
            "brochure_section",
            "Unknown",
        )
        source_file = metadata.get(
            "source_file",
            "Unknown",
        )
        page = metadata.get(
            "page_label",
            metadata.get("page", "Unknown"),
        )

        return (
            f"Brand: {brand}\n"
            f"Model: {model}\n"
            f"Brochure Section: {section}\n"
            f"Source File: {source_file}\n"
            f"Page: {page}\n\n"
            f"Content:\n"
            f"{document.page_content}"
        )

    def get_model_name(self) -> str:
        return self.model_name

    def get_top_n(self) -> int:
        return self.top_n