from typing import List

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import (
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,
    EMBEDDING_DEVICE,
)


QUERY_INSTRUCTION = (
    "Represent this sentence for searching relevant passages: "
)


class EmbeddingManager:
    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL,
        device: str = EMBEDDING_DEVICE,
    ):
        self.model_name = model_name
        self.device = device

        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={
                "device": self.device,
            },
            encode_kwargs={
                "normalize_embeddings": True,
            },
        )

    def embed_documents(
        self,
        documents: List[Document],
    ) -> List[List[float]]:
        if not documents:
            return []

        texts = [
            document.page_content.strip()
            for document in documents
            if document.page_content.strip()
        ]

        if not texts:
            return []

        return self.embeddings.embed_documents(texts)

    def embed_query(self, query: str) -> List[float]:
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        instructed_query = (
            QUERY_INSTRUCTION + query.strip()
        )

        return self.embeddings.embed_query(
            instructed_query
        )

    def get_embedding_dimension(self) -> int:
        return EMBEDDING_DIMENSION

    def get_model_name(self) -> str:
        return self.model_name