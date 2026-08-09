from typing import List, Optional, Tuple

from langchain_core.documents import Document

from src.config import RETRIEVAL_K
from src.embeddings import EmbeddingManager
from src.vector_store import VectorStore


class Retriever:
    def __init__(
        self,
        vector_store: VectorStore,
        embedding_manager: EmbeddingManager,
        top_k: int = RETRIEVAL_K,
    ):
        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        self.vector_store = vector_store
        self.embedding_manager = embedding_manager
        self.top_k = top_k

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        brand: Optional[str] = None,
        model: Optional[str] = None,
    ) -> List[Document]:
        if not query or not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        k = (
            top_k
            if top_k is not None
            else self.top_k
        )

        if k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        query_embedding = (
            self.embedding_manager.embed_query(
                query
            )
        )

        return self.vector_store.similarity_search(
            query_embedding=query_embedding,
            k=k,
            brand=brand,
            model=model,
        )

    def retrieve_with_scores(
        self,
        query: str,
        top_k: Optional[int] = None,
        brand: Optional[str] = None,
        model: Optional[str] = None,
    ) -> List[Tuple[Document, float]]:
        if not query or not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        k = (
            top_k
            if top_k is not None
            else self.top_k
        )

        if k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        query_embedding = (
            self.embedding_manager.embed_query(
                query
            )
        )

        return (
            self.vector_store
            .similarity_search_with_scores(
                query_embedding=query_embedding,
                k=k,
                brand=brand,
                model=model,
            )
        )

    def get_top_k(self) -> int:
        return self.top_k