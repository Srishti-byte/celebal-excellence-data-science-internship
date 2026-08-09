from typing import List, Optional, Tuple

import chromadb
from langchain_core.documents import Document

from src.config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_PERSIST_DIRECTORY,
    RETRIEVAL_K,
)


class VectorStore:
    def __init__(
        self,
        persist_directory: str = CHROMA_PERSIST_DIRECTORY,
        collection_name: str = CHROMA_COLLECTION_NAME,
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name

        self.client = chromadb.PersistentClient(
            path=self.persist_directory
        )

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name
        )

    def add_documents(
        self,
        documents: List[Document],
        embeddings: List[List[float]],
    ) -> None:
        if not documents:
            raise ValueError(
                "No documents were provided."
            )

        if len(documents) != len(embeddings):
            raise ValueError(
                "Number of documents and embeddings must match."
            )

        ids = []

        for document in documents:
            chunk_id = document.metadata.get("chunk_id")

            if not chunk_id:
                raise ValueError(
                    "Every document must have a chunk_id."
                )

            ids.append(str(chunk_id))

        self.collection.upsert(
            ids=ids,
            documents=[
                document.page_content
                for document in documents
            ],
            embeddings=embeddings,
            metadatas=[
                document.metadata
                for document in documents
            ],
        )

    def _build_metadata_filter(
        self,
        brand: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Optional[dict]:
        if not brand and not model:
            return None

        filters = []

        if brand:
            filters.append({"brand": brand})

        if model:
            filters.append({"model": model})

        if len(filters) == 1:
            return filters[0]

        return {"$and": filters}

    def similarity_search(
        self,
        query_embedding: List[float],
        k: int = RETRIEVAL_K,
        brand: Optional[str] = None,
        model: Optional[str] = None,
    ) -> List[Document]:
        if not query_embedding:
            raise ValueError(
                "Query embedding cannot be empty."
            )

        if k <= 0:
            raise ValueError(
                "k must be greater than zero."
            )

        query_params = {
            "query_embeddings": [query_embedding],
            "n_results": k,
            "include": [
                "documents",
                "metadatas",
                "distances",
            ],
        }

        metadata_filter = self._build_metadata_filter(
            brand=brand,
            model=model,
        )

        if metadata_filter:
            query_params["where"] = metadata_filter

        results = self.collection.query(
            **query_params
        )

        documents = results.get(
            "documents", [[]]
        )[0]

        metadatas = results.get(
            "metadatas", [[]]
        )[0]

        return [
            Document(
                page_content=content,
                metadata=metadata or {},
            )
            for content, metadata in zip(
                documents,
                metadatas,
            )
        ]

    def similarity_search_with_scores(
        self,
        query_embedding: List[float],
        k: int = RETRIEVAL_K,
        brand: Optional[str] = None,
        model: Optional[str] = None,
    ) -> List[Tuple[Document, float]]:
        if not query_embedding:
            raise ValueError(
                "Query embedding cannot be empty."
            )

        if k <= 0:
            raise ValueError(
                "k must be greater than zero."
            )

        query_params = {
            "query_embeddings": [query_embedding],
            "n_results": k,
            "include": [
                "documents",
                "metadatas",
                "distances",
            ],
        }

        metadata_filter = self._build_metadata_filter(
            brand=brand,
            model=model,
        )

        if metadata_filter:
            query_params["where"] = metadata_filter

        results = self.collection.query(
            **query_params
        )

        documents = results.get(
            "documents", [[]]
        )[0]

        metadatas = results.get(
            "metadatas", [[]]
        )[0]

        distances = results.get(
            "distances", [[]]
        )[0]

        return [
            (
                Document(
                    page_content=content,
                    metadata=metadata or {},
                ),
                float(distance),
            )
            for content, metadata, distance in zip(
                documents,
                metadatas,
                distances,
            )
        ]

    def count(self) -> int:
        return self.collection.count()

    def get_collection_name(self) -> str:
        return self.collection_name

    def clear(self) -> None:
        self.client.delete_collection(
            name=self.collection_name
        )

        self.collection = (
            self.client.get_or_create_collection(
                name=self.collection_name
            )
        )