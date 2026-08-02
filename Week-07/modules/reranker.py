"""This module reranks retrieved documents using Cohere's Rerank API."""

import cohere
from config import (
    COHERE_API_KEY,
    RERANK_MODEL,
    RERANK_TOP_K
)

class Reranker:
    """Reranks retrieved documents based on their relevance to the user query."""

    def __init__(self):
        self.client = cohere.Client(COHERE_API_KEY)

    def rerank_documents(self, query, retrieved_documents):
        """
        Rerank retrieved documents.
        Args:
            query: User query.
            retrieved_documents: List of LangChain Document objects.
        Returns:
            List of dictionaries containing the reranked document
            and its relevance score.
        """

        document_texts = [
            document.page_content
            for document in retrieved_documents
        ]

        response = self.client.rerank(
            model=RERANK_MODEL,
            query=query,
            documents=document_texts,
            top_n=RERANK_TOP_K
        )

        reranked_results = []

        for result in response.results:
            reranked_results.append(
                {
                    "document": retrieved_documents[result.index],
                    "relevance_score": result.relevance_score
                }
            )

        print("=" * 50)
        print("Document Reranking Completed")
        print(f"Returned Documents : {len(reranked_results)}")
        print("=" * 50)

        return reranked_results