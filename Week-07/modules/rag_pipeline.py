"""Coordinates the complete Retrieval-Augmented Generation (RAG) pipeline."""

class RAGPipeline:
    """End-to-end RAG pipeline."""
    def __init__(
        self,
        retriever,
        reranker,
        prompt_builder,
        gemini_generator
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.prompt_builder = prompt_builder
        self.gemini_generator = gemini_generator

        print("=" * 50)
        print("RAG Pipeline Initialized")
        print("=" * 50)

    def ask(self, query: str) -> dict:
        print("\nExecuting RAG Pipeline...\n")

        retrieved_documents = self.retriever.retrieve_documents(query)

        reranked_documents = self.reranker.rerank_documents(
            query,
            retrieved_documents
        )

        prompt = self.prompt_builder.build_prompt(
            query,
            reranked_documents
        )

        answer = self.gemini_generator.generate_answer(prompt)

        print("=" * 50)
        print("Pipeline Execution Completed")
        print("=" * 50)

        return {
            "answer": answer,
            "sources": reranked_documents
        }