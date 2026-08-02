"""Test file for the Prompt Builder module."""
from modules.embeddings import EmbeddingGenerator
from modules.vector_store import VectorStoreManager
from modules.retriever import Retriever
from modules.reranker import Reranker
from modules.prompt_builder import PromptBuilder


def main():
    embedding_generator = EmbeddingGenerator()
    embedding_model = embedding_generator.get_embedding_model()

    vector_store_manager = VectorStoreManager(embedding_model)
    vector_store = vector_store_manager.load_vector_store()

    retriever = Retriever(vector_store, embedding_model)

    query = "What is Retrieval-Augmented Generation?"
    retrieved_documents = retriever.retrieve_documents(query)

    reranker = Reranker()
    reranked_documents = reranker.rerank_documents(
        query,
        retrieved_documents
    )

    # Prompt Builder
    prompt_builder = PromptBuilder()
    prompt = prompt_builder.build_prompt(
        query,
        reranked_documents
    )

    print("\nGenerated Prompt\n")
    print("=" * 80)
    print(prompt)


if __name__ == "__main__":
    main()