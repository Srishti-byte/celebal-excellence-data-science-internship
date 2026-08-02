from modules.embeddings import EmbeddingGenerator
from modules.vector_store import VectorStoreManager
from modules.retriever import Retriever
from modules.reranker import Reranker
from modules.prompt_builder import PromptBuilder
from modules.gemini_generator import GeminiGenerator
from modules.rag_pipeline import RAGPipeline


def main():
    embedding_generator = EmbeddingGenerator()
    embedding_model = embedding_generator.get_embedding_model()

    vector_store_manager = VectorStoreManager(embedding_model)
    vector_store = vector_store_manager.load_vector_store()

    retriever = Retriever(
        vector_store,
        embedding_model
    )

    reranker = Reranker()

    prompt_builder = PromptBuilder()

    gemini_generator = GeminiGenerator()

    pipeline = RAGPipeline(
        retriever,
        reranker,
        prompt_builder,
        gemini_generator
    )

    query = "What is Retrieval-Augmented Generation?"

    result = pipeline.ask(query)

    print("\nGenerated Answer\n")
    print("=" * 80)
    print(result["answer"])

    print("\nRetrieved Sources\n")

    for index, source in enumerate(result["sources"], start=1):

        document = source["document"]

        print("=" * 60)
        print(f"Source {index}")
        print("=" * 60)

        print(f"Score    : {source['relevance_score']:.4f}")
        print(f"Page     : {document.metadata['page']}")
        print(f"Chunk ID : {document.metadata['chunk_id']}")
        print(f"File     : {document.metadata['source']}")

        print("\nPreview:\n")
        print(document.page_content[:250])

        print()


if __name__ == "__main__":
    main()