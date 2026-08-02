"""Test file for the Reranker module."""
from modules.embeddings import EmbeddingGenerator
from modules.vector_store import VectorStoreManager
from modules.retriever import Retriever
from modules.reranker import Reranker

def main():
    embedding_generator = EmbeddingGenerator()
    embedding_model = embedding_generator.get_embedding_model()

    vector_store_manager = VectorStoreManager(embedding_model)
    vector_store = vector_store_manager.load_vector_store()

    retriever = Retriever(vector_store, embedding_model)

    query = "What is Retrieval-Augmented Generation?"
    retrieved_documents = retriever.retrieve_documents(query)

    # Rerank documents
    reranker = Reranker()
    reranked_documents = reranker.rerank_documents(
        query,
        retrieved_documents
    )

    print("\nReranked Documents\n")

    for i, result in enumerate(reranked_documents, start=1):
        document = result["document"]

        print("=" * 70)
        print(f"Result {i}")
        print("=" * 70)

        print(f"Relevance Score : {result['relevance_score']:.4f}")
        print(f"Chunk ID        : {document.metadata['chunk_id']}")
        print(f"Page            : {document.metadata['page']}")
        print(f"Source          : {document.metadata['source']}")

        print("\nContent:\n")
        print(document.page_content[:350])

        print()
        

if __name__ == "__main__":
    main()