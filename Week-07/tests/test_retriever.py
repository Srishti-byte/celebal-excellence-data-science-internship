"""Test file for the Retriever module."""
from modules.embeddings import EmbeddingGenerator
from modules.vector_store import VectorStoreManager
from modules.retriever import Retriever

def main():
    embedding_generator = EmbeddingGenerator()
    embedding_model = embedding_generator.get_embedding_model()

    vector_store_manager = VectorStoreManager(embedding_model)
    vector_store = vector_store_manager.load_vector_store()

    # Retrieve Documents
    retriever = Retriever(vector_store, embedding_model)

    # Test Query
    query = "What is Retrieval-Augmented Generation?"
    retrieved_documents = retriever.retrieve_documents(query)

    print("\nRetrieved Documents\n")

    for i, document in enumerate(retrieved_documents, start=1):
        print("=" * 70)
        print(f"Result {i}")
        print("=" * 70)

        print(f"Chunk ID : {document.metadata['chunk_id']}")
        print(f"Page     : {document.metadata['page']}")
        print(f"Source   : {document.metadata['source']}")

        print("\nContent:\n")
        print(document.page_content[:350])

        print()


if __name__ == "__main__":
    main()