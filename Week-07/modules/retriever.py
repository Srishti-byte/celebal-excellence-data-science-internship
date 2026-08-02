"""This module retrieves the most relevant document chunks from the FAISS vector store."""
from langchain_core.documents import Document
from config import TOP_K

class Retriever:
    """Retrieves relevant document chunks from the vector store."""
    def __init__(self, vector_store, embedding_model):
        self.vector_store = vector_store
        self.embedding_model = embedding_model

    def retrieve_documents(self, query: str) -> list[Document]:
        """
        Retrieve the most relevant document chunks for a user query.
        Args:
            query: User query.
        Returns:
            List of relevant Document objects.
        """

        print("=" * 50)
        print("Generating Query Embedding...")
        print("=" * 50)
        query_embedding = self.embedding_model.embed_query(query)

        print(f"Embedding Dimension : {len(query_embedding)}")
        print("=" * 50)
        print("Searching Vector Store...")
        print("=" * 50)

        retrieved_documents = self.vector_store.similarity_search_by_vector(
            embedding=query_embedding,
            k=TOP_K
        )

        print(f"Retrieved Chunks : {len(retrieved_documents)}")
        print("=" * 50)

        return retrieved_documents