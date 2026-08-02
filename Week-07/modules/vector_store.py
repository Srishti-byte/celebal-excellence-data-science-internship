"""This module creates, saves, and loads the FAISS vector store."""
import os
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from config import VECTOR_STORE_PATH

class VectorStoreManager:
    """Creates and manages the FAISS vector store."""

    def __init__(self, embedding_model):
        self.embedding_model = embedding_model

    def create_vector_store(self, documents: list[Document]):
        """
        Create a FAISS vector store from processed documents.
        Args:
            documents: List of processed Document objects.
        Returns:
            FAISS vector store.
        """
        vector_store = FAISS.from_documents(
            documents=documents,
            embedding=self.embedding_model
        )

        print("=" * 50)
        print("Vector Store Created Successfully")
        print("=" * 50)

        return vector_store

    def save_vector_store(self, vector_store):
        """Save the FAISS vector store locally."""
        vector_store.save_local(VECTOR_STORE_PATH)

        print("=" * 50)
        print("Vector Store Saved Successfully")
        print("=" * 50)

    def load_vector_store(self):
        """
        Load the saved FAISS vector store.
        Returns:
            FAISS vector store.
        """

        if not os.path.exists(VECTOR_STORE_PATH):
            raise FileNotFoundError(
                "Vector store not found. Please create it first."
            )

        vector_store = FAISS.load_local(
            folder_path=VECTOR_STORE_PATH,
            embeddings=self.embedding_model,
            allow_dangerous_deserialization=True
        )

        print("=" * 50)
        print("Vector Store Loaded Successfully")
        print("=" * 50)

        return vector_store