"""This module initializes the Hugging Face embedding model."""
from langchain_huggingface import HuggingFaceEmbeddings
from config import EMBEDDING_MODEL

class EmbeddingGenerator:
    """Initializes the embedding model used for semantic search."""
    def __init__(self):
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL
        )

    def get_embedding_model(self):
        """
        Return the initialized embedding model.
        Returns:
            HuggingFaceEmbeddings
        """
        print("=" * 50)
        print("Embedding Model Initialized")
        print(f"Model: {EMBEDDING_MODEL}")
        print("=" * 50)

        return self.embedding_model