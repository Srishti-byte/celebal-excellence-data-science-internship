"""Test file for the EmbeddingGenerator module."""
from modules.embeddings import EmbeddingGenerator

def main():
    # Embeddings Generator
    embedding_generator = EmbeddingGenerator()
    embedding_model = embedding_generator.get_embedding_model()
    sample_embedding = embedding_model.embed_query(
        "What is Retrieval-Augmented Generation?"
    )
    print(f"\nEmbedding Dimension: {len(sample_embedding)}")
    print("\nFirst 10 Values:\n")
    print(sample_embedding[:10])
    
if __name__ == "__main__":
    main()