"""Test file for the VectorStoreManager module."""

from modules.loader import DocumentLoader
from modules.chunker import TextChunker
from modules.metadata_manager import MetadataManager
from modules.embeddings import EmbeddingGenerator
from modules.vector_store import VectorStoreManager

def main():
    loader = DocumentLoader("data/rag_paper.pdf")
    documents = loader.load_document()

    chunker = TextChunker()
    chunks = chunker.split_documents(documents)

    metadata_manager = MetadataManager()
    processed_chunks = metadata_manager.process_metadata(chunks)

    embedding_generator = EmbeddingGenerator()
    embedding_model = embedding_generator.get_embedding_model()

    # Vector Store
    vector_store_manager = VectorStoreManager(embedding_model)
    vector_store = vector_store_manager.create_vector_store(
        processed_chunks
    )

    vector_store_manager.save_vector_store(vector_store)
    loaded_vector_store = vector_store_manager.load_vector_store()
    print("\nTotal Documents in Vector Store:")
    print(loaded_vector_store.index.ntotal)

if __name__ == "__main__":
    main()