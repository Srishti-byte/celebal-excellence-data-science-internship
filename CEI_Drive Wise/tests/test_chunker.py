from collections import Counter

from src.chunker import StructuredChunker
from src.pdf_loader import PDFLoader
from src.text_cleaner import TextCleaner


def print_chunk(chunk, index):
    print("\n" + "=" * 80)
    print(f"Chunk {index}")
    print("=" * 80)

    print("\nMetadata")
    print("-" * 80)

    for key, value in chunk.metadata.items():
        print(f"{key}: {value}")

    print("\nContent")
    print("-" * 80)
    print(chunk.page_content)

    print("=" * 80)


def validate_chunks(chunks):
    assert chunks, "No chunks were generated."

    chunk_ids = [
        chunk.metadata.get("chunk_id")
        for chunk in chunks
    ]

    assert all(chunk_ids), "One or more chunks are missing chunk_id."
    assert len(chunk_ids) == len(set(chunk_ids)), (
        "Duplicate chunk IDs were generated."
    )

    required_metadata = {
        "brand",
        "model",
        "page",
        "page_label",
        "source_file",
        "document_type",
        "document_version",
        "brochure_section",
        "chunk_id",
        "chunk_index",
        "total_chunks",
    }

    for chunk in chunks:
        missing = required_metadata - set(chunk.metadata.keys())

        assert not missing, (
            f"Missing metadata fields: {missing}"
        )

        assert chunk.page_content.strip(), (
            "An empty chunk was generated."
        )

        assert (
            chunk.metadata["total_chunks"] > 0
        ), "Invalid total_chunks value."

    print("\nValidation")
    print("=" * 80)
    print("All chunk validation checks passed.")


def print_statistics(chunks):
    section_counts = Counter(
        chunk.metadata["brochure_section"]
        for chunk in chunks
    )

    document_counts = Counter(
        (
            chunk.metadata["brand"],
            chunk.metadata["model"],
        )
        for chunk in chunks
    )

    print("\nChunking Statistics")
    print("=" * 80)

    print(f"Total Chunks: {len(chunks)}")

    print("\nChunks by Section")
    print("-" * 80)

    for section, count in section_counts.items():
        print(f"{section}: {count}")

    print("\nChunks by Vehicle")
    print("-" * 80)

    for (brand, model), count in document_counts.items():
        print(f"{brand} {model}: {count}")


def main():
    loader = PDFLoader()
    cleaner = TextCleaner()
    chunker = StructuredChunker()

    print("\nLoading documents...")
    documents = loader.load_documents()

    print(f"Documents Loaded: {len(documents)}")

    print("\nCleaning documents...")
    documents = cleaner.clean_documents(documents)

    print("\nCreating structured chunks...")
    chunks = chunker.chunk_documents(documents)

    print_statistics(chunks)
    validate_chunks(chunks)

    sample_indices = [
        0,
        len(chunks) // 2,
        len(chunks) - 1,
    ]

    print("\nSample Chunks")

    for index in sample_indices:
        print_chunk(chunks[index], index)


if __name__ == "__main__":
    main()