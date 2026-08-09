from src.pdf_loader import PDFLoader
from src.text_cleaner import TextCleaner
from src.metadata_manager import MetadataManager


def print_metadata(title, metadata):
    print(f"\n{title}")
    print("=" * 80)

    for key, value in metadata.items():
        print(f"{key}: {value}")

    print("=" * 80)


def main():
    loader = PDFLoader()
    cleaner = TextCleaner()
    metadata_manager = MetadataManager()

    documents = loader.load_documents()
    documents = cleaner.clean_documents(documents)

    document = documents[5]

    print_metadata(
        "Original Metadata",
        document.metadata,
    )

    processed_metadata = metadata_manager.prepare_metadata(
        metadata=document.metadata,
        brochure_section="Safety",
        chunk_index=1,
        total_chunks=8,
    )

    print_metadata(
        "Processed Metadata",
        processed_metadata,
    )


if __name__ == "__main__":
    main()