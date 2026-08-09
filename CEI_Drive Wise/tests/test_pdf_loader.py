from src.pdf_loader import PDFLoader


def print_metadata(metadata):
    print("\nMetadata")
    print("-" * 50)

    for key, value in metadata.items():
        print(f"{key}: {value}")


def print_document_preview(document, title):
    print(f"\n{title}")
    print("=" * 60)

    print_metadata(document.metadata)

    print("\nExtracted Text Preview")
    print("-" * 50)
    print(document.page_content[:500])

    print("\n" + "=" * 60)


def main():
    loader = PDFLoader()
    documents = loader.load_documents()

    print("\nPDF Loader Test")
    print("=" * 60)
    print(f"Total Documents Loaded: {len(documents)}")

    if not documents:
        print("\nNo documents were loaded.")
        return

    print_document_preview(documents[0], "First Document")

    if len(documents) > 1:
        print_document_preview(documents[-1], "Last Document")


if __name__ == "__main__":
    main()