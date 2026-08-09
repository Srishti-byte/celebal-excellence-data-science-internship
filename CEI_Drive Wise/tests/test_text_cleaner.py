from src.pdf_loader import PDFLoader
from src.text_cleaner import TextCleaner


def print_document(document, title):
    print(f"\n{title}")
    print("=" * 80)

    print("\nMetadata")
    print("-" * 80)

    for key, value in document.metadata.items():
        print(f"{key}: {value}")

    print("\nPage Content")
    print("-" * 80)
    print(document.page_content)

    print("\n" + "=" * 80)


def main():
    loader = PDFLoader()
    cleaner = TextCleaner()

    documents = loader.load_documents()
    cleaned_documents = cleaner.clean_documents(documents)

    print(f"\nTotal Documents Loaded: {len(cleaned_documents)}")

    indices = [5, 70, 180]

    for index in indices:
        print_document(
            cleaned_documents[index],
            f"Document {index}"
        )


if __name__ == "__main__":
    main()