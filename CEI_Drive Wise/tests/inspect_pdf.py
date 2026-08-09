from src.pdf_loader import PDFLoader


def inspect_document(documents, index):
    if index < 0 or index >= len(documents):
        print(f"Invalid index. Enter a value between 0 and {len(documents) - 1}.")
        return

    document = documents[index]

    print("\n" + "=" * 80)
    print(f"Document Index : {index}")
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
    documents = loader.load_documents()

    print(f"\nTotal Documents Loaded: {len(documents)}")

    while True:
        choice = input(
            f"\nEnter a document index (0 - {len(documents)-1}) or 'q' to quit: "
        )

        if choice.lower() == "q":
            break

        try:
            inspect_document(documents, int(choice))
        except ValueError:
            print("Please enter a valid number.")


if __name__ == "__main__":
    main()