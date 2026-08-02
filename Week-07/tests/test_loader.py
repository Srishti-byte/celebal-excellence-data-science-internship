"""
Test file for the Document Loader module.
"""
from modules.loader import DocumentLoader
def main():
    pdf_path = "data/rag_paper.pdf"
    loader = DocumentLoader(pdf_path)
    documents = loader.load_document()

    print("\nFirst Page Metadata:")
    print(documents[0].metadata)
    print("\nFirst 500 Characters:\n")
    print(documents[0].page_content[:500])

if __name__ == "__main__":
    main()