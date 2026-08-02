"""
This module is responsible for loading PDF documents using LangChain.
"""

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document

class DocumentLoader:
    """
    Loads a PDF document and returns LangChain Document objects.
    """
    def __init__(self, file_path: str):
        """
        Initialize the document loader.

        Args:
            file_path (str): Path to the PDF document.
        """
        self.file_path = file_path

    def load_document(self) -> list[Document]:
        """
        Load the PDF document.
        Returns: 
        list[Document]: List of LangChain Document objects.
        """
        loader = PyMuPDFLoader(self.file_path)
        documents = loader.load()

        print("=" * 50)
        print("Document Loaded Successfully")
        print(f"Total Pages: {len(documents)}")
        print("=" * 50)

        return documents