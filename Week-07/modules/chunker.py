"""
This module is responsible for splitting documents into smaller chunks.
"""
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP

class TextChunker:
    """ Split LangChain Document objects into smaller chunks """
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = CHUNK_SIZE,
            chunk_overlap = CHUNK_OVERLAP,
            length_function = len,
            is_separator_regex = False
        )

    def split_documents(self, documents : list[Document])-> list[Document]:
        """
        Split documents into smaller chunks
        Args: documents: List of LangChain Document objects.
        Returns: List of chunked Document objects.
        """
        chunks = self.text_splitter.split_documents(documents)

        print("=" * 50)
        print("Document Chunking Completed")
        print(f"Total Chunks: {len(chunks)}")
        print("=" * 50)

        return chunks