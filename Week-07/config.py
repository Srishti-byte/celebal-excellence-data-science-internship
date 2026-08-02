"""Configuration settings for the RAG Document Question Answering System."""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Chunking Configuration
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100

# Retrieval Configuration
TOP_K = 5

# Embedding Model
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# Vector Store
VECTOR_STORE_PATH = "vector_store"

# API Keys
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Reranker
RERANK_TOP_K = 3
RERANK_MODEL = "rerank-v3.5"

# Gemini Configuration
GEMINI_MODEL = "gemini-3.6-flash"