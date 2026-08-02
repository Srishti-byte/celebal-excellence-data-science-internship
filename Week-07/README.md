# 📄 RAG-Based Document Question Answering System

A Retrieval-Augmented Generation (RAG) application that enables users to upload PDF documents, retrieve semantically relevant information, and generate grounded, context-aware answers using Google's Gemini model through an interactive Streamlit interface.

---

## Objective

Develop an end-to-end Retrieval-Augmented Generation (RAG) pipeline capable of:

- Ingesting PDF documents
- Processing and chunking unstructured text
- Generating semantic vector embeddings
- Performing efficient similarity-based retrieval
- Reranking retrieved document chunks
- Producing grounded responses using a Large Language Model
- Providing an interactive document question answering interface

---

## Features

### 📄 Document Processing
- PDF document loading
- Recursive text chunking
- Metadata extraction and standardization

### 🔍 Retrieval Pipeline
- Dense embeddings using **BAAI/bge-small-en-v1.5**
- FAISS vector database for semantic search
- Top-k document retrieval
- Cohere Reranker for relevance optimization

### 🤖 Response Generation
- Context-aware prompt construction
- Grounded answer generation using Google Gemini

### 💻 User Interface
- Interactive Streamlit application
- PDF upload support
- Natural language question answering
- Retrieved source visualization with metadata

---

## System Architecture

```text
                   PDF Document
                        │
                        ▼
               Document Loader
                        │
                        ▼
                Recursive Chunking
                        │
                        ▼
              Metadata Processing
                        │
                        ▼
              Embedding Generation
                        │
                        ▼
               FAISS Vector Store
                        │
                        ▼
                   Retriever
                        │
                        ▼
               Cohere Reranker
                        │
                        ▼
                Prompt Builder
                        │
                        ▼
              Gemini Generator
                        │
                        ▼
      Grounded Answer + Retrieved Sources
```

---

## Project Structure

```text
DocumentQA-RAG/
│
├── data/
├── modules/
│   ├── loader.py
│   ├── chunker.py
│   ├── metadata_manager.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── retriever.py
│   ├── reranker.py
│   ├── prompt_builder.py
│   ├── gemini_generator.py
│   └── rag_pipeline.py
│
├── tests/
├── vector_store/
│
├── app.py
├── run_pipeline.py
├── config.py
├── requirements.txt
└── README.md
```

---

## System Configuration

| Component | Configuration |
|-----------|---------------|
| Chunk Size | 600 |
| Chunk Overlap | 100 |
| Embedding Model | BAAI/bge-small-en-v1.5 |
| Embedding Dimension | 384 |
| Vector Store | FAISS |
| Retriever | Top-5 Semantic Retrieval |
| Reranker | Cohere rerank-v3.5 |
| Language Model | Google Gemini 3.6 Flash |

---

## Technologies Used

- Python
- Streamlit
- LangChain
- Sentence Transformers
- FAISS
- Hugging Face
- Cohere API
- Google Gemini API

---

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd DocumentQA-RAG
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the virtual environment:

**Windows**

```bash
venv\Scripts\activate
```

**Linux/macOS**

```bash
source venv/bin/activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file in the project root.

```env
COHERE_API_KEY=your_cohere_api_key
GEMINI_API_KEY=your_gemini_api_key
```

---

## Running the Application

Launch the Streamlit application:

```bash
streamlit run app.py
```

---

## Usage

1. Upload a PDF document.
2. Click **Process Document** to build the vector store.
3. Enter a question related to the uploaded document.
4. View the generated answer.
5. Inspect the retrieved source chunks along with their relevance scores and metadata.

---

## Testing

Run the complete pipeline:

```bash
python run_pipeline.py
```

Run the integration test:

```bash
python -m tests.test_rag_pipeline
```

Individual test modules for each major component are available in the `tests/` directory.

---

## Future Enhancements

- Multi-document support
- Conversational memory
- Hybrid retrieval (Dense + Keyword Search)
- Persistent vector databases (ChromaDB/Pinecone)
- Docker support
- Cloud deployment