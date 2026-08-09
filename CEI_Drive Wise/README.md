# 🚗 Drive Wise

A metadata-aware Retrieval-Augmented Generation (RAG) application for automotive brochure question answering. Drive Wise retrieves relevant brochure information and generates grounded answers using Google Gemini through an interactive Streamlit interface.

---

## Objective

Develop an end-to-end RAG pipeline capable of:

- Processing automotive brochure PDFs
- Cleaning and chunking brochure text
- Extracting and preserving document metadata
- Generating semantic embeddings
- Performing metadata-filtered vector retrieval
- Re-ranking retrieved document chunks
- Generating grounded answers using an LLM
- Providing source attribution
- Evaluating answer quality
- Logging pipeline activity
- Providing an interactive question-answering interface

---

## Features

### 📄 Document Processing

- PDF document loading
- Text cleaning and normalization
- Metadata extraction and standardization
- Structured brochure-section chunking

### 🔍 Retrieval Pipeline

- `BAAI/bge-small-en-v1.5` embeddings
- ChromaDB vector database
- Brand and model metadata filtering
- Top-5 semantic retrieval
- Cohere reranking
- Top-3 context selection

### 🤖 Response Generation

- Grounded prompt construction
- Controlled context generation
- Google Gemini answer generation
- Context-only answering to reduce unsupported information

### 📊 Evaluation

- LLM-based answer correctness using OpenRouter
- Cosine-similarity based faithfulness
- Cosine-similarity based context relevance

### 💻 User Interface

- Interactive Streamlit application
- Brand and model selection
- Natural-language question answering
- Source visualization with metadata
- Evaluation metrics
- Response-time information
- Monitoring logs

---

## System Architecture

    Automotive Brochure
            │
            ▼
      PDF Processing
            │
            ▼
    Text Cleaning & Metadata
            │
            ▼
     Structured Chunking
            │
            ▼
    Embedding Generation
            │
            ▼
         ChromaDB
            │
            ▼
    Metadata-Filtered Retriever
            │
            ▼
      Cohere Reranker
            │
            ▼
      Prompt Generator
            │
            ▼
      Gemini Generator
            │
            ▼
    Grounded Answer + Sources
            │
            ▼
         Evaluator
       ┌────┼────┐
       ▼    ▼    ▼
 Correctness  Faithfulness  Context Relevance
            │
            ▼
    Logging & Monitoring

---

## Project Structure

    DRIVE WISE/
    │
    ├── chroma_db/
    ├── dataset/
    │   ├── Hyundai/
    │   ├── Kia/
    │   ├── Mahindra/
    │   ├── Tata/
    │   └── Toyota/
    │
    ├── logs/
    │   └── drive_wise.log
    │
    ├── src/
    │   ├── chunker.py
    │   ├── config.py
    │   ├── embeddings.py
    │   ├── evaluator.py
    │   ├── gemini_generator.py
    │   ├── logger.py
    │   ├── metadata_manager.py
    │   ├── pdf_loader.py
    │   ├── prompt_generator.py
    │   ├── rag_pipeline.py
    │   ├── reranker.py
    │   ├── retriever.py
    │   ├── text_cleaner.py
    │   └── vector_store.py
    │
    ├── tests/
    │   ├── inspect_pdf.py
    │   ├── test_chunker.py
    │   ├── test_embedding.py
    │   ├── test_evaluator.py
    │   ├── test_gemini_generator.py
    │   ├── test_logger.py
    │   ├── test_metadata_manager.py
    │   ├── test_pdf_loader.py
    │   ├── test_prompt_generator.py
    │   ├── test_rag_pipeline.py
    │   ├── test_reranker.py
    │   ├── test_retriever.py
    │   ├── test_text_cleaner.py
    │   └── test_vector_store.py
    │
    ├── .env
    ├── .env.example
    ├── .gitignore
    ├── app.py
    ├── build_vector_store.py
    ├── README.md
    ├── requirements.txt
    └── run_pipeline.py

---

## System Configuration

| Component | Configuration |
|---|---|
| Embedding Model | `BAAI/bge-small-en-v1.5` |
| Embedding Dimension | 384 |
| Vector Store | ChromaDB |
| Metadata Filter | Brand + Model |
| Retriever | Top-5 |
| Reranker | Cohere |
| Reranker Top-N | 3 |
| Generator | Google Gemini |
| Answer Correctness | OpenRouter LLM Judge |
| Faithfulness | Cosine Similarity |
| Context Relevance | Cosine Similarity |
| Interface | Streamlit |

---

## Technologies Used

- Python
- Streamlit
- LangChain
- ChromaDB
- Sentence Transformers
- Hugging Face
- PyPDF
- Cohere API
- Google Gemini API
- OpenRouter API
- NumPy
- Python Dotenv

---

## Installation

Clone the repository:

    git clone <repository-url>
    cd DRIVE-WISE

Create a virtual environment:

    python -m venv venv

Activate the virtual environment.

**Windows**

    venv\Scripts\activate

**Linux/macOS**

    source venv/bin/activate

Install dependencies:

    pip install -r requirements.txt

---

## Environment Variables

Create a `.env` file in the project root:

    COHERE_API_KEY=your_cohere_api_key
    GEMINI_API_KEY=your_gemini_api_key
    OPENROUTER_API_KEY=your_openrouter_api_key

A `.env.example` file is provided as a template.

Do not commit the actual `.env` file.

---

## Running the Application

Build the vector store:

    python build_vector_store.py

Launch the Streamlit application:

    streamlit run app.py

---

## Usage

1. Select a car **brand**.
2. Select the **model**.
3. Enter a natural-language question.
4. The system retrieves the top 5 relevant chunks using metadata filtering and semantic search.
5. Cohere reranks the retrieved chunks and selects the top 3.
6. Gemini generates a grounded answer using the selected context.
7. The UI displays the answer, sources, evaluation metrics, and response time.
8. Pipeline activity is recorded in the monitoring logs.

### Example Query

    Does the Mahindra Thar ROXX have ADAS,
    and is cruise control available on all variants?

---

## Evaluation

Drive Wise evaluates each generated response using three metrics.

### Answer Correctness

An OpenRouter LLM judge evaluates the generated answer against the user query and retrieved brochure context.

Score range:

    0.0 → Incorrect
    0.5 → Partially correct
    1.0 → Fully correct and supported

### Faithfulness

Measures how strongly answer sentences are supported by the retrieved brochure chunks using cosine similarity.

### Context Relevance

Measures the semantic similarity between the user query and retrieved brochure chunks using cosine similarity.

---

## Source Attribution

Each response includes the brochure sources used by the pipeline, including:

- Brand
- Model
- Section
- Source file
- Page
- Chunk ID

This allows users to verify the information behind the generated answer.

---

## Logging & Monitoring

Logs are stored in:

    logs/drive_wise.log

The logger records:

- Queries
- Response time
- Retrieval results
- Generation status
- Evaluation metrics
- Failed queries

---

## Testing

Run the complete pipeline:

    python run_pipeline.py

Run the RAG integration test:

    python -m tests.test_rag_pipeline

Individual component tests are available in the `tests/` directory.

---

## Future Enhancements

- Multi-document conversational support
- Conversational memory
- Hybrid retrieval (Dense + Keyword)
- Advanced claim-level evaluation
- Response-time optimization
- Docker support
- Cloud deployment