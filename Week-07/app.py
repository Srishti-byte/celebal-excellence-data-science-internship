import os
import tempfile

import streamlit as st

from modules.loader import DocumentLoader
from modules.chunker import TextChunker
from modules.metadata_manager import MetadataManager
from modules.embeddings import EmbeddingGenerator
from modules.vector_store import VectorStoreManager
from modules.retriever import Retriever
from modules.reranker import Reranker
from modules.prompt_builder import PromptBuilder
from modules.gemini_generator import GeminiGenerator
from modules.rag_pipeline import RAGPipeline


st.set_page_config(
    page_title="Document Question Answering System",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Document Question Answering System")

st.markdown(
    "Upload a PDF document and ask questions using Retrieval-Augmented Generation (RAG)."
)

st.divider()

uploaded_file = st.file_uploader(
    "Upload a PDF Document",
    type=["pdf"]
)

if uploaded_file is not None:

    st.success(f"Uploaded: {uploaded_file.name}")

    if st.button("⚙️ Process Document"):

        with st.spinner("Processing document..."):

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf"
            ) as temp_file:

                temp_file.write(uploaded_file.getbuffer())
                temp_path = temp_file.name

            loader = DocumentLoader(temp_path)
            documents = loader.load_document()

            chunker = TextChunker()
            chunks = chunker.split_documents(documents)

            metadata_manager = MetadataManager()
            processed_chunks = metadata_manager.process_metadata(chunks)

            embedding_generator = EmbeddingGenerator()
            embedding_model = embedding_generator.get_embedding_model()

            vector_store_manager = VectorStoreManager(
                embedding_model
            )

            vector_store = vector_store_manager.create_vector_store(
                processed_chunks
            )

            vector_store_manager.save_vector_store(
                vector_store
            )

            os.remove(temp_path)

            retriever = Retriever(
                vector_store,
                embedding_model
            )

            reranker = Reranker()

            prompt_builder = PromptBuilder()

            gemini_generator = GeminiGenerator()

            pipeline = RAGPipeline(
                retriever,
                reranker,
                prompt_builder,
                gemini_generator
            )

            st.session_state.pipeline = pipeline
            st.session_state.document_ready = True

        st.success("Document processed successfully!")

if st.session_state.get("document_ready", False):

    st.divider()

    question = st.text_input(
        "Ask a Question"
    )

    if st.button("💬 Get Answer"):

        if question.strip():

            with st.spinner("Generating answer..."):

                result = st.session_state.pipeline.ask(
                    question
                )

            st.subheader("🤖 Answer")

            st.write(result["answer"])

            st.subheader("📚 Retrieved Sources")

            for index, source in enumerate(
                result["sources"],
                start=1
            ):

                document = source["document"]

                with st.expander(
                    f"Source {index}"
                ):

                    st.write(
                        f"**Relevance Score:** {source['relevance_score']:.4f}"
                    )

                    st.write(
                        f"**Page:** {document.metadata['page']}"
                    )

                    st.write(
                        f"**Chunk ID:** {document.metadata['chunk_id']}"
                    )

                    st.write(
                        f"**File:** {document.metadata['source']}"
                    )

                    st.write(
                        document.page_content
                    )