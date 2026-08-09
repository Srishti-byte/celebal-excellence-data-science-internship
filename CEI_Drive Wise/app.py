import streamlit as st

from src.config import LOGS_PATH
from src.vector_store import VectorStore
from run_pipeline import initialize_pipeline, run_query


st.set_page_config(
    page_title="Drive Wise",
    page_icon="🚗",
    layout="wide",
)


@st.cache_resource
def load_pipeline():
    return initialize_pipeline()


@st.cache_data(ttl=300)
def load_car_catalog():
    vector_store = VectorStore()

    result = vector_store.collection.get(
        include=["metadatas"]
    )

    metadatas = result.get("metadatas", [])

    catalog = {}

    for metadata in metadatas:
        if not metadata:
            continue

        brand = metadata.get("brand")
        model = metadata.get("model")

        if not brand or not model:
            continue

        catalog.setdefault(
            brand,
            set(),
        ).add(model)

    return {
        brand: sorted(models)
        for brand, models in sorted(
            catalog.items()
        )
    }


def initialize_session():
    if "history" not in st.session_state:
        st.session_state.history = []

    if "last_result" not in st.session_state:
        st.session_state.last_result = None


def display_sources(sources):
    st.subheader("Sources")

    for index, source in enumerate(
        sources,
        start=1,
    ):
        with st.expander(
            f"Source {index}: "
            f"{source['source_file']}"
        ):
            col1, col2 = st.columns(2)

            with col1:
                st.write(
                    f"**Brand:** {source['brand']}"
                )
                st.write(
                    f"**Model:** {source['model']}"
                )
                st.write(
                    f"**Section:** {source['section']}"
                )

            with col2:
                st.write(
                    f"**Page:** {source['page']}"
                )
                st.write(
                    f"**Chunk ID:** {source['chunk_id']}"
                )


def display_evaluation(metrics):
    st.subheader("Evaluation")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Answer Correctness",
            f"{metrics['answer_correctness']:.2f}",
        )

    with col2:
        st.metric(
            "Faithfulness",
            f"{metrics['faithfulness']:.2f}",
        )

    with col3:
        st.metric(
            "Context Relevance",
            f"{metrics['context_relevance']:.2f}",
        )

    st.caption(
        "Correctness is evaluated against the retrieved "
        "brochure context. Faithfulness and context "
        "relevance use embedding-based similarity."
    )


def display_monitoring():
    st.subheader("Monitoring")

    log_file = LOGS_PATH / "drive_wise.log"

    if not log_file.exists():
        st.info("No log entries available yet.")
        return

    content = log_file.read_text(
        encoding="utf-8"
    ).strip()

    if not content:
        st.info("No log entries available yet.")
        return

    lines = content.splitlines()

    st.write(
        f"Total log entries: {len(lines)}"
    )

    st.code(
        "\n".join(lines[-50:]),
        language="text",
    )


def main():
    initialize_session()

    st.title("🚗 Drive Wise")
    st.subheader(
        "Metadata-Aware Automotive RAG Assistant"
    )

    st.write(
        "Ask questions about a selected car using "
        "information grounded in its official brochure."
    )

    try:
        pipeline = load_pipeline()
        catalog = load_car_catalog()

    except Exception as error:
        st.error(
            f"Unable to initialize Drive Wise: {error}"
        )
        st.stop()

    if not catalog:
        st.error(
            "No vehicle data found in the production "
            "vector store."
        )
        st.info(
            "Run build_vector_store.py before starting "
            "the application."
        )
        st.stop()

    (
        retriever,
        reranker,
        prompt_generator,
        gemini_generator,
        evaluator,
        logger,
    ) = pipeline

    with st.sidebar:
        st.header("Vehicle Selection")

        brands = list(catalog.keys())

        brand = st.selectbox(
            "Brand",
            brands,
        )

        models = catalog[brand]

        model = st.selectbox(
            "Model",
            models,
        )

        st.divider()

        st.subheader("System")

        st.write(
            f"**Embedding:** "
            f"{retriever.embedding_manager.get_model_name()}"
        )

        st.write(
            f"**Retrieval Top-K:** "
            f"{retriever.get_top_k()}"
        )

        st.write(
            f"**Reranker:** "
            f"{reranker.get_model_name()}"
        )

        st.write(
            f"**Reranker Top-N:** "
            f"{reranker.get_top_n()}"
        )

        st.divider()

        if st.button(
            "Clear Conversation",
            use_container_width=True,
        ):
            st.session_state.history = []
            st.session_state.last_result = None
            st.rerun()

    st.markdown(
        f"### Selected vehicle: "
        f"**{brand} {model}**"
    )

    with st.form(
        "query_form",
        clear_on_submit=True,
    ):
        query = st.text_area(
            "Your question",
            placeholder=(
                "e.g. What safety features "
                "does this car have?"
            ),
            height=100,
        )

        submitted = st.form_submit_button(
            "Ask Drive Wise",
            use_container_width=True,
        )

    if submitted:
        if not query.strip():
            st.warning(
                "Please enter a question."
            )
        else:
            with st.status(
                "Running Drive Wise...",
                expanded=True,
            ) as status:

                st.write(
                    "Retrieving relevant brochure sections..."
                )

                result = run_query(
                    query=query.strip(),
                    brand=brand,
                    model=model,
                    retriever=retriever,
                    reranker=reranker,
                    prompt_generator=prompt_generator,
                    gemini_generator=gemini_generator,
                    evaluator=evaluator,
                    logger=logger,
                )

                if result is None:
                    status.update(
                        label="Query failed",
                        state="error",
                    )
                    st.error(
                        "The query could not be completed. "
                        "Check the monitoring logs for details."
                    )
                    st.stop()

                status.update(
                    label="Query completed",
                    state="complete",
                    expanded=False,
                )

            st.session_state.last_result = result

            st.session_state.history.append(
                {
                    "brand": brand,
                    "model": model,
                    "query": query.strip(),
                    "answer": result["answer"],
                    "sources": result["sources"],
                    "evaluation": result["evaluation"],
                    "response_time": result["response_time"],
                }
            )

    if st.session_state.history:
        st.divider()

        for item in reversed(
            st.session_state.history
        ):
            st.markdown(
                f"### {item['brand']} "
                f"{item['model']}"
            )

            st.markdown(
                f"**Question:** {item['query']}"
            )

            st.markdown("### Answer")
            st.write(item["answer"])

            display_sources(
                item["sources"]
            )

            display_evaluation(
                item["evaluation"]
            )

            st.caption(
                f"Response time: "
                f"{item['response_time']:.2f} seconds"
            )

            st.divider()

    with st.expander(
        "Monitoring Logs"
    ):
        display_monitoring()


if __name__ == "__main__":
    main()