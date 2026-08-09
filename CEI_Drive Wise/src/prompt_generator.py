from typing import List

from langchain_core.documents import Document


class PromptGenerator:
    """
    Creates a grounded prompt using the user's query
    and the documents returned by the reranker.
    """

    def __init__(self):
        self.system_instruction = (
            "You are a helpful assistant for a car "
            "brochure question-answering system.\n\n"
            "Answer the user's question using only "
            "the provided brochure context.\n\n"
            "Do not use outside knowledge or make up "
            "information.\n\n"
            "If the answer cannot be found in the "
            "provided context, clearly say that the "
            "information is not available in the "
            "provided brochures.\n\n"
            "Keep the answer clear, concise, and "
            "directly relevant to the question."
        )

    def generate_prompt(
        self,
        query: str,
        documents: List[Document],
    ) -> str:
        """
        Generate the final grounded prompt.
        """

        if not query or not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        if not documents:
            raise ValueError(
                "Documents cannot be empty."
            )

        context = self._build_context(
            documents
        )

        prompt = (
            f"{self.system_instruction}\n\n"
            f"================ CONTEXT ================\n\n"
            f"{context}\n\n"
            f"================ QUESTION ================\n\n"
            f"{query.strip()}\n\n"
            f"================ ANSWER ==================\n\n"
            f"Answer using only the context above."
        )

        return prompt

    def _build_context(
        self,
        documents: List[Document],
    ) -> str:
        context_parts = []

        for index, document in enumerate(
            documents,
            start=1,
        ):
            metadata = document.metadata

            brand = metadata.get(
                "brand",
                "Unknown",
            )

            model = metadata.get(
                "model",
                "Unknown",
            )

            section = metadata.get(
                "brochure_section",
                "Unknown",
            )

            source_file = metadata.get(
                "source_file",
                "Unknown",
            )

            page = metadata.get(
                "page_label",
                metadata.get(
                    "page",
                    "Unknown",
                ),
            )

            context_parts.append(
                f"Context {index}\n"
                f"Brand: {brand}\n"
                f"Model: {model}\n"
                f"Section: {section}\n"
                f"Source: {source_file}\n"
                f"Page: {page}\n\n"
                f"{document.page_content.strip()}"
            )

        return "\n\n------------------------------\n\n".join(
            context_parts
        )