"""Builds the final prompt for Gemini usingthe reranked document chunks."""

class PromptBuilder:
    """Builds a grounded prompt using the reranked documents."""

    def build_prompt(self, query, reranked_documents):
        prompt = """
You are an AI assistant for document question answering.

Instructions:
- Answer ONLY using the retrieved context below.
- Do not add information that is not supported by the context.
- If the context is insufficient, respond:
  "I could not find the answer in the provided document."
- If multiple retrieved chunks contribute to the answer, combine them naturally.

==================================================

Retrieved Context

"""
        for result in reranked_documents:
            document = result["document"]
            prompt += (
                f"Source : {document.metadata['source']}\n"
                f"Page   : {document.metadata['page']}\n"
                f"Score  : {result['relevance_score']:.4f}\n\n"
                f"Content:\n"
                f"{document.page_content}\n\n"
                + "-" * 60
                + "\n\n"
            )

        prompt += (
            "User Question:\n"
            f"{query}\n\n"
            "Answer:"
        )

        print("=" * 50)
        print("Prompt Built Successfully")
        print("=" * 50)

        return prompt