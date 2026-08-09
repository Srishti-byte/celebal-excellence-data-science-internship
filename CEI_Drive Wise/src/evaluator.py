from typing import List, Optional
import json
import re

import numpy as np
import requests
from langchain_core.documents import Document

from src.config import (
    OPENROUTER_API_KEY,
    OPENROUTER_MODEL,
)
from src.embeddings import EmbeddingManager


class Evaluator:
    def __init__(
        self,
        embedding_manager: Optional[EmbeddingManager] = None,
    ):
        self.embedding_manager = (
            embedding_manager or EmbeddingManager()
        )

        if not OPENROUTER_API_KEY:
            raise ValueError(
                "OPENROUTER_API_KEY is not configured."
            )

    def evaluate(
        self,
        query: str,
        answer: str,
        documents: List[Document],
    ) -> dict:
        self._validate_inputs(query, answer, documents)

        return {
            "answer_correctness": self.evaluate_answer_correctness(
                query, answer, documents
            ),
            "faithfulness": self.evaluate_faithfulness(
                answer, documents
            ),
            "context_relevance": self.evaluate_context_relevance(
                query, documents
            ),
        }

    def evaluate_answer_correctness(
        self,
        query: str,
        answer: str,
        documents: List[Document],
    ) -> float:
        context = self._build_context(documents)

        prompt = f"""
You are evaluating a car brochure RAG system.

Determine whether the generated answer correctly answers the question
and whether its factual claims are supported by the provided brochure
context.

Question:
{query}

Generated Answer:
{answer}

Brochure Context:
{context}

Instructions:
1. Identify the factual claims made by the generated answer.
2. For each claim, determine whether it is supported by the brochure
   context.
3. Do not use outside knowledge.
4. Do not treat absence of information as proof that a fact is false.
5. If the answer correctly states that the provided context does not
   confirm or contain information, consider that statement supported
   when the context genuinely lacks that information.
6. Ignore wording differences when the meaning is the same.

Return ONLY valid JSON in this format:

{{
  "claims": [
    {{
      "claim": "claim text",
      "supported": true
    }}
  ]
}}

Do not include markdown or any additional text.
"""

        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": (
                    f"Bearer {OPENROUTER_API_KEY}"
                ),
                "Content-Type": "application/json",
            },
            json={
                "model": OPENROUTER_MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt.strip(),
                    }
                ],
                "temperature": 0,
            },
            timeout=60,
        )

        response.raise_for_status()

        content = (
            response.json()["choices"][0]["message"]["content"]
            .strip()
        )

        evaluation = self._parse_json_response(content)
        claims = evaluation.get("claims", [])

        if not claims:
            return 0.0

        supported_claims = sum(
            1
            for claim in claims
            if claim.get("supported") is True
        )

        return round(
            supported_claims / len(claims),
            4,
        )

    def evaluate_faithfulness(
        self,
        answer: str,
        documents: List[Document],
    ) -> float:
        if not answer or not answer.strip():
            raise ValueError("Answer cannot be empty.")

        if not documents:
            raise ValueError("Documents cannot be empty.")

        sentences = self._split_sentences(answer)

        if not sentences:
            return 0.0

        context_vectors = [
            self._embed_text(document.page_content)
            for document in documents
        ]

        scores = []

        for sentence in sentences:
            sentence_vector = self._embed_text(sentence)

            scores.append(
                max(
                    self._cosine_similarity(
                        sentence_vector,
                        context_vector,
                    )
                    for context_vector in context_vectors
                )
            )

        return float(np.mean(scores))

    def evaluate_context_relevance(
        self,
        query: str,
        documents: List[Document],
    ) -> float:
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        if not documents:
            raise ValueError("Documents cannot be empty.")

        query_vector = self._embed_text(query)

        scores = [
            self._cosine_similarity(
                query_vector,
                self._embed_text(document.page_content),
            )
            for document in documents
        ]

        return float(np.mean(scores))

    def _build_context(
        self,
        documents: List[Document],
    ) -> str:
        return "\n\n".join(
            f"Context {index}:\n{document.page_content.strip()}"
            for index, document in enumerate(
                documents,
                start=1,
            )
        )

    def _parse_json_response(
        self,
        content: str,
    ) -> dict:
        content = content.strip()

        # Handle responses enclosed in a markdown JSON block.
        content = re.sub(
            r"^```json\s*|\s*```$",
            "",
            content,
            flags=re.IGNORECASE,
        ).strip()

        try:
            result = json.loads(content)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Invalid evaluator response: {content}"
            ) from error

        if not isinstance(result, dict):
            raise ValueError(
                "Evaluator response must be a JSON object."
            )

        return result

    def _embed_text(self, text: str) -> np.ndarray:
        return np.asarray(
            self.embedding_manager.embed_query(text),
            dtype=float,
        )

    def _cosine_similarity(
        self,
        vector_a: np.ndarray,
        vector_b: np.ndarray,
    ) -> float:
        denominator = (
            np.linalg.norm(vector_a)
            * np.linalg.norm(vector_b)
        )

        if denominator == 0:
            return 0.0

        similarity = np.dot(
            vector_a,
            vector_b,
        ) / denominator

        return float(np.clip(similarity, 0.0, 1.0))

    def _split_sentences(
        self,
        text: str,
    ) -> List[str]:
        normalized = (
            text.replace("!", ".")
            .replace("?", ".")
        )

        return [
            sentence.strip()
            for sentence in normalized.split(".")
            if sentence.strip()
        ]

    def _validate_inputs(
        self,
        query: str,
        answer: str,
        documents: List[Document],
    ) -> None:
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        if not answer or not answer.strip():
            raise ValueError("Answer cannot be empty.")

        if not documents:
            raise ValueError("Documents cannot be empty.")