"""Test file for the Gemini Generator module."""
from modules.gemini_generator import GeminiGenerator
def main():
    prompt = """
You are a helpful AI assistant.

Context:
Retrieval-Augmented Generation (RAG) combines information retrieval with
large language models to generate grounded responses.

Question:
What is Retrieval-Augmented Generation (RAG)?

Answer:
"""
    gemini = GeminiGenerator()
    answer = gemini.generate_answer(prompt)

    print("\nGenerated Answer\n")
    print("=" * 70)
    print(answer)

if __name__ == "__main__":
    main()