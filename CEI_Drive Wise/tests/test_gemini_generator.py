from src.gemini_generator import GeminiGenerator


def main():
    print("\nInitializing Gemini Generator...")

    generator = GeminiGenerator()

    print(
        f"Gemini Model: "
        f"{generator.get_model_name()}"
    )

    prompt = """
Answer the following question in one or two sentences.

What is the purpose of a car's ABS?
"""

    print("\nTest Prompt")
    print("=" * 50)
    print(prompt.strip())

    print("\nGenerating Answer...")

    answer = generator.generate_answer(
        prompt
    )

    print("\nGenerated Answer")
    print("=" * 50)
    print(answer)

    print("\nValidation")
    print("=" * 50)

    assert isinstance(answer, str)
    assert answer.strip()

    print(
        "Gemini generator validation "
        "checks passed."
    )


if __name__ == "__main__":
    main()