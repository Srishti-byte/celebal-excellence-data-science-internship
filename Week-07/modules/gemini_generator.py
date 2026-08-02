"""This module generates grounded answers using Google's Gemini model."""

from config import GEMINI_API_KEY, GEMINI_MODEL

class GeminiGenerator:
    """Generates grounded answers using the Gemini model."""
    def __init__(self):
        self.client = None
        self.model = GEMINI_MODEL

    def _initialize_client(self):
        """Initialize the Gemini client only when required."""

        if self.client is None:
            from google import genai

            self.client = genai.Client(
                api_key=GEMINI_API_KEY
            )

            print("=" * 50)
            print("Gemini Model Initialized")
            print(f"Model : {GEMINI_MODEL}")
            print("=" * 50)

    def generate_answer(self, prompt: str) -> str:
        """
        Generate an answer using Gemini.
        Args:
            prompt: Final prompt created by the Prompt Builder.
        Returns:
            Generated answer as a string.
        """

        self._initialize_client()
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt
            )

            print("=" * 50)
            print("Answer Generated Successfully")
            print("=" * 50)

            return response.text.strip()

        except Exception as error:

            print("=" * 50)
            print("Gemini Generation Failed")
            print(error)
            print("=" * 50)

            raise