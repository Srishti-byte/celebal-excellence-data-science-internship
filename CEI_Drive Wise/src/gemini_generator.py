from src.config import GEMINI_API_KEY, GEMINI_MODEL


class GeminiGenerator:
    def __init__(self):
        self.client = None
        self.model = GEMINI_MODEL

    def _initialize_client(self):
        if self.client is None:
            from google import genai

            self.client = genai.Client(
                api_key=GEMINI_API_KEY
            )

            print("=" * 50)
            print("Gemini Model Initialized")
            print(f"Model : {self.model}")
            print("=" * 50)

    def generate_answer(self, prompt: str) -> str:
        if not prompt or not prompt.strip():
            raise ValueError(
                "Prompt cannot be empty."
            )

        self._initialize_client()

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
            )

            if not response.text:
                raise ValueError(
                    "Gemini returned an empty response."
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

    def get_model_name(self) -> str:
        return self.model