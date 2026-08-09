import re


class TextCleaner:
    def __init__(
        self,
        normalize_unicode=True,
        normalize_whitespace=True,
        remove_invisible_characters=True,
        normalize_line_breaks=True,
    ):
        self.normalize_unicode = normalize_unicode
        self.normalize_whitespace = normalize_whitespace
        self.remove_invisible_characters = remove_invisible_characters
        self.normalize_line_breaks = normalize_line_breaks

        self.character_map = {
            "ﬁ": "fi",
            "ﬂ": "fl",
            "ﬀ": "ff",
            "ﬃ": "ffi",
            "ﬄ": "ffl",
            "’": "'",
            "‘": "'",
            "“": '"',
            "”": '"',
            "–": "-",
            "—": "-",
            "…": "...",
        }

    def clean_text(self, text):
        if not text:
            return ""

        if self.normalize_unicode:
            text = self._normalize_unicode(text)

        if self.normalize_whitespace:
            text = self._normalize_whitespace(text)

        if self.remove_invisible_characters:
            text = self._remove_invisible_characters(text)

        if self.normalize_line_breaks:
            text = self._normalize_line_breaks(text)

        return text.strip()

    def clean_documents(self, documents):
        for document in documents:
            document.page_content = self.clean_text(document.page_content)

        return documents

    def _normalize_unicode(self, text):
        for original, replacement in self.character_map.items():
            text = text.replace(original, replacement)

        return text

    def _normalize_whitespace(self, text):
        text = text.replace("\t", " ")
        text = re.sub(r"[ ]{2,}", " ", text)

        return text

    def _remove_invisible_characters(self, text):
        return re.sub(r"[\u200b-\u200f\u2060\ufeff]", "", text)

    def _normalize_line_breaks(self, text):
        text = re.sub(r"\r\n?", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text