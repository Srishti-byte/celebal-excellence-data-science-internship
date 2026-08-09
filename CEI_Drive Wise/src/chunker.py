from collections import defaultdict

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHUNK_SIZE, CHUNK_OVERLAP, SECTION_KEYWORDS
from src.metadata_manager import MetadataManager


class StructuredChunker:
    def __init__(
        self,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        section_keywords=SECTION_KEYWORDS,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.section_keywords = section_keywords

        self.metadata_manager = MetadataManager()

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=[
                "\n\n",
                "\n",
                ". ",
                "; ",
                ", ",
                " ",
            ],
        )

    def chunk_documents(self, documents):
        grouped_documents = self._group_documents(documents)
        all_chunks = []

        for document_group in grouped_documents.values():
            sections = self._build_sections(document_group)
            document_chunks = []

            for section_name, section_documents in sections:
                section_text = self._combine_section_text(
                    section_documents
                )

                if not section_text.strip():
                    continue

                section_chunks = self._split_section(section_text)

                for chunk_text in section_chunks:
                    if not chunk_text.strip():
                        continue

                    source_document = section_documents[0]

                    document_chunks.append(
                        Document(
                            page_content=chunk_text,
                            metadata={
                                **source_document.metadata,
                                "brochure_section": section_name,
                            },
                        )
                    )

            total_chunks = len(document_chunks)

            for chunk_index, chunk in enumerate(document_chunks):
                chunk.metadata = self.metadata_manager.prepare_metadata(
                    metadata=chunk.metadata,
                    brochure_section=chunk.metadata[
                        "brochure_section"
                    ],
                    chunk_index=chunk_index,
                    total_chunks=total_chunks,
                )

                all_chunks.append(chunk)

        return all_chunks

    def _group_documents(self, documents):
        groups = defaultdict(list)

        for document in documents:
            metadata = document.metadata

            group_key = (
                metadata.get("source_file"),
                metadata.get("brand"),
                metadata.get("model"),
            )

            groups[group_key].append(document)

        for group in groups.values():
            group.sort(
                key=lambda document: document.metadata.get("page", 0)
            )

        return groups

    def _build_sections(self, documents):
        sections = []
        current_section = "General"
        current_documents = []

        for document in documents:
            detected_section = self._detect_section(
                document.page_content
            )

            if detected_section:
                if current_documents:
                    sections.append(
                        (
                            current_section,
                            current_documents,
                        )
                    )

                current_section = detected_section
                current_documents = [document]

            else:
                current_documents.append(document)

        if current_documents:
            sections.append(
                (
                    current_section,
                    current_documents,
                )
            )

        return sections

    def _detect_section(self, text):
        lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]

        best_section = None
        best_score = 0

        for line in lines:
            if not self._is_heading_candidate(line):
                continue

            normalized_line = self._normalize_heading(line)

            for section, keywords in self.section_keywords.items():
                score = self._calculate_section_score(
                    normalized_line,
                    keywords,
                )

                if score > best_score:
                    best_score = score
                    best_section = section

        return best_section if best_score >= 5 else None

    def _is_heading_candidate(self, line):
        if not line:
            return False

        if len(line) > 100:
            return False

        words = line.split()

        if len(words) > 12:
            return False

        if line.endswith((".", ",", ";", ":")):
            return False

        if line[0].isdigit():
            return False

        return True

    def _normalize_heading(self, heading):
        return (
            " ".join(heading.lower().split())
            .replace("&", " and ")
            .replace("-", " ")
            .replace("/", " ")
        )

    def _calculate_section_score(self, heading, keywords):
        score = 0

        for keyword in keywords:
            normalized_keyword = self._normalize_heading(keyword)

            if heading == normalized_keyword:
                score = max(score, 5)

            elif (
                normalized_keyword in heading
                and len(normalized_keyword.split()) >= 2
            ):
                score = max(score, 5)

        return score

    def _combine_section_text(self, documents):
        page_texts = []

        for document in documents:
            text = document.page_content.strip()

            if text:
                page_texts.append(text)

        return "\n\n".join(page_texts)

    def _split_section(self, text):
        text = text.strip()

        if not text:
            return []

        if len(text) <= self.chunk_size:
            return [text]

        chunks = self.splitter.split_text(text)

        return [
            chunk.strip()
            for chunk in chunks
            if chunk.strip()
        ]