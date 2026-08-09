from copy import deepcopy


class MetadataManager:
    def __init__(self):
        self.document_metadata = {
            "brand",
            "model",
            "page",
            "page_label",
            "source_file",
            "document_type",
            "document_version",
        }

        self.chunk_metadata = {
            "brochure_section",
            "chunk_index",
            "total_chunks",
            "chunk_id",
        }

    def prepare_metadata(
        self,
        metadata,
        brochure_section=None,
        chunk_index=0,
        total_chunks=1,
    ):
        metadata = self._filter_document_metadata(metadata)

        metadata.update(
            {
                "brochure_section": brochure_section,
                "chunk_index": chunk_index,
                "total_chunks": total_chunks,
                "chunk_id": self.generate_chunk_id(
                    metadata,
                    chunk_index,
                ),
            }
        )

        return metadata

    def _filter_document_metadata(self, metadata):
        filtered_metadata = {}

        for key in self.document_metadata:
            if key in metadata:
                filtered_metadata[key] = metadata[key]

        return deepcopy(filtered_metadata)

    def generate_chunk_id(self, metadata, chunk_index):
        brand = metadata.get("brand", "Unknown")
        model = metadata.get("model", "Unknown")
        page = metadata.get("page", 0)

        return (
            f"{brand}_"
            f"{model}_"
            f"Page_{page}_"
            f"Chunk_{chunk_index}"
        )