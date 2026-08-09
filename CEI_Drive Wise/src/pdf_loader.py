from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from src.config import DATASET_PATH

class PDFLoader:
    def __init__(self, dataset_path=DATASET_PATH):
        self.dataset_path = Path(dataset_path)

    def load_documents(self):
        documents = []

        for brand_folder in sorted(self.dataset_path.iterdir()):
            if not brand_folder.is_dir():
                continue

            brand = brand_folder.name

            for pdf_file in sorted(brand_folder.glob("*.pdf")):
                model = pdf_file.stem.replace("_"," ")
                loader = PyPDFLoader(str(pdf_file))
                pdf_documents = loader.load()

                for document in pdf_documents:
                    document.metadata.update(
                        {
                            "brand": brand,
                            "model": model,
                            "source_file": pdf_file.name,
                            "document_type": "Brochure",
                            "document_version": "Unknown",
                        }
                    )


                documents.extend(pdf_documents)

        return documents