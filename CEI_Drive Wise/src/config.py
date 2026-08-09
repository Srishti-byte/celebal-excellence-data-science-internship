import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


# Project Paths
BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = BASE_DIR / "dataset"
CHROMA_DB_PATH = BASE_DIR / "chroma_db"
LOGS_PATH = BASE_DIR / "logs"


# Chunking Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100


# Retrieval
TOP_K = 5

SECTION_KEYWORDS = {
    "Engine and Performance": [
        "engine",
        "engine specifications",
        "engine and performance",
        "performance",
        "performance specifications",
        "powertrain",
        "powertrain and performance",
        "transmission",
        "drivetrain",
        "driving dynamics",
        "performance and dynamics",
    ],

    "Mileage and Fuel Efficiency": [
        "mileage",
        "mileage and fuel efficiency",
        "fuel efficiency",
        "fuel economy",
        "fuel consumption",
        "fuel efficiency and economy",
        "fuel economy and efficiency",
        "kmpl",
        "km/kg",
        "driving range",
        "range",
    ],

    "Safety": [
        "safety",
        "safety features",
        "safety feature",
        "advanced safety",
        "advanced safety features",
        "safety and security",
        "safety & security",
        "safety equipment",
        "security and safety",
        "active safety",
        "passive safety",
    ],

    "Dimensions": [
        "dimensions",
        "dimensions and capacities",
        "dimensions and weights",
        "vehicle dimensions",
        "length",
        "width",
        "height",
        "wheelbase",
        "ground clearance",
        "turning radius",
        "boot space",
        "cargo space",
        "capacity",
    ],

    "Interior and Comfort": [
        "interior",
        "interior and comfort",
        "interiors",
        "comfort",
        "comfort and convenience",
        "interior features",
        "interior features and comfort",
        "cabin",
        "cabin comfort",
        "seating",
        "seating comfort",
        "convenience",
        "comfort features",
    ],

    "Infotainment and Connectivity": [
        "infotainment",
        "infotainment and connectivity",
        "connectivity",
        "connectivity features",
        "connected features",
        "connected technology",
        "technology and connectivity",
        "entertainment",
        "audio and connectivity",
        "android auto",
        "apple carplay",
        "bluetooth connectivity",
        "touchscreen infotainment",
        "wireless connectivity",
        "wireless charger",
        "usb type-c",
    ],
}


# Embedding Configuration
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIMENSION = 384
EMBEDDING_DEVICE = "cpu"


# Vector Store Configuration
CHROMA_PERSIST_DIRECTORY = "chroma_db"
CHROMA_COLLECTION_NAME = "drive_wise_documents"
CHROMA_TEST_COLLECTION_NAME = "drive_wise_test"
RETRIEVAL_K = 5


# Reranking Configuration
RERANK_MODEL = "rerank-v4.0-fast"
RERANK_TOP_N = 3
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# LLM Configuration
GEMINI_MODEL = "gemini-3.6-flash"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# OPEN ROUTER CONFIGURATION
OPENROUTER_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")